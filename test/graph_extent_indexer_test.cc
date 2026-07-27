#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "tools/vamana_offline/graph_extent_indexer.hh"
#include "vamana/hot_graph.hh"

namespace {

constexpr u32 kDegree = 16;
constexpr u32 kCapacity = 18;
constexpr u32 kEntryBytes = 160;
constexpr u32 kNodeBytes = 64;
constexpr u32 kShards = 2;
constexpr u64 kGraphHeaderOffset = 256;
constexpr u64 kGraphOffset = 320;
constexpr u64 kBuildFingerprint = 0x123456789abcdef0ULL;
constexpr std::array<u64, kShards> kCounts{2, 3};
constexpr std::array<u64, kShards> kDynamicOffsets{640, 832};
constexpr std::array<u64, kShards> kShardFingerprints{
  0x1111222233334444ULL,
  0x5555666677778888ULL,
};

static_assert(
  kEntryBytes ==
  vamana::hot_graph::kTaggedNeighborBaseOffset +
    kCapacity * vamana::hot_graph::kCompactPointerBytes);

struct TemporaryDirectory {
  filepath_t path;

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

TemporaryDirectory make_temporary_directory() {
  const u64 nonce = static_cast<u64>(
    std::chrono::high_resolution_clock::now()
      .time_since_epoch().count());
  const filepath_t path =
    std::filesystem::temp_directory_path() /
    ("dvstor_graph_extent_test_" + std::to_string(nonce));
  std::filesystem::create_directories(path);
  return TemporaryDirectory{path};
}

void store_u32(byte_t* destination, u32 value) {
  std::memcpy(destination, &value, sizeof(value));
}

std::vector<byte_t> make_record(
    u32 stable_count, u32 provisional_count, u32 seed) {
  assert(stable_count <= kDegree);
  assert(provisional_count <= kCapacity - kDegree);
  assert(stable_count + provisional_count <= kCapacity);
  std::vector<byte_t> record(kEntryBytes, 0);
  record[0] = static_cast<byte_t>(stable_count);
  vamana::hot_graph::store_provisional_count(
    record.data(), static_cast<u8>(provisional_count));
  store_u32(record.data() + 4, 0);
  store_u32(record.data() + 8, 0);
  store_u32(record.data() + 12, 0);
  const u32 live = stable_count + provisional_count;
  for (u32 neighbor = 0; neighbor < live; ++neighbor) {
    const u32 shard = (seed + neighbor) % kShards;
    const u64 slot = (seed + neighbor) % kCounts[shard];
    const RemotePtr pointer{
      shard,
      vamana::hot_graph::kNodeBaseOffset + slot * kNodeBytes};
    std::memcpy(
      record.data() + vamana::hot_graph::neighbor_offset(neighbor),
      &pointer.raw_address, sizeof(pointer.raw_address));
  }
  const u16 checksum =
    vamana::hot_graph::checksum16(record.data(), record.size());
  vamana::hot_graph::store_u16_le(record.data() + 2, checksum);
  return record;
}

void write_shard(
    const filepath_t& prefix, u32 shard,
    const std::vector<std::pair<u32, u32>>& counts) {
  assert(counts.size() == kCounts[shard]);
  const filepath_t path =
    index_path::shard_file(prefix, shard + 1, kShards);
  std::fstream output(
    path, std::ios::binary | std::ios::in | std::ios::out |
            std::ios::trunc);
  output.seekp(
    static_cast<std::streamoff>(kDynamicOffsets[shard] - 1));
  output.put(0);
  output.seekp(0);
  output.write(
    reinterpret_cast<const char*>(&kDynamicOffsets[shard]), sizeof(u64));
  output.write(
    reinterpret_cast<const char*>(&kShardFingerprints[shard]),
    sizeof(u64));
  vamana::hot_graph::Header header;
  header.version = vamana::hot_graph::kVersion3;
  header.entry_bytes = kEntryBytes;
  header.max_degree = kDegree;
  header.compact_pointer_shard_bits = 1;
  header.entry_count = kCounts[shard];
  header.reserved0 = kDynamicOffsets[shard];
  header.reserved1 = kNodeBytes + kEntryBytes;
  header.reserved2 = kNodeBytes;
  output.seekp(static_cast<std::streamoff>(kGraphHeaderOffset));
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.seekp(static_cast<std::streamoff>(kGraphOffset));
  for (u32 slot = 0; slot < counts.size(); ++slot) {
    const auto record =
      make_record(counts[slot].first, counts[slot].second, shard + slot);
    output.write(
      reinterpret_cast<const char*>(record.data()),
      static_cast<std::streamsize>(record.size()));
  }
  output.flush();
  assert(output.good());
}

filepath_t write_index(const filepath_t& directory) {
  const filepath_t prefix = directory / "tiny";
  nlohmann::json metadata{
    {"schema_version", gpu_search::format::kMetadataSchemaVersion},
    {"node_layout", "plain"},
    {"storage_format", "vamana_tagged_v2"},
    {"remote_ptr_format", "tagged_inc24_shard6_off34x16_v1"},
    {"navigation_format", "opq_pq_graph_v1"},
    {"R", kDegree},
    {"num_memory_nodes", kShards},
    {"hot_graph_shard_bits", 1},
    {"node_size", kNodeBytes},
    {"hot_graph_entry_size", kEntryBytes},
    {"hot_graph_pointer_bytes",
     vamana::hot_graph::kCompactPointerBytes},
    {"num_vectors", kCounts[0] + kCounts[1]},
    {"index_build_fingerprint", kBuildFingerprint},
    {"hot_graph_entry_counts",
     std::vector<u64>(kCounts.begin(), kCounts.end())},
    {"hot_graph_header_offsets",
     std::vector<u64>(kShards, kGraphHeaderOffset)},
    {"hot_graph_offsets",
     std::vector<u64>(kShards, kGraphOffset)},
    {"hot_graph_dynamic_base_offsets",
     std::vector<u64>(
       kDynamicOffsets.begin(), kDynamicOffsets.end())},
    {"shard_build_fingerprints",
     std::vector<u64>(
       kShardFingerprints.begin(), kShardFingerprints.end())},
  };
  std::ofstream metadata_output(
    filepath_t{prefix.string() + ".meta.json"});
  metadata_output << metadata;
  metadata_output.close();
  assert(metadata_output.good());
  write_shard(prefix, 0, {{0, 0}, {1, 0}});
  write_shard(prefix, 1, {{8, 0}, {7, 2}, {16, 0}});
  return prefix;
}

void reseal_record(
    const filepath_t& prefix, u32 shard, u64 slot,
    const std::function<void(byte_t*)>& mutate) {
  const filepath_t path =
    index_path::shard_file(prefix, shard + 1, kShards);
  std::fstream file(
    path, std::ios::binary | std::ios::in | std::ios::out);
  std::vector<byte_t> record(kEntryBytes);
  const u64 offset = kGraphOffset + slot * kEntryBytes;
  file.seekg(static_cast<std::streamoff>(offset));
  file.read(
    reinterpret_cast<char*>(record.data()),
    static_cast<std::streamsize>(record.size()));
  assert(static_cast<size_t>(file.gcount()) == record.size());
  mutate(record.data());
  const u16 checksum =
    vamana::hot_graph::checksum16(record.data(), record.size());
  vamana::hot_graph::store_u16_le(record.data() + 2, checksum);
  file.seekp(static_cast<std::streamoff>(offset));
  file.write(
    reinterpret_cast<const char*>(record.data()),
    static_cast<std::streamsize>(record.size()));
  file.flush();
  assert(file.good());
}

}  // namespace

int main() {
  namespace format = gpu_search::format;
  namespace offline = tools::vamana_offline;

  assert(sizeof(format::GraphExtentHeader) == 128);
  assert(format::graph_extent_class(0) == 0);
  assert(format::graph_extent_class(1) == 1);
  assert(format::graph_extent_class(8) == 1);
  assert(format::graph_extent_class(9) == 2);
  assert(format::graph_extent_class(136) == 17);
  assert(format::graph_extent_read_bytes(0, 1104) == 16);
  assert(format::graph_extent_read_bytes(1, 1104) == 80);
  assert(format::graph_extent_read_bytes(17, 1104) == 1104);

  const TemporaryDirectory temporary = make_temporary_directory();
  const filepath_t prefix = write_index(temporary.path);
  assert(
    index_path::graph_extent_file(prefix) ==
    filepath_t{prefix.string() + ".gextent8"});

  offline::GraphExtentIndexOptions options{
    .index_prefix = prefix,
    .chunk_records = 2,
  };
  const auto built = offline::build_graph_extent_index(options);
  assert(built.output == index_path::graph_extent_file(prefix));
  assert(built.node_count == 5);
  assert(built.payload_bytes == 5);
  assert(built.graph_bytes_validated == 5 * kEntryBytes);
  assert(built.maximum_class == 2);

  format::GraphExtentHeader header;
  std::vector<u8> classes;
  std::string error;
  assert(format::read_graph_extent_sidecar(
    built.output, header, classes, &error));
  assert(classes == std::vector<u8>({0, 1, 1, 2, 2}));
  assert(header.num_nodes == 5);
  assert(header.num_shards == kShards);
  assert(header.graph_entry_bytes == kEntryBytes);
  assert(header.graph_entry_capacity == kCapacity);
  assert(header.build_fingerprint == kBuildFingerprint);
  assert(header.payload_checksum == built.payload_checksum);

  bool rejected = false;
  try {
    (void)offline::build_graph_extent_index(options);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);

  // Header identity fields are covered by the independent header checksum.
  {
    std::fstream sidecar(
      built.output, std::ios::binary | std::ios::in | std::ios::out);
    const u64 changed_fingerprint = header.build_fingerprint ^ 1u;
    sidecar.seekp(static_cast<std::streamoff>(
      offsetof(format::GraphExtentHeader, build_fingerprint)));
    sidecar.write(
      reinterpret_cast<const char*>(&changed_fingerprint),
      sizeof(changed_fingerprint));
  }
  assert(!format::read_graph_extent_sidecar(
    built.output, header, classes, &error));
  options.overwrite = true;
  (void)offline::build_graph_extent_index(options);

  // A payload mutation is rejected independently of the exact file envelope.
  {
    std::fstream sidecar(
      built.output, std::ios::binary | std::ios::in | std::ios::out);
    sidecar.seekp(
      static_cast<std::streamoff>(sizeof(format::GraphExtentHeader) + 1));
    const u8 changed = 7;
    sidecar.write(
      reinterpret_cast<const char*>(&changed), sizeof(changed));
  }
  assert(!format::read_graph_extent_sidecar(
    built.output, header, classes, &error));
  (void)offline::build_graph_extent_index(options);

  // Even a checksummed payload cannot encode a class beyond this graph
  // layout's ceil(capacity / quantum) range.
  assert(format::read_graph_extent_sidecar(
    built.output, header, classes, &error));
  classes[0] = 255;
  header.payload_checksum =
    format::checksum64(classes.data(), classes.size());
  {
    std::fstream sidecar(
      built.output, std::ios::binary | std::ios::in | std::ios::out);
    sidecar.seekp(
      static_cast<std::streamoff>(sizeof(format::GraphExtentHeader)));
    sidecar.write(
      reinterpret_cast<const char*>(classes.data()),
      static_cast<std::streamsize>(classes.size()));
    assert(format::write_graph_extent_header(sidecar, header, &error));
  }
  assert(!format::read_graph_extent_sidecar(
    built.output, header, classes, &error));
  (void)offline::build_graph_extent_index(options);

  // A checksum-valid graph record with a nonzero slot incarnation is not an
  // immutable base record and must not produce a static extent hint.
  reseal_record(prefix, 0, 0, [](byte_t* record) {
    store_u32(record + 8, 1);
  });
  auto invalid_options = options;
  invalid_options.output = temporary.path / "invalid.gextent8";
  invalid_options.overwrite = false;
  rejected = false;
  try {
    (void)offline::build_graph_extent_index(invalid_options);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  assert(!std::filesystem::exists(invalid_options.output));
  return 0;
}
