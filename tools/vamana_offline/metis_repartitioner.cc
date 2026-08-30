#include "tools/vamana_offline/metis_repartitioner.hh"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "gpu_search/pq_index.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "service/index_metadata.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/graph_extent_indexer.hh"
#include "tools/vamana_offline/partitioning.hh"
#include "tools/vamana_offline/pq_indexer.hh"
#include "tools/vamana_offline/shard_writer.hh"
#include "vamana/centroid_state.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_offline {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

constexpr u64 kMaximumMetadataBytes = 4ull << 20;
constexpr u32 kRepartitionerVersion = 1;
constexpr size_t kReadChunkRecords = 4096;
constexpr size_t kChecksumChunkBytes = 8ull << 20;

[[noreturn]] void fail(const str &message) {
  throw std::runtime_error("schema-16 METIS repartition: " + message);
}

u64 checked_add(u64 lhs, u64 rhs, const char *description) {
  if (lhs > std::numeric_limits<u64>::max() - rhs)
    fail(description);
  return lhs + rhs;
}

u64 checked_multiply(u64 lhs, u64 rhs, const char *description) {
  if (lhs != 0 && rhs > std::numeric_limits<u64>::max() / lhs) {
    fail(description);
  }
  return lhs * rhs;
}

str normalized_prefix(const filepath_t &path) {
  std::error_code error;
  const filepath_t absolute = fs::absolute(path, error);
  if (error)
    fail("cannot resolve path " + path.string() + ": " + error.message());
  return absolute.lexically_normal().string();
}

filepath_t metadata_path(const filepath_t &prefix) {
  return filepath_t{prefix.string() + ".meta.json"};
}

filepath_t graph_metadata_path(const filepath_t &prefix) {
  return filepath_t{prefix.string() + ".graph.meta.json"};
}

filepath_t plan_path(const filepath_t &prefix) {
  return filepath_t{prefix.string() + ".repartition.plan.json"};
}

void sync_file(const filepath_t &path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    fail("cannot open output for fsync " + path.string() + ": " +
         std::strerror(errno));
  }
  const int result = ::fsync(fd);
  const int saved_errno = errno;
  const int close_result = ::close(fd);
  if (result != 0) {
    fail("cannot fsync output " + path.string() + ": " +
         std::strerror(saved_errno));
  }
  if (close_result != 0) {
    fail("cannot close output after fsync " + path.string());
  }
}

void sync_directory(const filepath_t &directory) {
  const filepath_t path = directory.empty() ? filepath_t{"."} : directory;
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) {
    fail("cannot open output directory for fsync " + path.string() + ": " +
         std::strerror(errno));
  }
  const int result = ::fsync(fd);
  const int saved_errno = errno;
  const int close_result = ::close(fd);
  if (result != 0) {
    fail("cannot fsync output directory " + path.string() + ": " +
         std::strerror(saved_errno));
  }
  if (close_result != 0)
    fail("cannot close output directory after fsync");
}

json read_json_document(const filepath_t &path) {
  std::error_code error;
  const std::uintmax_t bytes = fs::file_size(path, error);
  if (error || bytes == 0 || bytes > kMaximumMetadataBytes) {
    fail("JSON file is missing, empty, or too large: " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input.good())
    fail("cannot open JSON file: " + path.string());
  str document(static_cast<size_t>(bytes), '\0');
  input.read(document.data(), static_cast<std::streamsize>(document.size()));
  if (input.gcount() != static_cast<std::streamsize>(document.size())) {
    fail("short read from JSON file: " + path.string());
  }
  char extra = 0;
  if (input.get(extra) || !input.eof()) {
    fail("JSON file changed while it was read: " + path.string());
  }
  json parsed = json::parse(document);
  if (!parsed.is_object())
    fail("JSON root is not an object: " + path.string());
  return parsed;
}

void write_json_atomic(const filepath_t &path, const json &document) {
  const filepath_t temporary{path.string() + ".repartition.tmp." +
                             std::to_string(::getpid())};
  std::error_code ignored;
  fs::remove(temporary, ignored);
  try {
    {
      std::ofstream output(temporary, std::ios::out | std::ios::trunc);
      if (!output.good())
        fail("cannot create temporary JSON: " + temporary.string());
      output << std::setw(2) << document << '\n';
      output.close();
      if (output.fail())
        fail("cannot finalize temporary JSON: " + temporary.string());
    }
    sync_file(temporary);
    if (read_json_document(temporary) != document) {
      fail("temporary JSON round-trip mismatch: " + temporary.string());
    }
    std::error_code rename_error;
    fs::rename(temporary, path, rename_error);
    if (rename_error) {
      fail("cannot publish JSON " + path.string() + ": " +
           rename_error.message());
    }
    sync_directory(path.parent_path());
  } catch (...) {
    fs::remove(temporary, ignored);
    throw;
  }
}

class PrefixLock {
public:
  explicit PrefixLock(const filepath_t &prefix) {
    path_ = filepath_t{prefix.string() + ".repartition.lock"};
    fd_ = ::open(path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (fd_ < 0)
      fail("cannot open output lock: " + path_.string());
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      const int saved_errno = errno;
      ::close(fd_);
      fd_ = -1;
      fail("another repartitioner holds " + path_.string() + ": " +
           std::strerror(saved_errno));
    }
  }

  ~PrefixLock() {
    if (fd_ >= 0) {
      (void)::flock(fd_, LOCK_UN);
      (void)::close(fd_);
    }
  }

  PrefixLock(const PrefixLock &) = delete;
  PrefixLock &operator=(const PrefixLock &) = delete;

private:
  filepath_t path_;
  int fd_{-1};
};

template <typename T> T required(const json &document, const char *name) {
  try {
    return document.at(name).get<T>();
  } catch (const std::exception &) {
    fail(str{"missing or invalid metadata field: "} + name);
  }
}

struct SourceContract {
  service::index_metadata::Metadata metadata;
  json document;
  filepath_t data_file;
  filepath_t model_file;
  gpu_search::pq::Model model;
  u32 partition_max_degree{};
  f64 partition_imbalance{};
  u32 beam_width{};
  f64 alpha{};
};

u64 checksum_stream(std::istream &input, u64 bytes) {
  vec<byte_t> buffer(kChecksumChunkBytes);
  u64 checksum = gpu_search::format::checksum64_initial();
  for (u64 consumed = 0; consumed < bytes;) {
    const size_t chunk =
        static_cast<size_t>(std::min<u64>(buffer.size(), bytes - consumed));
    input.read(reinterpret_cast<char *>(buffer.data()),
               static_cast<std::streamsize>(chunk));
    if (input.gcount() != static_cast<std::streamsize>(chunk)) {
      fail("short read while checksumming a sidecar payload");
    }
    checksum =
        gpu_search::format::checksum64_update(checksum, buffer.data(), chunk);
    consumed += chunk;
  }
  return checksum;
}

void validate_code_sidecars(const filepath_t &prefix,
                            const service::index_metadata::Metadata &metadata,
                            u64 model_checksum) {
  for (u32 shard = 0; shard < metadata.num_memory_nodes; ++shard) {
    const filepath_t path = index_path::navigation_code_file(
        prefix, shard + 1, metadata.num_memory_nodes,
        metadata.pq_subquantizers);
    gpu_search::format::CodeHeader header;
    str error;
    if (!gpu_search::format::read_code_header(path, header, &error)) {
      fail(error);
    }
    if (header.memory_node != shard ||
        header.entry_count != metadata.hot_graph_entry_counts[shard] ||
        header.code_bytes != metadata.navigation_code_bytes ||
        header.node_size != metadata.node_size ||
        header.vector_dtype != static_cast<u32>(metadata.vector_dtype) ||
        header.remote_offset !=
            metadata.navigation_code_remote_offsets[shard] ||
        header.payload_bytes != metadata.navigation_code_region_bytes[shard] ||
        header.model_checksum != model_checksum ||
        header.build_fingerprint != metadata.index_build_fingerprint ||
        header.shard_fingerprint != metadata.shard_build_fingerprints[shard]) {
      fail("PQ sidecar is not bound to source layout: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(sizeof(header)));
    if (!input.good() || checksum_stream(input, header.payload_bytes) !=
                             header.payload_checksum) {
      fail("PQ sidecar payload checksum mismatch: " + path.string());
    }
  }
}

void validate_extent_sidecar(
    const filepath_t &prefix,
    const service::index_metadata::Metadata &metadata) {
  gpu_search::format::GraphExtentHeader header;
  vec<u8> classes;
  str error;
  const filepath_t path = index_path::graph_extent_file(prefix);
  if (!gpu_search::format::read_graph_extent_sidecar(path, header, classes,
                                                     &error)) {
    fail(error);
  }
  if (header.num_nodes != metadata.num_vectors ||
      header.num_shards != metadata.num_memory_nodes ||
      header.graph_entry_bytes != metadata.hot_graph_entry_size ||
      header.graph_entry_capacity != VamanaNode::graph_entry_capacity() ||
      header.build_fingerprint != metadata.index_build_fingerprint) {
    fail("graph extent sidecar is not bound to index: " + path.string());
  }
}

SourceContract load_source_contract(const MetisRepartitionOptions &options) {
  SourceContract source;
  str error;
  if (!service::index_metadata::load_metadata(options.input_prefix,
                                              source.metadata, &error)) {
    fail(error);
  }
  if (source.metadata.schema_version !=
      gpu_search::format::kMetadataSchemaVersion) {
    fail("input must be a complete schema-16 index");
  }
  source.document = read_json_document(metadata_path(options.input_prefix));
  if (required<str>(source.document, "partition_strategy") != "balanced") {
    fail("input partition_strategy must be balanced");
  }
  if (normalized_prefix(
          required<filepath_t>(source.document, "output_prefix")) !=
      normalized_prefix(options.input_prefix)) {
    fail("source metadata output_prefix does not match --input-prefix");
  }
  source.partition_max_degree =
      options.partition_max_degree == 0
          ? required<u32>(source.document, "partition_max_degree")
          : options.partition_max_degree;
  source.partition_imbalance =
      options.partition_imbalance == 0.0
          ? required<f64>(source.document, "partition_imbalance")
          : options.partition_imbalance;
  source.beam_width = required<u32>(source.document, "beam_width_construction");
  source.alpha = required<f64>(source.document, "alpha");
  if (source.partition_max_degree == 0 ||
      source.partition_max_degree > source.metadata.R ||
      !std::isfinite(source.partition_imbalance) ||
      source.partition_imbalance < 1.0 || !std::isfinite(source.alpha) ||
      source.alpha < 1.0) {
    fail("METIS degree/imbalance or source alpha is invalid");
  }
  if (options.threads == 0 || options.threads > 32 ||
      options.pq_chunk_vectors == 0) {
    fail("--threads must be in [1,32] and --pq-chunk-vectors must be > 0");
  }
  source.data_file = options.data_path.empty()
                         ? required<filepath_t>(source.document, "data_file")
                         : options.data_path;
  if (source.data_file.empty())
    fail("source data_file is empty");
  source.model_file = options.reuse_model;
  if (source.model_file.empty() &&
      source.document.contains("navigation_model_file")) {
    source.model_file =
        source.document.at("navigation_model_file").get<filepath_t>();
  }
  if (source.model_file.empty()) {
    source.model_file = index_path::navigation_model_file(
        options.input_prefix, source.metadata.pq_subquantizers);
  }
  if (!fs::is_regular_file(source.model_file)) {
    fail("source OPQ/PQ model is missing: " + source.model_file.string());
  }

  VamanaNode::init_static_storage(source.metadata.dim, source.metadata.R,
                                  source.metadata.vector_dtype);
  str model_error;
  if (!gpu_search::pq::read_model(source.model_file, source.model,
                                  &model_error)) {
    fail(model_error);
  }
  if (source.model.dim != source.metadata.dim ||
      source.model.subquantizers != source.metadata.pq_subquantizers ||
      source.model.bits_per_code != 8 ||
      source.model.code_bytes() != source.metadata.navigation_code_bytes ||
      source.model.checksum() != source.metadata.navigation_model_checksum) {
    fail("source OPQ/PQ model does not match schema-16 metadata");
  }
  gpu_search::format::View view;
  if (!gpu_search::format::synthesize_distributed_view(options.input_prefix,
                                                       view, &error)) {
    fail(error);
  }
  validate_code_sidecars(options.input_prefix, source.metadata,
                         source.model.checksum());
  validate_extent_sidecar(options.input_prefix, source.metadata);
  return source;
}

struct LoadedIndex {
  Dataset dataset;
  VamanaGraph graph;
  u64 edge_count{};
};

void read_exact_at(std::ifstream &input, const filepath_t &path, u64 offset,
                   void *destination, size_t bytes, const char *description) {
  if (offset > static_cast<u64>(std::numeric_limits<std::streamoff>::max()) ||
      bytes >
          static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    fail(str{description} + " exceeds host I/O limits: " + path.string());
  }
  input.clear();
  input.seekg(static_cast<std::streamoff>(offset));
  if (!input.good()) {
    fail("cannot seek while reading " + str{description} + ": " +
         path.string());
  }
  input.read(static_cast<char *>(destination),
             static_cast<std::streamsize>(bytes));
  if (input.gcount() != static_cast<std::streamsize>(bytes)) {
    fail("short read while reading " + str{description} + ": " + path.string());
  }
}

template <typename T> T load_unaligned(const byte_t *source) {
  T value{};
  std::memcpy(&value, source, sizeof(value));
  return value;
}

vec<vec<node_t>>
load_physical_slot_ids(const filepath_t &prefix,
                       const service::index_metadata::Metadata &metadata) {
  const node_t missing = std::numeric_limits<node_t>::max();
  if (metadata.num_vectors > std::numeric_limits<size_t>::max()) {
    fail("source vector count exceeds host address space");
  }
  vec<vec<node_t>> slot_ids(metadata.num_memory_nodes);
  for (u32 shard = 0; shard < metadata.num_memory_nodes; ++shard) {
    if (metadata.hot_graph_entry_counts[shard] >
        std::numeric_limits<size_t>::max()) {
      fail("source shard count exceeds host address space");
    }
    slot_ids[shard].assign(
        static_cast<size_t>(metadata.hot_graph_entry_counts[shard]), missing);
  }
  vec<u8> seen(static_cast<size_t>(metadata.num_vectors), 0);
  u64 mapped = 0;
  const span<const u64> counts{metadata.hot_graph_entry_counts.data(),
                               metadata.hot_graph_entry_counts.size()};
  for (u32 owner = 0; owner < metadata.num_memory_nodes; ++owner) {
    const filepath_t path = index_path::owner_idmap_file(
        prefix, owner + 1, metadata.num_memory_nodes);
    std::error_code size_error;
    const std::uintmax_t file_bytes = fs::file_size(path, size_error);
    if (size_error || file_bytes > std::numeric_limits<u64>::max()) {
      fail("cannot inspect source idmap: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
      fail("cannot open source idmap: " + path.string());
    vamana::idmap::Header header;
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) {
      fail("source idmap header is truncated: " + path.string());
    }
    const vamana::idmap::ValidationContext context{
        .build_fingerprint = metadata.index_build_fingerprint,
        .owner_shard_fingerprint = metadata.shard_build_fingerprints[owner],
        .node_base_offset = vamana::hot_graph::kNodeBaseOffset,
        .owner_shard = owner,
        .shard_count = metadata.num_memory_nodes,
        .node_size = metadata.node_size,
        .id_offset = static_cast<u32>(VamanaNode::offset_id()),
        .generation_offset = static_cast<u32>(VamanaNode::offset_generation()),
        .slot_incarnation_offset = metadata.slot_incarnation_offset,
        .static_entry_counts = counts,
    };
    if (!vamana::idmap::valid_header(header, static_cast<u64>(file_bytes),
                                     context)) {
      fail("source idmap header is invalid: " + path.string());
    }
    const bool payload_ok = vamana::idmap::read_validated_payload(
        input, header, context, [&](const vamana::idmap::Entry &entry) {
          if (entry.id >= metadata.num_vectors || seen[entry.id] != 0) {
            return false;
          }
          const RemotePtr pointer{entry.rptr_raw};
          const u64 slot =
              (pointer.byte_offset() - vamana::hot_graph::kNodeBaseOffset) /
              metadata.node_size;
          if (pointer.memory_node() >= slot_ids.size() ||
              slot >= slot_ids[pointer.memory_node()].size() ||
              slot_ids[pointer.memory_node()][static_cast<size_t>(slot)] !=
                  missing) {
            return false;
          }
          seen[entry.id] = 1;
          slot_ids[pointer.memory_node()][static_cast<size_t>(slot)] = entry.id;
          ++mapped;
          return true;
        });
    if (!payload_ok)
      fail("source idmap payload is invalid: " + path.string());
  }
  if (mapped != metadata.num_vectors ||
      std::any_of(seen.begin(), seen.end(),
                  [](u8 value) { return value != 1; })) {
    fail("source idmaps do not provide a bijection over all base vectors");
  }
  for (const auto &slots : slot_ids) {
    if (std::find(slots.begin(), slots.end(), missing) != slots.end()) {
      fail("source idmaps leave an unmapped physical slot");
    }
  }
  return slot_ids;
}

void validate_source_graph_header(
    const vamana::hot_graph::Header &header,
    const service::index_metadata::Metadata &metadata, u32 shard) {
  if (header.magic != vamana::hot_graph::kMagic ||
      header.version != vamana::hot_graph::kVersion3 ||
      header.header_bytes != sizeof(header) ||
      header.entry_bytes != metadata.hot_graph_entry_size ||
      header.max_degree != metadata.R ||
      header.compact_pointer_bytes != vamana::hot_graph::kCompactPointerBytes ||
      header.compact_pointer_shard_bits != metadata.hot_graph_shard_bits ||
      header.flags != 0 ||
      header.entry_count != metadata.hot_graph_entry_counts[shard] ||
      header.node_base_offset != vamana::hot_graph::kNodeBaseOffset ||
      header.reserved0 != metadata.hot_graph_dynamic_base_offsets[shard] ||
      header.reserved1 != VamanaNode::dynamic_record_size() ||
      header.reserved2 != metadata.node_size) {
    fail("source compact graph header is inconsistent for shard " +
         std::to_string(shard + 1));
  }
}

LoadedIndex load_source_index(const MetisRepartitionOptions &options,
                              const SourceContract &source) {
  VamanaBuildConfig dataset_config;
  dataset_config.data_path = source.data_file;
  dataset_config.vector_data_type =
      vector_dtype_name(source.metadata.vector_dtype);
  dataset_config.max_vectors = static_cast<size_t>(source.metadata.num_vectors);
  Dataset dataset = read_dataset(dataset_config);
  if (dataset.size() != source.metadata.num_vectors ||
      dataset.dim != source.metadata.dim ||
      dataset.dtype != source.metadata.vector_dtype ||
      dataset.vector_bytes != source.metadata.vector_bytes) {
    fail("base dataset does not match source index metadata");
  }

  vec<vec<node_t>> slot_ids =
      load_physical_slot_ids(options.input_prefix, source.metadata);
  VamanaGraph graph;
  graph.init(dataset.size(), dataset.dim, source.metadata.R);
  graph.medoid = 0; // METIS placement does not consume the Vamana entry point.

  VamanaNode::configure_hot_graph(
      source.metadata.hot_graph_offsets, source.metadata.hot_graph_entry_counts,
      source.metadata.hot_graph_entry_size,
      source.metadata.hot_graph_shard_bits,
      source.metadata.hot_graph_dynamic_base_offsets,
      source.metadata.hot_graph_dynamic_record_bytes,
      source.metadata.hot_graph_dynamic_hot_offset,
      source.metadata.dynamic_navigation_code_offset,
      source.metadata.navigation_code_bytes);
  if (!VamanaNode::HAS_HOT_GRAPH) {
    fail("source schema-16 hot graph layout cannot be configured");
  }

  u64 total_edges = 0;
  u64 processed_nodes = 0;
  u64 next_progress = std::min<u64>(source.metadata.num_vectors, 1000000);
  vec<byte_t> decoded(VamanaNode::neighbor_read_size());
  vec<u32> neighbors;
  vec<u32> sorted;
  for (u32 shard = 0; shard < source.metadata.num_memory_nodes; ++shard) {
    const filepath_t path = index_path::shard_file(
        options.input_prefix, shard + 1, source.metadata.num_memory_nodes);
    std::error_code size_error;
    const std::uintmax_t file_bytes = fs::file_size(path, size_error);
    if (size_error ||
        file_bytes != source.metadata.hot_graph_dynamic_base_offsets[shard]) {
      fail("source shard has an unexpected size: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.good())
      fail("cannot open source shard: " + path.string());
    std::array<u64, 2> envelope{};
    read_exact_at(input, path, 0, envelope.data(), sizeof(envelope),
                  "source shard envelope");
    if (envelope[0] != file_bytes ||
        envelope[1] != source.metadata.shard_build_fingerprints[shard]) {
      fail("source shard envelope is invalid: " + path.string());
    }
    const u64 graph_offset = source.metadata.hot_graph_offsets[shard];
    if (graph_offset < sizeof(vamana::hot_graph::Header)) {
      fail("source graph offset underflows its header");
    }
    vamana::hot_graph::Header graph_header;
    read_exact_at(input, path, graph_offset - sizeof(graph_header),
                  &graph_header, sizeof(graph_header),
                  "source compact graph header");
    validate_source_graph_header(graph_header, source.metadata, shard);

    const u64 count = source.metadata.hot_graph_entry_counts[shard];
    for (u64 base = 0; base < count; base += kReadChunkRecords) {
      const size_t records =
          static_cast<size_t>(std::min<u64>(kReadChunkRecords, count - base));
      const u64 node_bytes_wide = checked_multiply(
          records, source.metadata.node_size, "node chunk size overflows");
      const u64 graph_bytes_wide =
          checked_multiply(records, source.metadata.hot_graph_entry_size,
                           "graph chunk size overflows");
      vec<byte_t> node_bytes(static_cast<size_t>(node_bytes_wide));
      vec<byte_t> graph_bytes(static_cast<size_t>(graph_bytes_wide));
      read_exact_at(
          input, path,
          checked_add(vamana::hot_graph::kNodeBaseOffset,
                      checked_multiply(base, source.metadata.node_size,
                                       "node offset overflows"),
                      "node offset overflows"),
          node_bytes.data(), node_bytes.size(), "source fixed-node chunk");
      read_exact_at(input, path,
                    checked_add(graph_offset,
                                checked_multiply(
                                    base, source.metadata.hot_graph_entry_size,
                                    "graph offset overflows"),
                                "graph offset overflows"),
                    graph_bytes.data(), graph_bytes.size(),
                    "source graph chunk");

      for (size_t local = 0; local < records; ++local) {
        const u64 slot = base + local;
        const node_t id = slot_ids[shard][static_cast<size_t>(slot)];
        const byte_t *node =
            node_bytes.data() + local * source.metadata.node_size;
        const u64 expected_header =
            VamanaNode::make_header(0, VamanaNode::HEADER_CENTROID_ACCOUNTED);
        if (load_unaligned<u64>(node) != expected_header ||
            load_unaligned<node_t>(node + VamanaNode::offset_id()) != id ||
            load_unaligned<u32>(node + VamanaNode::offset_generation()) != 0 ||
            load_unaligned<u32>(node + VamanaNode::offset_slot_incarnation()) !=
                0 ||
            std::memcmp(node + VamanaNode::offset_vector(),
                        dataset.raw_vector(id), dataset.vector_bytes) != 0) {
          fail("source fixed node does not match idmap/base dataset: shard=" +
               std::to_string(shard + 1) + " slot=" + std::to_string(slot));
        }

        const byte_t *compact =
            graph_bytes.data() + local * source.metadata.hot_graph_entry_size;
        if (compact[1] != 0 ||
            vamana::hot_graph::load_u32_le(compact + 4) != 0 ||
            !VamanaNode::decode_hot_graph_entry(compact, decoded.data(), 0)) {
          fail(
              "source immutable graph record is invalid or non-static: shard=" +
              std::to_string(shard + 1) + " slot=" + std::to_string(slot));
        }
        const u32 degree =
            decoded[VamanaNode::stable_neighbor_count_offset_in_read()];
        if (decoded[VamanaNode::provisional_neighbor_count_offset_in_read()] !=
            0) {
          fail("source base graph contains provisional neighbors");
        }
        neighbors.clear();
        neighbors.reserve(degree);
        const byte_t *encoded_neighbors =
            decoded.data() + VamanaNode::neighbor_payload_offset_in_read();
        for (u32 edge = 0; edge < degree; ++edge) {
          const RemotePtr pointer{load_unaligned<u64>(
              encoded_neighbors + edge * sizeof(RemotePtr))};
          if (!pointer.is_static() ||
              pointer.memory_node() >= slot_ids.size() ||
              pointer.byte_offset() < vamana::hot_graph::kNodeBaseOffset) {
            fail("source graph contains a non-static or out-of-range edge");
          }
          const u64 relative =
              pointer.byte_offset() - vamana::hot_graph::kNodeBaseOffset;
          if (relative % source.metadata.node_size != 0) {
            fail("source graph edge is not aligned to a fixed node");
          }
          const u64 neighbor_slot = relative / source.metadata.node_size;
          if (neighbor_slot >= slot_ids[pointer.memory_node()].size()) {
            fail("source graph edge references a slot outside its shard");
          }
          const node_t neighbor = slot_ids[pointer.memory_node()]
                                          [static_cast<size_t>(neighbor_slot)];
          if (neighbor == id)
            fail("source graph contains a self edge");
          neighbors.push_back(neighbor);
        }
        sorted = neighbors;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
          fail("source graph contains a duplicate stable edge");
        }
        graph.set_neighbors(id, neighbors);
        total_edges = checked_add(total_edges, degree,
                                  "source graph edge count overflows");
        ++processed_nodes;
        if (processed_nodes >= next_progress &&
            processed_nodes < source.metadata.num_vectors) {
          std::cerr << "[repartition][source] validated_nodes="
                    << processed_nodes << "/" << source.metadata.num_vectors
                    << " edges=" << total_edges << '\n';
          next_progress = std::min<u64>(source.metadata.num_vectors,
                                        next_progress + 1000000);
        }
      }
    }
    std::cerr << "validated source shard " << (shard + 1) << "/"
              << source.metadata.num_memory_nodes << " nodes=" << count
              << " cumulative_edges=" << total_edges << '\n';
  }
  return LoadedIndex{
      .dataset = std::move(dataset),
      .graph = std::move(graph),
      .edge_count = total_edges,
  };
}

json make_plan(const MetisRepartitionOptions &options,
               const SourceContract &source) {
  return {
      {"repartitioner_version", kRepartitionerVersion},
      {"source_prefix", normalized_prefix(options.input_prefix)},
      {"output_prefix", normalized_prefix(options.output_prefix)},
      {"source_schema_version", source.metadata.schema_version},
      {"source_build_fingerprint", source.metadata.index_build_fingerprint},
      {"data_file", normalized_prefix(source.data_file)},
      {"reuse_model", normalized_prefix(source.model_file)},
      {"navigation_model_checksum", source.model.checksum()},
      {"num_vectors", source.metadata.num_vectors},
      {"num_memory_nodes", source.metadata.num_memory_nodes},
      {"dim", source.metadata.dim},
      {"R", source.metadata.R},
      {"vector_data_type", vector_dtype_name(source.metadata.vector_dtype)},
      {"pq_subquantizers", source.metadata.pq_subquantizers},
      {"partition_strategy", "metis"},
      {"partition_max_degree", source.partition_max_degree},
      {"partition_imbalance", source.partition_imbalance},
  };
}

bool allowed_control_artifact(const str &suffix) {
  return suffix == ".repartition.lock" || suffix == ".repartition.plan.json" ||
         suffix == ".graph-build.lock" || suffix == ".build.lock";
}

void remove_stale_stage_temporaries(const filepath_t &prefix) {
  filepath_t parent = prefix.parent_path();
  if (parent.empty())
    parent = ".";
  const str base = prefix.filename().string();
  for (const auto &entry : fs::directory_iterator(parent)) {
    const str name = entry.path().filename().string();
    if (name.rfind(base, 0) != 0)
      continue;
    const str suffix = name.substr(base.size());
    const bool repartition_json =
        suffix.rfind(".repartition.plan.json.repartition.tmp.", 0) == 0 ||
        suffix.rfind(".meta.json.repartition.tmp.", 0) == 0 ||
        suffix.rfind(".graph.meta.json.repartition.tmp.", 0) == 0;
    const bool pq_temporary =
        suffix.find(".pq-indexer.tmp.") != str::npos &&
        (suffix.rfind(".pq", 0) == 0 || suffix.rfind("_node", 0) == 0 ||
         suffix.rfind(".meta.json", 0) == 0 ||
         suffix.rfind(".graph.meta.json", 0) == 0);
    const bool extent_temporary = suffix.rfind(".gextent8.tmp.", 0) == 0;
    if (!repartition_json && !pq_temporary && !extent_temporary)
      continue;
    std::error_code status_error;
    const fs::file_status status =
        fs::symlink_status(entry.path(), status_error);
    if (status_error)
      fail("cannot inspect stale stage temporary " + entry.path().string() +
           ": " + status_error.message());
    if (!fs::is_regular_file(status) && !fs::is_symlink(status)) {
      fail("refusing to remove non-file stage temporary: " +
           entry.path().string());
    }
    std::error_code remove_error;
    if (!fs::remove(entry.path(), remove_error) || remove_error) {
      fail("cannot remove stale stage temporary " + entry.path().string() +
           ": " + remove_error.message());
    }
    std::cerr << "[repartition][resume] removed stale temporary: "
              << entry.path() << '\n';
  }
}

void require_fresh_output_prefix(const filepath_t &prefix) {
  filepath_t parent = prefix.parent_path();
  if (parent.empty())
    parent = ".";
  std::error_code error;
  fs::create_directories(parent, error);
  if (error)
    fail("cannot create output directory: " + error.message());
  const str base = prefix.filename().string();
  for (const auto &entry : fs::directory_iterator(parent)) {
    const str name = entry.path().filename().string();
    if (name.rfind(base, 0) != 0)
      continue;
    const str suffix = name.substr(base.size());
    if (!allowed_control_artifact(suffix)) {
      fail("output prefix contains an unexpected partial artifact: " +
           entry.path().string());
    }
  }
}

void remove_uncommitted_graph_artifacts(const filepath_t &prefix, u32 shards) {
  vec<filepath_t> final_paths;
  final_paths.reserve(static_cast<size_t>(shards) * 3 + 1);
  final_paths.push_back(metadata_path(prefix));
  for (u32 shard = 0; shard < shards; ++shard) {
    final_paths.push_back(index_path::shard_file(prefix, shard + 1, shards));
    final_paths.push_back(
        index_path::owner_idmap_file(prefix, shard + 1, shards));
    final_paths.push_back(
        index_path::centroid_state_file(prefix, shard + 1, shards));
  }
  for (const filepath_t &final_path : final_paths) {
    for (const filepath_t &candidate :
         {final_path, filepath_t{final_path.string() + ".graph-build.tmp"}}) {
      std::error_code inspect_error;
      const fs::file_status status =
          fs::symlink_status(candidate, inspect_error);
      if (inspect_error == std::errc::no_such_file_or_directory)
        continue;
      if (inspect_error) {
        fail("cannot inspect uncommitted output artifact " +
             candidate.string() + ": " + inspect_error.message());
      }
      if (status.type() == fs::file_type::not_found)
        continue;
      std::error_code remove_error;
      if (!fs::remove(candidate, remove_error) || remove_error) {
        fail("cannot remove uncommitted output artifact " + candidate.string() +
             ": " + remove_error.message());
      }
      std::cerr << "[repartition][resume] removed uncommitted artifact: "
                << candidate << '\n';
    }
  }
}

u64 estimate_target_bytes(const MetisRepartitionOptions &options,
                          const SourceContract &source) {
  const u64 nodes = source.metadata.num_vectors;
  const u64 shards = source.metadata.num_memory_nodes;
  u64 bytes =
      checked_multiply(nodes,
                       checked_add(source.metadata.node_size,
                                   source.metadata.hot_graph_entry_size,
                                   "target graph record width overflows"),
                       "target graph bytes overflow");
  bytes = checked_add(bytes,
                      checked_multiply(nodes, sizeof(vamana::idmap::Entry),
                                       "target idmap bytes overflow"),
                      "target bytes overflow");
  bytes = checked_add(
      bytes,
      checked_multiply(shards,
                       sizeof(vamana::idmap::Header) +
                           sizeof(vamana::centroid_state::Header) +
                           static_cast<u64>(source.metadata.dim) * sizeof(f64) +
                           vamana::centroid_state::kMaxLiveEntries *
                               sizeof(vamana::centroid_state::Entry) +
                           512,
                       "target per-shard overhead overflows"),
      "target bytes overflow");
  if (!options.graph_only) {
    bytes = checked_add(bytes,
                        checked_multiply(nodes,
                                         source.metadata.navigation_code_bytes,
                                         "target PQ bytes overflow"),
                        "target bytes overflow");
    bytes = checked_add(bytes,
                        checked_multiply(shards,
                                         sizeof(gpu_search::format::CodeHeader),
                                         "target PQ header bytes overflow"),
                        "target bytes overflow");
    bytes = checked_add(
        bytes,
        checked_add(nodes, sizeof(gpu_search::format::GraphExtentHeader),
                    "target extent bytes overflow"),
        "target bytes overflow");
    std::error_code model_error;
    const std::uintmax_t model_bytes =
        fs::file_size(source.model_file, model_error);
    if (model_error || model_bytes > std::numeric_limits<u64>::max()) {
      fail("cannot inspect source PQ model size: " +
           source.model_file.string());
    }
    bytes = checked_add(bytes, static_cast<u64>(model_bytes),
                        "target model bytes overflow");
  }
  return checked_add(bytes, bytes / 20, "target disk safety margin overflows");
}

u64 estimate_peak_memory_bytes(const SourceContract &source) {
  const u64 nodes = source.metadata.num_vectors;
  const u64 partition_edges =
      checked_multiply(nodes, source.partition_max_degree,
                       "partition edge-count estimate overflows");
  u64 bytes = checked_multiply(nodes, source.metadata.vector_bytes,
                               "dataset memory estimate overflows");
  bytes = checked_add(
      bytes,
      checked_multiply(
          nodes, static_cast<u64>(source.metadata.R) * sizeof(u32) + sizeof(u8),
          "graph memory estimate overflows"),
      "memory estimate overflows");
  bytes = checked_add(bytes,
                      checked_multiply(nodes, sizeof(node_t) + sizeof(u8),
                                       "idmap memory estimate overflows"),
                      "memory estimate overflows");
  const u64 metis_vectors = checked_add(
      checked_multiply(partition_edges, 3 * sizeof(u64),
                       "METIS edge memory estimate overflows"),
      checked_multiply(nodes, 44, "METIS node memory estimate overflows"),
      "METIS memory estimate overflows");
  bytes = checked_add(bytes, metis_vectors, "memory estimate overflows");
  return checked_add(bytes, metis_vectors / 2,
                     "memory safety estimate overflows");
}

u64 available_memory_bytes() {
  std::ifstream input("/proc/meminfo");
  str key;
  u64 kib = 0;
  str unit;
  while (input >> key >> kib >> unit) {
    if (key == "MemAvailable:") {
      return checked_multiply(kib, 1024, "MemAvailable overflows");
    }
  }
  return 0;
}

void preflight_target_resources(const MetisRepartitionOptions &options,
                                const SourceContract &source) {
  filepath_t directory = options.output_prefix.parent_path();
  if (directory.empty())
    directory = ".";
  std::error_code space_error;
  const fs::space_info space = fs::space(directory, space_error);
  if (space_error) {
    fail("cannot inspect free space in " + directory.string() + ": " +
         space_error.message());
  }
  const u64 target_bytes = estimate_target_bytes(options, source);
  std::cerr << "[repartition][preflight] target_disk_required_gib="
            << std::fixed << std::setprecision(2)
            << static_cast<long double>(target_bytes) / (1ull << 30)
            << " available_gib="
            << static_cast<long double>(space.available) / (1ull << 30) << '\n';
  if (space.available < target_bytes) {
    fail("insufficient free disk for an independently committed METIS index");
  }

  const u64 memory_bytes = estimate_peak_memory_bytes(source);
  const u64 memory_available = available_memory_bytes();
  std::cerr << "[repartition][preflight] peak_memory_estimate_gib="
            << static_cast<long double>(memory_bytes) / (1ull << 30);
  if (memory_available != 0) {
    std::cerr << " available_gib="
              << static_cast<long double>(memory_available) / (1ull << 30);
  } else {
    std::cerr << " available_gib=unknown";
  }
  std::cerr << '\n';
  if (memory_available != 0 && memory_available < memory_bytes) {
    fail("insufficient MemAvailable for graph reconstruction and 64-bit METIS");
  }
}

void bind_graph_metadata(const filepath_t &output_prefix, const json &plan) {
  json metadata = read_json_document(metadata_path(output_prefix));
  if (metadata.value("schema_version", 0u) != 15u ||
      metadata.value("partition_strategy", str{}) != "metis") {
    fail("repartition graph stage is not schema-15 METIS");
  }
  metadata["repartitioner_version"] = kRepartitionerVersion;
  metadata["repartition_source_prefix"] = plan.at("source_prefix");
  metadata["repartition_source_schema_version"] =
      plan.at("source_schema_version");
  metadata["repartition_source_build_fingerprint"] =
      plan.at("source_build_fingerprint");
  metadata["repartition_source_model_checksum"] =
      plan.at("navigation_model_checksum");
  write_json_atomic(metadata_path(output_prefix), metadata);
  write_json_atomic(graph_metadata_path(output_prefix), metadata);
}

void validate_output_provenance(const json &metadata, const json &plan,
                                u32 expected_schema) {
  if (metadata.value("schema_version", 0u) != expected_schema ||
      metadata.value("partition_strategy", str{}) != "metis" ||
      normalized_prefix(metadata.value("output_prefix", filepath_t{})) !=
          plan.at("output_prefix").get<str>() ||
      metadata.value("num_vectors", u64{0}) !=
          plan.at("num_vectors").get<u64>() ||
      metadata.value("num_memory_nodes", u32{0}) !=
          plan.at("num_memory_nodes").get<u32>() ||
      metadata.value("dim", u32{0}) != plan.at("dim").get<u32>() ||
      metadata.value("R", u32{0}) != plan.at("R").get<u32>() ||
      metadata.value("partition_max_degree", u32{0}) !=
          plan.at("partition_max_degree").get<u32>() ||
      metadata.value("partition_imbalance", f64{0}) !=
          plan.at("partition_imbalance").get<f64>() ||
      metadata.value("repartitioner_version", u32{0}) !=
          kRepartitionerVersion ||
      metadata.value("repartition_source_prefix", str{}) !=
          plan.at("source_prefix").get<str>() ||
      metadata.value("repartition_source_build_fingerprint", u64{0}) !=
          plan.at("source_build_fingerprint").get<u64>() ||
      metadata.value("repartition_source_model_checksum", u64{0}) !=
          plan.at("navigation_model_checksum").get<u64>()) {
    fail("output metadata does not match the durable repartition plan");
  }
}

void validate_centroid_sidecars(
    const filepath_t &prefix,
    const service::index_metadata::Metadata &metadata) {
  for (u32 shard = 0; shard < metadata.num_memory_nodes; ++shard) {
    const filepath_t path = index_path::centroid_state_file(
        prefix, shard + 1, metadata.num_memory_nodes);
    std::error_code error;
    const std::uintmax_t file_bytes = fs::file_size(path, error);
    if (error || file_bytes > std::numeric_limits<size_t>::max()) {
      fail("cannot inspect output centroid: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    vamana::centroid_state::Header header;
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(header)) ||
        header.magic != vamana::centroid_state::kMagic ||
        header.version != vamana::centroid_state::kVersion ||
        header.header_bytes != sizeof(header) ||
        !vamana::centroid_state::valid_header_checksum(header) ||
        header.build_fingerprint != metadata.index_build_fingerprint ||
        header.shard_fingerprint != metadata.shard_build_fingerprints[shard] ||
        header.shard != shard ||
        header.shard_count != metadata.num_memory_nodes ||
        header.dim != metadata.dim || header.max_degree != metadata.R ||
        header.vector_count != metadata.hot_graph_entry_counts[shard] ||
        header.entry_count == 0 ||
        header.entry_count > vamana::centroid_state::kMaxLiveEntries ||
        header.vector_dtype != static_cast<u32>(metadata.vector_dtype) ||
        header.node_size != metadata.node_size ||
        header.vector_offset != metadata.vector_offset ||
        header.vector_bytes != metadata.vector_bytes ||
        header.slot_incarnation_offset != metadata.slot_incarnation_offset ||
        header.hot_graph_entry_size != metadata.hot_graph_entry_size ||
        header.hot_graph_shard_bits != metadata.hot_graph_shard_bits ||
        header.payload_bytes != vamana::centroid_state::payload_bytes(
                                    metadata.dim, header.entry_count) ||
        file_bytes != sizeof(header) + header.payload_bytes) {
      fail("output centroid header is invalid: " + path.string());
    }
    vec<byte_t> payload(static_cast<size_t>(header.payload_bytes));
    input.read(reinterpret_cast<char *>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
    if (input.gcount() != static_cast<std::streamsize>(payload.size()) ||
        vamana::centroid_state::checksum(payload) != header.payload_checksum) {
      fail("output centroid payload checksum mismatch: " + path.string());
    }
    for (u32 dimension = 0; dimension < metadata.dim; ++dimension) {
      if (!std::isfinite(
              load_unaligned<f64>(payload.data() + dimension * sizeof(f64)))) {
        fail("output centroid contains a non-finite component");
      }
    }
    const byte_t *entry_bytes = payload.data() + metadata.dim * sizeof(f64);
    for (u32 index = 0; index < header.entry_count; ++index) {
      const auto entry = load_unaligned<vamana::centroid_state::Entry>(
          entry_bytes + index * sizeof(vamana::centroid_state::Entry));
      const RemotePtr pointer{entry.remote_node};
      if (entry.generation != 0 || entry.reserved != 0 || pointer.is_null() ||
          !pointer.is_static() || pointer.memory_node() != shard ||
          pointer.byte_offset() < vamana::hot_graph::kNodeBaseOffset ||
          (pointer.byte_offset() - vamana::hot_graph::kNodeBaseOffset) %
                  metadata.node_size !=
              0 ||
          (pointer.byte_offset() - vamana::hot_graph::kNodeBaseOffset) /
                  metadata.node_size >=
              metadata.hot_graph_entry_counts[shard]) {
        fail("output centroid route entry is invalid: " + path.string());
      }
    }
  }
}

service::index_metadata::Metadata
validate_complete_output(const filepath_t &output_prefix, const json &plan) {
  service::index_metadata::Metadata metadata;
  str error;
  if (!service::index_metadata::load_metadata(output_prefix, metadata,
                                              &error)) {
    fail(error);
  }
  validate_output_provenance(read_json_document(metadata_path(output_prefix)),
                             plan, gpu_search::format::kMetadataSchemaVersion);
  gpu_search::format::View view;
  if (!gpu_search::format::synthesize_distributed_view(output_prefix, view,
                                                       &error)) {
    fail(error);
  }
  gpu_search::pq::Model model;
  if (!gpu_search::pq::read_model(index_path::navigation_model_file(
                                      output_prefix, metadata.pq_subquantizers),
                                  model, &error)) {
    fail(error);
  }
  if (model.checksum() != metadata.navigation_model_checksum ||
      model.checksum() != plan.at("navigation_model_checksum").get<u64>()) {
    fail("output model is not an exact copy of the selected source model");
  }
  validate_code_sidecars(output_prefix, metadata, model.checksum());
  validate_centroid_sidecars(output_prefix, metadata);
  validate_extent_sidecar(output_prefix, metadata);
  return metadata;
}

} // namespace

MetisRepartitionResult
repartition_schema16_index(const MetisRepartitionOptions &options) {
  if (options.input_prefix.empty() || options.output_prefix.empty()) {
    throw std::invalid_argument(
        "METIS repartition requires input and output prefixes");
  }
  if (normalized_prefix(options.input_prefix) ==
      normalized_prefix(options.output_prefix)) {
    fail("in-place repartition is forbidden");
  }
  if (!metis_partitioning_available() || metis_index_bits() < 64) {
    fail(metis_partitioning_available()
             ? "large-index repartition requires METIS with 64-bit idx_t"
             : metis_unavailable_reason());
  }

  const SourceContract source = load_source_contract(options);
  MetisRepartitionResult result{
      .node_count = source.metadata.num_vectors,
      .shards = source.metadata.num_memory_nodes,
      .source_build_fingerprint = source.metadata.index_build_fingerprint,
      .metadata_file = metadata_path(options.output_prefix),
  };
  if (options.validate_only) {
    LoadedIndex loaded = load_source_index(options, source);
    result.edge_count = loaded.edge_count;
    return result;
  }

  filepath_t output_directory = options.output_prefix.parent_path();
  if (output_directory.empty())
    output_directory = ".";
  std::error_code directory_error;
  fs::create_directories(output_directory, directory_error);
  if (directory_error) {
    fail("cannot create output directory: " + directory_error.message());
  }
  PrefixLock output_lock{options.output_prefix};
  remove_stale_stage_temporaries(options.output_prefix);
  const json plan = make_plan(options, source);
  if (fs::exists(plan_path(options.output_prefix))) {
    if (read_json_document(plan_path(options.output_prefix)) != plan) {
      fail("existing repartition plan does not match this invocation");
    }
    result.resumed = true;
  } else {
    require_fresh_output_prefix(options.output_prefix);
    write_json_atomic(plan_path(options.output_prefix), plan);
  }

  const filepath_t output_metadata = metadata_path(options.output_prefix);
  u32 output_schema = 0;
  if (fs::exists(output_metadata)) {
    const json existing = read_json_document(output_metadata);
    output_schema = existing.value("schema_version", 0u);
    if (output_schema == 15u) {
      if (!existing.contains("repartition_source_build_fingerprint")) {
        bind_graph_metadata(options.output_prefix, plan);
      }
      const json graph_stage = read_json_document(output_metadata);
      validate_output_provenance(graph_stage, plan, 15u);
      // This is a small recovery commit. Re-publishing it is cheap and closes
      // the interruption window between the main graph commit and its copy.
      write_json_atomic(graph_metadata_path(options.output_prefix),
                        graph_stage);
    } else if (output_schema == gpu_search::format::kMetadataSchemaVersion) {
      validate_output_provenance(existing, plan, output_schema);
    } else {
      fail("output metadata has an unsupported or corrupt schema");
    }
  } else {
    if (result.resumed) {
      remove_uncommitted_graph_artifacts(options.output_prefix,
                                         source.metadata.num_memory_nodes);
    }
    require_fresh_output_prefix(options.output_prefix);
    preflight_target_resources(options, source);
    LoadedIndex loaded = load_source_index(options, source);
    result.edge_count = loaded.edge_count;
    VamanaBuildConfig target;
    target.data_path = loaded.dataset.source_file;
    target.output_prefix = options.output_prefix;
    target.num_memory_nodes = source.metadata.num_memory_nodes;
    target.threads = options.threads;
    target.R = source.metadata.R;
    target.beam_width = source.beam_width;
    target.alpha = source.alpha;
    target.vector_data_type = vector_dtype_name(source.metadata.vector_dtype);
    target.partition_strategy = "metis";
    target.partition_max_degree = source.partition_max_degree;
    target.partition_imbalance = source.partition_imbalance;
    target.skip_sanity_check = true;
    target.max_vectors = static_cast<size_t>(source.metadata.num_vectors);
    validate_vamana_shard_capacity(loaded.dataset.size(), target);
    write_vamana_shards(loaded.graph, loaded.dataset, target,
                        options.output_prefix);
    bind_graph_metadata(options.output_prefix, plan);
    result.graph_written = true;
    output_schema = 15u;
  }

  if (options.graph_only) {
    const filepath_t committed_graph =
        graph_metadata_path(options.output_prefix);
    const json graph_metadata = read_json_document(
        fs::exists(committed_graph) ? committed_graph : output_metadata);
    validate_output_provenance(graph_metadata, plan, 15u);
    result.output_build_fingerprint =
        graph_metadata.at("index_build_fingerprint").get<u64>();
    return result;
  }

  if (output_schema == 15u) {
    PqIndexOptions pq;
    pq.index_prefix = options.output_prefix;
    pq.reuse_model = source.model_file;
    pq.subquantizers = source.metadata.pq_subquantizers;
    pq.chunk_vectors = options.pq_chunk_vectors;
    pq.threads = options.threads;
    pq.overwrite = true;
    (void)build_pq_index(pq);
    result.pq_built = true;
    output_schema = gpu_search::format::kMetadataSchemaVersion;
  }

  const json schema16 = read_json_document(output_metadata);
  validate_output_provenance(schema16, plan, output_schema);
  bool extent_valid = false;
  if (fs::exists(index_path::graph_extent_file(options.output_prefix))) {
    try {
      service::index_metadata::Metadata output_contract;
      str error;
      if (service::index_metadata::load_metadata(options.output_prefix,
                                                 output_contract, &error)) {
        validate_extent_sidecar(options.output_prefix, output_contract);
        extent_valid = true;
      }
    } catch (const std::exception &) {
      extent_valid = false;
    }
  }
  if (!extent_valid) {
    GraphExtentIndexOptions extent;
    extent.index_prefix = options.output_prefix;
    extent.overwrite = true;
    (void)build_graph_extent_index(extent);
    result.extent_built = true;
  }

  const auto output = validate_complete_output(options.output_prefix, plan);
  result.output_build_fingerprint = output.index_build_fingerprint;
  if (result.output_build_fingerprint == result.source_build_fingerprint) {
    fail("output unexpectedly reused the source build fingerprint");
  }
  return result;
}

} // namespace tools::vamana_offline
