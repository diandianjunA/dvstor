#include "tools/legacy_index/migrator.hh"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "common/index_path.hh"
#include "common/vector_dtype.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/anchor_index.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/vamana_node.hh"

namespace tools::legacy_index {
namespace {

struct Layout {
  u32 dim{};
  u32 degree{};
  u32 shards{};
  u32 shard_bits{};
  u32 old_node_bytes{};
  u32 new_node_bytes{};
  u32 legacy_payload_offset{};
  u32 vector_offset{};
  u32 vector_bytes{};
  u32 graph_entry_bytes{};
  u32 old_dynamic_record_bytes{};
  u32 new_dynamic_record_bytes{};
  u32 new_dynamic_hot_offset{};
  VectorDType dtype{VectorDType::float32};
  vec<u64> counts;
  vec<u64> old_graph_header_offsets;
  vec<u64> old_graph_offsets;
  vec<u64> old_dynamic_offsets;
  vec<u64> new_graph_header_offsets;
  vec<u64> new_graph_offsets;
  vec<u64> new_dynamic_offsets;
};

struct PendingOutput {
  filepath_t temporary;
  filepath_t final;
};

u64 align_up(u64 value, u64 alignment) {
  const u64 remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

void read_exact(std::istream& input, void* destination, size_t bytes,
                const filepath_t& path) {
  input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
}

void write_exact(std::ostream& output, const void* source, size_t bytes,
                 const filepath_t& path) {
  output.write(reinterpret_cast<const char*>(source), static_cast<std::streamsize>(bytes));
  if (!output.good()) throw std::runtime_error("short write to " + path.string());
}

filepath_t temporary_path(const filepath_t& path) {
  return filepath_t(path.string() + ".migration.tmp");
}

void prepare_output(const filepath_t& final, bool overwrite) {
  const filepath_t parent = final.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  if (!overwrite && std::filesystem::exists(final)) {
    throw std::runtime_error("migration output already exists: " + final.string());
  }
  const filepath_t temporary = temporary_path(final);
  if (std::filesystem::exists(temporary)) std::filesystem::remove(temporary);
}

Layout parse_layout(const nlohmann::json& metadata) {
  if (metadata.value("schema_version", 0u) != 13 ||
      metadata.value("node_layout", str{}) != "rabitq" ||
      metadata.value("storage_format", str{}) != "vamana_compact_v1" ||
      metadata.value("distance", str{"l2"}) != "l2") {
    throw std::runtime_error(
      "legacy migration requires a schema-13 compact L2 index with embedded RaBitQ");
  }
  Layout layout;
  layout.dim = metadata.at("dim").get<u32>();
  layout.degree = metadata.at("R").get<u32>();
  layout.shards = metadata.at("num_memory_nodes").get<u32>();
  layout.old_node_bytes = metadata.at("node_size").get<u32>();
  layout.vector_offset = metadata.at("vector_offset").get<u32>();
  layout.vector_bytes = metadata.at("vector_bytes").get<u32>();
  layout.dtype = parse_vector_dtype(metadata.at("vector_data_type").get<str>());
  layout.graph_entry_bytes = metadata.at("hot_graph_entry_size").get<u32>();
  layout.shard_bits = metadata.at("hot_graph_shard_bits").get<u32>();
  layout.old_dynamic_record_bytes =
    metadata.at("hot_graph_dynamic_record_bytes").get<u32>();
  layout.counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
  layout.old_graph_header_offsets =
    metadata.at("hot_graph_header_offsets").get<vec<u64>>();
  layout.old_graph_offsets = metadata.at("hot_graph_offsets").get<vec<u64>>();
  layout.old_dynamic_offsets =
    metadata.at("hot_graph_dynamic_base_offsets").get<vec<u64>>();

  VamanaNode::disable_hot_graph();
  VamanaNode::init_static_storage(layout.dim, layout.degree, layout.dtype);
  layout.new_node_bytes = static_cast<u32>(VamanaNode::total_size());
  layout.new_dynamic_hot_offset = layout.new_node_bytes;
  layout.new_dynamic_record_bytes = static_cast<u32>(
    VamanaNode::align_compact(layout.new_node_bytes + layout.graph_entry_bytes));

  layout.legacy_payload_offset = metadata.at("rabitq_offset").get<u32>();
  if (layout.shards == 0 || layout.counts.size() != layout.shards ||
      layout.old_graph_header_offsets.size() != layout.shards ||
      layout.old_graph_offsets.size() != layout.shards ||
      layout.old_dynamic_offsets.size() != layout.shards ||
      layout.vector_offset != VamanaNode::offset_vector() ||
      layout.vector_bytes != VamanaNode::vector_bytes() ||
      layout.legacy_payload_offset !=
        layout.vector_offset + metadata.at("vector_storage_bytes").get<u32>() ||
      layout.legacy_payload_offset > layout.new_node_bytes ||
      layout.old_node_bytes <= layout.new_node_bytes ||
      layout.graph_entry_bytes != vamana::hot_graph::entry_bytes(layout.degree) ||
      layout.shard_bits != vamana::hot_graph::shard_bits_for(layout.shards) ||
      metadata.value("hot_graph_pointer_bytes", 0u) !=
        vamana::hot_graph::kCompactPointerBytes) {
    throw std::runtime_error("legacy index metadata has an unsupported byte layout");
  }

  layout.new_graph_header_offsets.resize(layout.shards);
  layout.new_graph_offsets.resize(layout.shards);
  layout.new_dynamic_offsets.resize(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    if (layout.counts[shard] == 0) {
      throw std::runtime_error("legacy index contains an empty shard");
    }
    const u64 old_static_end = vamana::hot_graph::kNodeBaseOffset +
      layout.counts[shard] * layout.old_node_bytes;
    const u64 old_graph_end = layout.old_graph_offsets[shard] +
      layout.counts[shard] * layout.graph_entry_bytes;
    if (layout.old_graph_header_offsets[shard] < old_static_end ||
        layout.old_graph_offsets[shard] <
          layout.old_graph_header_offsets[shard] + sizeof(vamana::hot_graph::Header) ||
        layout.old_dynamic_offsets[shard] < old_graph_end) {
      throw std::runtime_error("legacy index contains overlapping shard regions");
    }
    const u64 new_static_end = vamana::hot_graph::kNodeBaseOffset +
      layout.counts[shard] * layout.new_node_bytes;
    layout.new_graph_header_offsets[shard] = align_up(new_static_end, 64);
    layout.new_graph_offsets[shard] = align_up(
      layout.new_graph_header_offsets[shard] + sizeof(vamana::hot_graph::Header), 64);
    layout.new_dynamic_offsets[shard] = align_up(
      layout.new_graph_offsets[shard] +
        layout.counts[shard] * layout.graph_entry_bytes,
      64);
  }
  return layout;
}

RemotePtr translate_pointer(RemotePtr pointer, const Layout& layout) {
  if (pointer.is_null()) return pointer;
  const u32 shard = pointer.memory_node();
  if (shard >= layout.shards) {
    throw std::runtime_error("legacy index contains a pointer to an invalid shard");
  }
  const u64 offset = pointer.byte_offset();
  if (offset < vamana::hot_graph::kNodeBaseOffset) {
    throw std::runtime_error("legacy index contains a pointer before the node region");
  }
  const u64 relative = offset - vamana::hot_graph::kNodeBaseOffset;
  if (relative % layout.old_node_bytes != 0) {
    throw std::runtime_error(
      "legacy index contains dynamic or unaligned pointers; persist a static snapshot first");
  }
  const u64 slot = relative / layout.old_node_bytes;
  if (slot >= layout.counts[shard]) {
    throw std::runtime_error(
      "legacy index contains dynamic pointers; static schema migration cannot preserve them");
  }
  return RemotePtr{shard,
    vamana::hot_graph::kNodeBaseOffset + slot * layout.new_node_bytes};
}

void migrate_shard(const MigrationOptions& options, const Layout& layout,
                   u32 shard, std::atomic<u64>& completed_nodes,
                   std::mutex& log_mutex) {
  const filepath_t source = index_path::shard_file(
    options.source_prefix, shard + 1, layout.shards);
  const filepath_t final = index_path::shard_file(
    options.output_prefix, shard + 1, layout.shards);
  const filepath_t temporary = temporary_path(final);
  prepare_output(final, options.overwrite);
  if (!std::filesystem::exists(source) ||
      std::filesystem::file_size(source) != layout.old_dynamic_offsets[shard]) {
    throw std::runtime_error(
      "legacy shard must be a static snapshot without appended dynamic records: " +
      source.string());
  }

  std::ifstream input(source, std::ios::binary);
  if (!input.good()) throw std::runtime_error("failed to open " + source.string());
  {
    std::ofstream create(temporary, std::ios::binary | std::ios::trunc);
    if (!create.good()) throw std::runtime_error("failed to create " + temporary.string());
  }
  std::filesystem::resize_file(temporary, layout.new_dynamic_offsets[shard]);
  std::fstream output(temporary,
    std::ios::binary | std::ios::in | std::ios::out);
  if (!output.good()) throw std::runtime_error("failed to open " + temporary.string());

  const u64 output_bytes = layout.new_dynamic_offsets[shard];
  write_exact(output, &output_bytes, sizeof(output_bytes), temporary);
  const u64 zero = 0;
  write_exact(output, &zero, sizeof(zero), temporary);

  const u32 chunk_nodes = std::max<u32>(1, options.chunk_nodes);
  vec<byte_t> old_nodes(static_cast<size_t>(chunk_nodes) * layout.old_node_bytes);
  vec<byte_t> new_nodes(static_cast<size_t>(chunk_nodes) * layout.new_node_bytes);
  for (u64 base = 0; base < layout.counts[shard]; base += chunk_nodes) {
    const u32 count = static_cast<u32>(
      std::min<u64>(chunk_nodes, layout.counts[shard] - base));
    const size_t source_bytes = static_cast<size_t>(count) * layout.old_node_bytes;
    input.seekg(static_cast<std::streamoff>(
      vamana::hot_graph::kNodeBaseOffset + base * layout.old_node_bytes));
    read_exact(input, old_nodes.data(), source_bytes, source);
    for (u32 index = 0; index < count; ++index) {
      std::memset(new_nodes.data() + static_cast<size_t>(index) * layout.new_node_bytes,
                  0, layout.new_node_bytes);
      std::memcpy(new_nodes.data() + static_cast<size_t>(index) * layout.new_node_bytes,
                  old_nodes.data() + static_cast<size_t>(index) * layout.old_node_bytes,
                  layout.legacy_payload_offset);
    }
    output.seekp(static_cast<std::streamoff>(
      vamana::hot_graph::kNodeBaseOffset + base * layout.new_node_bytes));
    write_exact(output, new_nodes.data(),
                static_cast<size_t>(count) * layout.new_node_bytes, temporary);
  }

  vamana::hot_graph::Header graph_header;
  graph_header.version = vamana::hot_graph::kVersion2;
  graph_header.entry_bytes = layout.graph_entry_bytes;
  graph_header.max_degree = layout.degree;
  graph_header.compact_pointer_shard_bits = layout.shard_bits;
  graph_header.entry_count = layout.counts[shard];
  graph_header.reserved0 = layout.new_dynamic_offsets[shard];
  graph_header.reserved1 = layout.new_dynamic_record_bytes;
  graph_header.reserved2 = layout.new_dynamic_hot_offset;
  output.seekp(static_cast<std::streamoff>(layout.new_graph_header_offsets[shard]));
  write_exact(output, &graph_header, sizeof(graph_header), temporary);

  const u32 graph_chunk = std::max<u32>(1,
    static_cast<u32>(std::min<u64>(chunk_nodes, 65536)));
  vec<byte_t> graph(static_cast<size_t>(graph_chunk) * layout.graph_entry_bytes);
  for (u64 base = 0; base < layout.counts[shard]; base += graph_chunk) {
    const u32 count = static_cast<u32>(
      std::min<u64>(graph_chunk, layout.counts[shard] - base));
    const size_t bytes = static_cast<size_t>(count) * layout.graph_entry_bytes;
    input.seekg(static_cast<std::streamoff>(
      layout.old_graph_offsets[shard] + base * layout.graph_entry_bytes));
    read_exact(input, graph.data(), bytes, source);
    for (u32 index = 0; index < count; ++index) {
      byte_t* entry = graph.data() + static_cast<size_t>(index) * layout.graph_entry_bytes;
      if (entry[0] > layout.degree ||
          vamana::hot_graph::load_u16_le(entry + 2) !=
            vamana::hot_graph::checksum16(entry, layout.graph_entry_bytes)) {
        throw std::runtime_error("legacy compact graph checksum mismatch in " + source.string());
      }
      for (u32 neighbor = 0; neighbor < layout.degree; ++neighbor) {
        byte_t* encoded = entry + vamana::hot_graph::neighbor_offset(neighbor);
        const RemotePtr old_pointer = vamana::hot_graph::decode_remote_ptr(
          encoded, layout.shard_bits);
        if (old_pointer.is_null()) continue;
        const RemotePtr new_pointer = translate_pointer(old_pointer, layout);
        if (!vamana::hot_graph::encode_remote_ptr(
              new_pointer, layout.shard_bits, encoded)) {
          throw std::runtime_error("translated graph pointer does not fit compact encoding");
        }
      }
      vamana::hot_graph::store_u16_le(
        entry + 2, vamana::hot_graph::checksum16(entry, layout.graph_entry_bytes));
    }
    output.seekp(static_cast<std::streamoff>(
      layout.new_graph_offsets[shard] + base * layout.graph_entry_bytes));
    write_exact(output, graph.data(), bytes, temporary);
    completed_nodes.fetch_add(count, std::memory_order_relaxed);
  }
  output.flush();
  if (!output.good()) throw std::runtime_error("failed to finalize " + temporary.string());
  std::lock_guard lock(log_mutex);
  std::cerr << "migrated shard " << (shard + 1) << "/" << layout.shards
            << " nodes=" << layout.counts[shard] << '\n';
}

void migrate_idmap(const MigrationOptions& options, const Layout& layout,
                   u32 owner) {
  const filepath_t source = index_path::owner_idmap_file(
    options.source_prefix, owner + 1, layout.shards);
  const filepath_t final = index_path::owner_idmap_file(
    options.output_prefix, owner + 1, layout.shards);
  const filepath_t temporary = temporary_path(final);
  prepare_output(final, options.overwrite);
  std::ifstream input(source, std::ios::binary);
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!input.good() || !output.good()) {
    throw std::runtime_error("failed to open ID-map sidecars for migration");
  }
  vamana::idmap::Header header;
  read_exact(input, &header, sizeof(header), source);
  if (header.magic != vamana::idmap::kMagic ||
      header.version != vamana::idmap::kVersion ||
      header.owner_shard != owner || header.shard_count != layout.shards ||
      std::filesystem::file_size(source) != sizeof(header) +
        header.entry_count * sizeof(vamana::idmap::Entry)) {
    throw std::runtime_error("invalid legacy ID-map sidecar: " + source.string());
  }
  write_exact(output, &header, sizeof(header), temporary);
  constexpr size_t kEntriesPerChunk = 1u << 20;
  vec<vamana::idmap::Entry> entries(kEntriesPerChunk);
  for (u64 base = 0; base < header.entry_count; base += kEntriesPerChunk) {
    const size_t count = static_cast<size_t>(
      std::min<u64>(kEntriesPerChunk, header.entry_count - base));
    read_exact(input, entries.data(), count * sizeof(entries.front()), source);
    for (size_t index = 0; index < count; ++index) {
      if (entries[index].rptr_raw != 0) {
        entries[index].rptr_raw = translate_pointer(
          RemotePtr{entries[index].rptr_raw}, layout).raw_address;
      }
    }
    write_exact(output, entries.data(), count * sizeof(entries.front()), temporary);
  }
}

void migrate_anchors(const MigrationOptions& options, const Layout& layout) {
  const filepath_t source = index_path::anchor_file(options.source_prefix);
  const filepath_t final = index_path::anchor_file(options.output_prefix);
  const filepath_t temporary = temporary_path(final);
  prepare_output(final, options.overwrite);
  std::ifstream input(source, std::ios::binary);
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!input.good() || !output.good()) {
    throw std::runtime_error("failed to open anchor sidecars for migration");
  }
  vamana::anchor::Header header;
  read_exact(input, &header, sizeof(header), source);
  if (header.magic != vamana::anchor::kMagic ||
      header.version != vamana::anchor::kVersion || header.dim != layout.dim ||
      header.shard_count != layout.shards ||
      header.vector_dtype != static_cast<u32>(layout.dtype) ||
      header.vector_bytes != layout.vector_bytes) {
    throw std::runtime_error("invalid legacy anchor sidecar: " + source.string());
  }
  write_exact(output, &header, sizeof(header), temporary);
  vec<byte_t> vector(layout.vector_bytes);
  u64 entries = 0;
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    vamana::anchor::ShardHeader shard_header;
    read_exact(input, &shard_header, sizeof(shard_header), source);
    if (shard_header.shard != shard ||
        entries + shard_header.anchor_count > header.total_anchors) {
      throw std::runtime_error("invalid legacy anchor shard: " + source.string());
    }
    write_exact(output, &shard_header, sizeof(shard_header), temporary);
    vec<f32> centroid(layout.dim);
    read_exact(input, centroid.data(), centroid.size() * sizeof(f32), source);
    write_exact(output, centroid.data(), centroid.size() * sizeof(f32), temporary);
    for (u32 index = 0; index < shard_header.anchor_count; ++index) {
      vamana::anchor::EntryHeader entry;
      read_exact(input, &entry, sizeof(entry), source);
      entry.rptr_raw = translate_pointer(RemotePtr{entry.rptr_raw}, layout).raw_address;
      read_exact(input, vector.data(), vector.size(), source);
      write_exact(output, &entry, sizeof(entry), temporary);
      write_exact(output, vector.data(), vector.size(), temporary);
      ++entries;
    }
  }
  if (entries != header.total_anchors || input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("legacy anchor sidecar has a size mismatch");
  }
}

void write_metadata(const MigrationOptions& options, const Layout& layout,
                    nlohmann::json metadata, RemotePtr medoid) {
  for (auto iterator = metadata.begin(); iterator != metadata.end();) {
    if (iterator.key().find("rabitq") != str::npos) iterator = metadata.erase(iterator);
    else ++iterator;
  }
  metadata["schema_version"] = 14;
  metadata["node_layout"] = "plain";
  metadata["output_prefix"] = options.output_prefix.string();
  metadata["node_size"] = layout.new_node_bytes;
  metadata["graph_hot_bytes"] = VamanaNode::graph_hot_bytes();
  metadata["vector_offset"] = layout.vector_offset;
  metadata.erase("neighbors_offset");
  metadata["vector_storage_bytes"] = VamanaNode::vector_storage_bytes();
  metadata["medoid"] = {
    {"memory_node", medoid.memory_node()}, {"offset", medoid.byte_offset()}};
  metadata["hot_graph_header_offsets"] = layout.new_graph_header_offsets;
  metadata["hot_graph_offsets"] = layout.new_graph_offsets;
  metadata["hot_graph_dynamic_base_offsets"] = layout.new_dynamic_offsets;
  metadata["hot_graph_dynamic_record_bytes"] = layout.new_dynamic_record_bytes;
  metadata["hot_graph_dynamic_hot_offset"] = layout.new_dynamic_hot_offset;
  metadata["allocation_size"] = layout.new_dynamic_record_bytes;
  metadata["navigation_quantizer"] = "";
  metadata["navigation_code_bytes"] = 0;
  metadata["pq_subquantizers"] = 0;
  metadata["pq_bits"] = 0;
  metadata["navigation_model_checksum"] = 0;
  metadata["navigation_format"] = "";
  metadata["navigation_entry_points"] = 0;
  metadata["navigation_code_remote_offsets"] = nlohmann::json::array();
  metadata["navigation_code_region_bytes"] = nlohmann::json::array();
  metadata["navigation_code_materialization"] = "";
  metadata["navigation_graph_source"] = "storage_compact_graph";
  metadata["navigation_execution"] = "";
  metadata["migration"] = {
    {"source_schema", 13},
    {"source_prefix", options.source_prefix.string()},
    {"method", "static_stride_compaction_v1"},
  };
  const filepath_t final{options.output_prefix.string() + ".meta.json"};
  const filepath_t temporary = temporary_path(final);
  prepare_output(final, options.overwrite);
  std::ofstream output(temporary, std::ios::trunc);
  output << std::setw(2) << metadata << '\n';
  if (!output.good()) throw std::runtime_error("failed to write migrated metadata");
}

void publish_outputs(const vec<PendingOutput>& outputs) {
  for (const PendingOutput& output : outputs) {
    std::filesystem::rename(output.temporary, output.final);
  }
}

}  // namespace

MigrationResult migrate_schema13_index(const MigrationOptions& options) {
  if (options.source_prefix.empty() || options.output_prefix.empty()) {
    throw std::invalid_argument("source and output index prefixes are required");
  }
  if (std::filesystem::absolute(options.source_prefix) ==
      std::filesystem::absolute(options.output_prefix)) {
    throw std::invalid_argument("legacy migration requires a distinct output prefix");
  }
  const filepath_t metadata_path{options.source_prefix.string() + ".meta.json"};
  std::ifstream metadata_input(metadata_path);
  if (!metadata_input.good()) {
    throw std::runtime_error("missing legacy metadata: " + metadata_path.string());
  }
  nlohmann::json metadata;
  metadata_input >> metadata;
  const Layout layout = parse_layout(metadata);
  const RemotePtr old_medoid{
    metadata.at("medoid").at("memory_node").get<u32>(),
    metadata.at("medoid").at("offset").get<u64>()};
  const RemotePtr new_medoid = translate_pointer(old_medoid, layout);

  const u64 total_nodes = std::accumulate(
    layout.counts.begin(), layout.counts.end(), u64{0});
  if (total_nodes != metadata.at("num_vectors").get<u64>()) {
    throw std::runtime_error("legacy metadata node count mismatch");
  }
  std::atomic<u64> completed_nodes{0};
  std::mutex log_mutex;
  const u32 requested_threads = options.io_threads == 0
    ? std::max(1u, std::thread::hardware_concurrency()) : options.io_threads;
  const u32 thread_count = std::min(layout.shards, requested_threads);
  std::atomic<u32> next_shard{0};
  std::exception_ptr worker_error;
  std::mutex error_mutex;
  vec<std::thread> workers;
  workers.reserve(thread_count);
  for (u32 thread = 0; thread < thread_count; ++thread) {
    workers.emplace_back([&]() {
      try {
        while (true) {
          const u32 shard = next_shard.fetch_add(1, std::memory_order_relaxed);
          if (shard >= layout.shards) break;
          migrate_shard(options, layout, shard, completed_nodes, log_mutex);
        }
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (worker_error == nullptr) worker_error = std::current_exception();
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  if (worker_error != nullptr) std::rethrow_exception(worker_error);

  for (u32 owner = 0; owner < layout.shards; ++owner) {
    migrate_idmap(options, layout, owner);
  }
  migrate_anchors(options, layout);
  write_metadata(options, layout, metadata, new_medoid);

  vec<PendingOutput> outputs;
  outputs.reserve(layout.shards * 2 + 2);
  u64 source_bytes = std::filesystem::file_size(metadata_path);
  u64 output_bytes = 0;
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    const filepath_t source = index_path::shard_file(
      options.source_prefix, shard + 1, layout.shards);
    const filepath_t output = index_path::shard_file(
      options.output_prefix, shard + 1, layout.shards);
    outputs.push_back({temporary_path(output), output});
    source_bytes += std::filesystem::file_size(source);
    output_bytes += std::filesystem::file_size(temporary_path(output));
    const filepath_t source_map = index_path::owner_idmap_file(
      options.source_prefix, shard + 1, layout.shards);
    const filepath_t output_map = index_path::owner_idmap_file(
      options.output_prefix, shard + 1, layout.shards);
    outputs.push_back({temporary_path(output_map), output_map});
    source_bytes += std::filesystem::file_size(source_map);
    output_bytes += std::filesystem::file_size(temporary_path(output_map));
  }
  const filepath_t source_anchor = index_path::anchor_file(options.source_prefix);
  const filepath_t output_anchor = index_path::anchor_file(options.output_prefix);
  outputs.push_back({temporary_path(output_anchor), output_anchor});
  source_bytes += std::filesystem::file_size(source_anchor);
  output_bytes += std::filesystem::file_size(temporary_path(output_anchor));
  const filepath_t output_metadata{options.output_prefix.string() + ".meta.json"};
  outputs.push_back({temporary_path(output_metadata), output_metadata});
  output_bytes += std::filesystem::file_size(temporary_path(output_metadata));
  publish_outputs(outputs);

  std::cerr << "legacy index migrated: nodes=" << total_nodes
            << " source_bytes=" << source_bytes
            << " output_bytes=" << output_bytes
            << " removed_bytes=" << (source_bytes > output_bytes
                 ? source_bytes - output_bytes : 0) << '\n';
  return {
    .output_prefix = options.output_prefix,
    .node_count = total_nodes,
    .source_bytes = source_bytes,
    .output_bytes = output_bytes,
  };
}

}  // namespace tools::legacy_index
