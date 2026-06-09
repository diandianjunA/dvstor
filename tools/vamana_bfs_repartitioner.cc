#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/index_path.hh"
#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "tools/vamana_offline/partitioning.hh"

namespace {

constexpr size_t kShardHeaderBytes = 16;
constexpr size_t kNodeHeaderBytes = 8;
constexpr size_t kNodeMetaBytes = 8;
constexpr size_t kNodeFixedPrefixBytes = kNodeHeaderBytes + kNodeMetaBytes;

struct Options {
  filepath_t input_prefix;
  filepath_t output_prefix;
  u32 memory_nodes{};
  u32 dim{};
  u32 R{};
  VectorDType vector_dtype{VectorDType::float32};
  bool vector_dtype_set{false};
  bool overwrite{false};
};

struct NodeFormat {
  size_t vector_offset{};
  size_t neighbors_offset{};
  size_t vector_bytes{};
  size_t neighbors_bytes{};
  size_t node_bytes{};
  size_t aligned_node_bytes{};
};

struct ShardInfo {
  filepath_t path;
  size_t file_size{};
  u64 free_ptr{};
  u64 medoid_raw{};
  size_t node_count{};
  size_t base_vertex{};
};

struct CrossShardStats {
  size_t total_edges{};
  size_t cross_edges{};

  double ratio() const {
    return total_edges == 0 ? 0.0 : static_cast<double>(cross_edges) / static_cast<double>(total_edges);
  }
};

[[noreturn]] void usage(const char* argv0, const std::string& error = {}) {
  if (!error.empty()) {
    std::cerr << "error: " << error << "\n\n";
  }
  std::cerr
    << "Usage: " << argv0 << " --input-prefix OLD_PREFIX --output-prefix NEW_PREFIX\n"
    << "       [--memory-nodes N --dim D --R R]\n"
    << "       [--vector-data-type auto|float32|uint8|int8]\n"
    << "       [--overwrite]\n\n"
    << "Repartitions existing fixed-size Vamana shards with multi-source BFS and rewrites neighbor RemotePtr values.\n"
    << "All layout parameters (memory-nodes, dim, R, vector-data-type) are auto-detected from metadata.\n";
  std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

std::string require_value(int& i, int argc, char** argv, const std::string& key) {
  if (i + 1 >= argc) {
    usage(argv[0], key + " requires a value");
  }
  return argv[++i];
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else if (arg == "--input-prefix") {
      options.input_prefix = require_value(i, argc, argv, arg);
    } else if (arg == "--output-prefix") {
      options.output_prefix = require_value(i, argc, argv, arg);
    } else if (arg == "--memory-nodes") {
      options.memory_nodes = static_cast<u32>(std::stoul(require_value(i, argc, argv, arg)));
    } else if (arg == "--dim") {
      options.dim = static_cast<u32>(std::stoul(require_value(i, argc, argv, arg)));
    } else if (arg == "--R") {
      options.R = static_cast<u32>(std::stoul(require_value(i, argc, argv, arg)));
    } else if (arg == "--vector-data-type") {
      const std::string value = require_value(i, argc, argv, arg);
      if (value == "auto") {
        options.vector_dtype_set = false;
      } else {
        options.vector_dtype = parse_vector_dtype(value);
        options.vector_dtype_set = true;
      }
    } else if (arg == "--overwrite") {
      options.overwrite = true;
    } else {
      usage(argv[0], "unknown argument: " + arg);
    }
  }
  if (options.input_prefix.empty()) {
    usage(argv[0], "--input-prefix is required");
  }
  if (options.output_prefix.empty()) {
    options.output_prefix = filepath_t(options.input_prefix.string() + "_bfs");
  }
  return options;
}

filepath_t metadata_path(const filepath_t& prefix) {
  return filepath_t(prefix.string() + ".meta.json");
}

nlohmann::json load_metadata(const filepath_t& prefix) {
  const filepath_t path = metadata_path(prefix);
  std::ifstream input(path);
  if (!input.good()) {
    throw std::runtime_error("failed to open metadata: " + path.string());
  }
  nlohmann::json metadata;
  input >> metadata;
  return metadata;
}

void fill_from_metadata(Options& options, const nlohmann::json& metadata) {
  if (options.memory_nodes == 0) {
    options.memory_nodes = metadata.value("num_memory_nodes", 0u);
  }
  if (options.dim == 0) {
    options.dim = metadata.value("dim", 0u);
  }
  if (options.R == 0) {
    options.R = metadata.value("R", 0u);
  }
  const VectorDType metadata_dtype = parse_vector_dtype(metadata.value("vector_data_type", std::string{"float32"}));
  if (options.vector_dtype_set && options.vector_dtype != metadata_dtype) {
    throw std::runtime_error("--vector-data-type does not match metadata vector_data_type");
  }
  if (!options.vector_dtype_set) {
    options.vector_dtype = metadata_dtype;
  }
}

size_t align8(size_t value) {
  return (value + 7) & ~static_cast<size_t>(7);
}

NodeFormat make_format(u32 dim, u32 R, VectorDType vector_dtype, const std::string& node_layout) {
  NodeFormat format;
  format.vector_bytes = vector_dtype_bytes(vector_dtype, dim);
  format.neighbors_bytes = static_cast<size_t>(R) * sizeof(u64);
  format.vector_offset = kNodeFixedPrefixBytes;
  format.neighbors_offset = format.vector_offset + format.vector_bytes;
  format.node_bytes = kNodeFixedPrefixBytes + format.vector_bytes + format.neighbors_bytes;
  if (node_layout == "rabitq") {
    format.node_bytes += sizeof(u64);  // rabitq_code
  }
  format.aligned_node_bytes = align8(format.node_bytes);
  return format;
}

u64 read_u64_le(const unsigned char* ptr) {
  u64 value = 0;
  std::memcpy(&value, ptr, sizeof(value));
  return value;
}

void read_exact(std::istream& input, void* dst, size_t bytes, const filepath_t& path) {
  input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
}

void write_u64(std::ostream& output, u64 value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

size_t checked_u32_vertex(size_t vertex) {
  if (vertex > static_cast<size_t>(std::numeric_limits<u32>::max())) {
    throw std::runtime_error("index has more than 2^32 vertices; repartitioner needs u32 vertex ids");
  }
  return vertex;
}

size_t vertex_from_old_ptr(RemotePtr ptr, const std::vector<ShardInfo>& shards, const NodeFormat& format) {
  const u32 shard = ptr.memory_node();
  if (shard >= shards.size()) {
    throw std::runtime_error("neighbor RemotePtr has invalid shard id: " + std::to_string(shard));
  }
  const auto& info = shards[shard];
  const u64 offset = ptr.byte_offset();
  if (offset < kShardHeaderBytes || offset + format.aligned_node_bytes > info.free_ptr) {
    throw std::runtime_error("neighbor RemotePtr offset out of shard bounds");
  }
  const u64 relative = offset - kShardHeaderBytes;
  if (relative % format.aligned_node_bytes != 0) {
    throw std::runtime_error("neighbor RemotePtr offset is not aligned to node size");
  }
  const size_t local = static_cast<size_t>(relative / format.aligned_node_bytes);
  if (local >= info.node_count) {
    throw std::runtime_error("neighbor RemotePtr maps past shard node count");
  }
  return info.base_vertex + local;
}

tools::vamana_offline::NodePlacement placement_for_vertex(
    size_t vertex, const vec<tools::vamana_offline::NodePlacement>& placements) {
  if (vertex >= placements.size()) {
    throw std::runtime_error("vertex id maps past placement table");
  }
  return placements[vertex];
}

RemotePtr new_ptr_for_vertex(size_t vertex, const vec<tools::vamana_offline::NodePlacement>& placements) {
  const auto& placement = placement_for_vertex(vertex, placements);
  return RemotePtr{placement.memory_node, placement.offset};
}

std::vector<ShardInfo> inspect_shards(const Options& options, const NodeFormat& format) {
  std::vector<ShardInfo> shards(options.memory_nodes);
  size_t base_vertex = 0;
  for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
    ShardInfo info;
    info.path = index_path::shard_file(options.input_prefix, shard + 1, options.memory_nodes);
    if (!std::filesystem::exists(info.path)) {
      throw std::runtime_error("input shard does not exist: " + info.path.string());
    }
    info.file_size = std::filesystem::file_size(info.path);
    if (info.file_size < kShardHeaderBytes) {
      throw std::runtime_error("input shard too small: " + info.path.string());
    }
    std::ifstream input(info.path, std::ios::binary);
    unsigned char header[kShardHeaderBytes]{};
    read_exact(input, header, sizeof(header), info.path);
    info.free_ptr = read_u64_le(header);
    info.medoid_raw = read_u64_le(header + sizeof(u64));
    if (info.free_ptr < kShardHeaderBytes || info.free_ptr > info.file_size) {
      throw std::runtime_error("invalid free_ptr in " + info.path.string());
    }
    const u64 payload = info.free_ptr - kShardHeaderBytes;
    if (payload % format.aligned_node_bytes != 0) {
      throw std::runtime_error("shard payload is not aligned to node size: " + info.path.string());
    }
    info.node_count = static_cast<size_t>(payload / format.aligned_node_bytes);
    info.base_vertex = base_vertex;
    base_vertex += info.node_count;
    checked_u32_vertex(base_vertex);
    shards[shard] = info;
  }
  return shards;
}

size_t total_nodes(const std::vector<ShardInfo>& shards) {
  size_t total = 0;
  for (const auto& shard : shards) total += shard.node_count;
  return total;
}

CrossShardStats build_neighbor_list(const std::vector<ShardInfo>& shards,
                                    const NodeFormat& format,
                                    u32 R,
                                    vec<vec<u32>>& neighbors,
                                    size_t& medoid_vertex) {
  CrossShardStats stats;
  const size_t total = total_nodes(shards);
  neighbors.resize(total);
  medoid_vertex = 0;
  bool medoid_found = false;

  std::vector<unsigned char> node(format.aligned_node_bytes);

  for (u32 shard_id = 0; shard_id < shards.size(); ++shard_id) {
    const auto& info = shards[shard_id];
    std::ifstream input(info.path, std::ios::binary);
    if (!input.good()) {
      throw std::runtime_error("failed to open shard: " + info.path.string());
    }
    input.seekg(static_cast<std::streamoff>(kShardHeaderBytes));
    for (size_t local = 0; local < info.node_count; ++local) {
      read_exact(input, node.data(), node.size(), info.path);
      const size_t source_vertex = info.base_vertex + local;

      // Check for medoid
      if (!medoid_found && shard_id == 0 && local == 0) {
        const u64 medoid_raw = shards[0].medoid_raw;
        const u64 first_offset = kShardHeaderBytes;
        if (medoid_raw == ((static_cast<u64>(0) << 32) | first_offset)) {
          medoid_vertex = source_vertex;
          medoid_found = true;
        }
      }

      const u8 edge_count = *reinterpret_cast<const u8*>(node.data() + kNodeHeaderBytes + sizeof(u32));
      const size_t active_edges = std::min<size_t>(edge_count, R);
      const auto* nbr_ptrs = reinterpret_cast<const u64*>(node.data() + format.neighbors_offset);

      auto& nbr_list = neighbors[source_vertex];
      nbr_list.reserve(active_edges);
      for (size_t j = 0; j < active_edges; ++j) {
        const RemotePtr old_neighbor{nbr_ptrs[j]};
        if (old_neighbor.is_null()) continue;
        const size_t neighbor_vertex = vertex_from_old_ptr(old_neighbor, shards, format);
        nbr_list.push_back(static_cast<u32>(neighbor_vertex));
        ++stats.total_edges;
        if (old_neighbor.memory_node() != shard_id) {
          ++stats.cross_edges;
        }
      }
    }
  }

  // Find medoid if not in first position: scan shard0 header
  if (!medoid_found) {
    const RemotePtr old_medoid{shards[0].medoid_raw};
    if (!old_medoid.is_null()) {
      medoid_vertex = vertex_from_old_ptr(old_medoid, shards, format);
      medoid_found = true;
    }
  }
  if (!medoid_found) {
    throw std::runtime_error("could not locate medoid in input shards");
  }

  return stats;
}

void ensure_output_paths_available(const Options& options) {
  if (options.input_prefix == options.output_prefix) {
    throw std::runtime_error("input-prefix and output-prefix must be different");
  }
  for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
    const filepath_t output = index_path::shard_file(options.output_prefix, shard + 1, options.memory_nodes);
    if (std::filesystem::exists(output) && !options.overwrite) {
      throw std::runtime_error("output shard exists, pass --overwrite: " + output.string());
    }
  }
  const filepath_t metadata = metadata_path(options.output_prefix);
  if (std::filesystem::exists(metadata) && !options.overwrite) {
    throw std::runtime_error("output metadata exists, pass --overwrite: " + metadata.string());
  }
}

std::vector<filepath_t> open_output_shards(const Options& options,
                                           std::vector<std::unique_ptr<std::ofstream>>& outputs) {
  const filepath_t output_dir = options.output_prefix.parent_path();
  if (!output_dir.empty()) {
    std::filesystem::create_directories(output_dir);
  }
  std::vector<filepath_t> temp_paths(options.memory_nodes);
  outputs.resize(options.memory_nodes);
  for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
    const filepath_t output = index_path::shard_file(options.output_prefix, shard + 1, options.memory_nodes);
    const filepath_t temp = filepath_t(output.string() + ".tmp");
    std::filesystem::remove(temp);
    auto stream = std::make_unique<std::ofstream>(temp, std::ios::binary | std::ios::trunc);
    if (!stream->good()) {
      throw std::runtime_error("failed to open output shard: " + temp.string());
    }
    unsigned char header[kShardHeaderBytes]{};
    stream->write(reinterpret_cast<const char*>(header), sizeof(header));
    if (!stream->good()) {
      throw std::runtime_error("failed to write output shard header: " + temp.string());
    }
    temp_paths[shard] = temp;
    outputs[shard] = std::move(stream);
  }
  return temp_paths;
}

CrossShardStats rewrite_shards(const std::vector<ShardInfo>& shards,
                               const NodeFormat& format,
                               u32 R,
                               const vec<tools::vamana_offline::NodePlacement>& placements,
                               const RemotePtr& new_medoid,
                               const Options& options) {
  ensure_output_paths_available(options);
  std::vector<std::unique_ptr<std::ofstream>> outputs;
  const std::vector<filepath_t> temp_paths = open_output_shards(options, outputs);

  vec<u64> shard_sizes(options.memory_nodes, kShardHeaderBytes);
  std::vector<unsigned char> node(format.aligned_node_bytes);
  CrossShardStats stats;

  for (u32 shard_id = 0; shard_id < shards.size(); ++shard_id) {
    const auto& info = shards[shard_id];
    std::ifstream input(info.path, std::ios::binary);
    if (!input.good()) {
      throw std::runtime_error("failed to open shard: " + info.path.string());
    }
    input.seekg(static_cast<std::streamoff>(kShardHeaderBytes));
    for (size_t local = 0; local < info.node_count; ++local) {
      read_exact(input, node.data(), node.size(), info.path);
      const size_t source_vertex = info.base_vertex + local;
      const auto& placement = placements[source_vertex];

      const u8 edge_count = *reinterpret_cast<const u8*>(node.data() + kNodeHeaderBytes + sizeof(u32));
      const size_t active_edges = std::min<size_t>(edge_count, R);
      auto* neighbors = reinterpret_cast<u64*>(node.data() + format.neighbors_offset);
      for (size_t j = 0; j < active_edges; ++j) {
        const RemotePtr old_neighbor{neighbors[j]};
        if (old_neighbor.is_null()) continue;
        const size_t neighbor_vertex = vertex_from_old_ptr(old_neighbor, shards, format);
        const RemotePtr rewritten = new_ptr_for_vertex(neighbor_vertex, placements);
        neighbors[j] = rewritten.raw_address;
        ++stats.total_edges;
        if (rewritten.memory_node() != placement.memory_node) {
          ++stats.cross_edges;
        }
      }

      auto& output = *outputs[placement.memory_node];
      output.seekp(static_cast<std::streamoff>(placement.offset));
      output.write(reinterpret_cast<const char*>(node.data()), static_cast<std::streamsize>(node.size()));
      if (!output.good()) {
        throw std::runtime_error("failed to write repartitioned node");
      }

      shard_sizes[placement.memory_node] =
        std::max<u64>(shard_sizes[placement.memory_node], placement.offset + format.aligned_node_bytes);
    }
  }

  for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
    auto& output = *outputs[shard];
    output.seekp(0);
    write_u64(output, shard_sizes[shard]);
    write_u64(output, shard == 0 ? new_medoid.raw_address : 0);
    output.close();
    if (!output.good()) {
      throw std::runtime_error("failed to close output shard: " + temp_paths[shard].string());
    }
  }

  for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
    const filepath_t output = index_path::shard_file(options.output_prefix, shard + 1, options.memory_nodes);
    if (std::filesystem::exists(output)) {
      std::filesystem::remove(output);
    }
    std::filesystem::rename(temp_paths[shard], output);
  }
  return stats;
}

void write_metadata(const Options& options,
                    nlohmann::json metadata,
                    const NodeFormat& format,
                    const tools::vamana_offline::PartitionStats& /*partition_stats*/,
                    const CrossShardStats& before_stats,
                    const CrossShardStats& after_stats,
                    RemotePtr new_medoid) {
  const filepath_t output = metadata_path(options.output_prefix);
  if (std::filesystem::exists(output) && !options.overwrite) {
    throw std::runtime_error("output metadata exists, pass --overwrite: " + output.string());
  }
  metadata["output_prefix"] = options.output_prefix.string();
  metadata["num_memory_nodes"] = options.memory_nodes;
  metadata["dim"] = options.dim;
  metadata["R"] = options.R;
  metadata["schema_version"] = 3;
  metadata["node_size"] = format.node_bytes;
  metadata["node_layout"] = metadata.value("node_layout", std::string{"standard"});
  metadata["vector_data_type"] = vector_dtype_name(options.vector_dtype);
  metadata["vector_component_size"] = vector_dtype_component_size(options.vector_dtype);
  metadata["vector_bytes"] = format.vector_bytes;
  metadata["medoid"] = {{"memory_node", new_medoid.memory_node()}, {"offset", new_medoid.byte_offset()}};
  metadata["partition_strategy"] = "bfs";
  metadata["partition_cross_shard_ratio"] = after_stats.ratio();
  metadata["partition_edge_cut"] = after_stats.cross_edges;
  metadata["partition_source_prefix"] = options.input_prefix.string();
  metadata["partition_before_cross_shard_ratio"] = before_stats.ratio();
  metadata["partition_before_cross_shard_edges"] = before_stats.cross_edges;
  metadata["partition_before_total_edges"] = before_stats.total_edges;
  metadata["partition_after_cross_shard_edges"] = after_stats.cross_edges;
  metadata["partition_after_total_edges"] = after_stats.total_edges;

  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  std::ofstream out(output, std::ios::trunc);
  if (!out.good()) {
    throw std::runtime_error("failed to open output metadata: " + output.string());
  }
  out << std::setw(2) << metadata << '\n';
}

void print_part_counts(const vec<size_t>& counts) {
  std::cerr << "partition node counts:";
  for (size_t count : counts) std::cerr << " " << count;
  std::cerr << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    using namespace tools::vamana_offline;

    Options options = parse_options(argc, argv);
    const nlohmann::json metadata = load_metadata(options.input_prefix);
    fill_from_metadata(options, metadata);

    if (options.memory_nodes == 0 || options.dim == 0 || options.R == 0) {
      throw std::runtime_error("missing layout parameters; provide metadata or --memory-nodes/--dim/--R");
    }

    const std::string node_layout = metadata.value("node_layout", std::string{"standard"});
    if (node_layout != "standard" && node_layout != "rabitq") {
      throw std::runtime_error("unsupported node_layout: " + node_layout);
    }

    const NodeFormat format = make_format(options.dim, options.R, options.vector_dtype, node_layout);
    const size_t metadata_node_size = metadata.value("node_size", format.node_bytes);
    if (metadata_node_size != format.node_bytes) {
      throw std::runtime_error(
          "metadata node_size " + std::to_string(metadata_node_size) +
          " does not match computed node_bytes " + std::to_string(format.node_bytes));
    }

    ensure_output_paths_available(options);
    std::cerr << "input prefix: " << options.input_prefix << "\n"
              << "output prefix: " << options.output_prefix << "\n"
              << "memory nodes: " << options.memory_nodes << "\n"
              << "dim=" << options.dim << " R=" << options.R
              << " node_layout=" << node_layout
              << " vector_data_type=" << vector_dtype_name(options.vector_dtype)
              << " vector_bytes=" << format.vector_bytes
              << " node_size=" << format.node_bytes
              << " aligned_node_size=" << format.aligned_node_bytes << "\n";

    std::vector<ShardInfo> shards = inspect_shards(options, format);
    const size_t n = total_nodes(shards);
    checked_u32_vertex(n);
    std::cerr << "input nodes: " << n << "\n";

    // Build neighbor list from shard files
    vec<vec<u32>> neighbors;
    size_t medoid_vertex = 0;
    CrossShardStats before_stats = build_neighbor_list(shards, format, options.R, neighbors, medoid_vertex);
    std::cerr << "before cross-shard ratio: " << before_stats.ratio()
              << " (" << before_stats.cross_edges << "/" << before_stats.total_edges << ")\n";

    // Run multi-source BFS partition
    PartitionStats partition_stats;
    vec<u32> parts = compute_bfs_partition(
        neighbors, options.memory_nodes, static_cast<u32>(medoid_vertex), &partition_stats);
    print_part_counts(partition_stats.part_node_counts);
    std::cerr << "BFS partition (multi-source) edges: input=" << partition_stats.input_edges
              << " edge_cut=" << partition_stats.edge_cut
              << " partition_cut_ratio=" << partition_stats.partition_cross_shard_ratio << "\n";

    // Generate placements and rewrite
    vec<NodePlacement> placements =
        assign_nodes_to_shards_from_partition(parts, options.memory_nodes, format.aligned_node_bytes);
    const RemotePtr new_medoid = new_ptr_for_vertex(medoid_vertex, placements);

    CrossShardStats after_stats = rewrite_shards(shards, format, options.R, placements, new_medoid, options);
    std::cerr << "after cross-shard ratio: " << after_stats.ratio()
              << " (" << after_stats.cross_edges << "/" << after_stats.total_edges << ")\n";

    write_metadata(options, metadata, format, partition_stats, before_stats, after_stats, new_medoid);

    std::cerr << "repartition finished: " << n << " nodes\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "vamana_bfs_repartitioner failed: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
