#include "tools/vamana_offline/shard_writer.hh"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <library/utils.hh>

#include "common/index_path.hh"
#include "common/vector_dtype.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/centroid_seed_policy.hh"
#include "vamana/centroid_state.hh"
#include "vamana/idmap.hh"
#include "vamana/storage_layout_resolver.hh"
#include "vamana/vamana_node.hh"
#include "tools/vamana_offline/partitioning.hh"
#include "tools/vamana_offline/progress.hh"

namespace tools::vamana_offline {

vec<NodePlacement> assign_nodes_to_shards(size_t num_vectors, u32 num_memory_nodes) {
  const size_t node_size = VamanaNode::total_size();
  const size_t aligned_size = (node_size + 7) & ~7ULL;
  return assign_nodes_to_shards_balanced(num_vectors, num_memory_nodes, aligned_size);
}

namespace {

struct PlacementResult {
  vec<NodePlacement> placements;
  PartitionStats stats;
  double cross_shard_ratio{0.0};
};

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

u64 make_build_fingerprint(const filepath_t& output_prefix,
                           size_t vector_count,
                           u32 dim,
                           u32 max_degree,
                           u32 shard_count) {
  const str path = output_prefix.string();
  u64 value = vamana::centroid_state::checksum(span<const byte_t>{
    reinterpret_cast<const byte_t*>(path.data()), path.size()});
  value = mix64(value ^ static_cast<u64>(vector_count));
  value = mix64(value ^ (static_cast<u64>(dim) << 32) ^ max_degree);
  value = mix64(value ^ shard_count ^ static_cast<u64>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  std::random_device entropy;
  value = mix64(value ^ (static_cast<u64>(entropy()) << 32) ^ entropy());
  return value == 0 ? 0x9e3779b97f4a7c15ULL : value;
}

size_t graph_input_edges(const VamanaGraph& graph) {
  size_t total = 0;
  for (size_t i = 0; i < graph.num_nodes; ++i) total += graph.degree(i);
  return total;
}

void validate_graph_for_serialization(const VamanaGraph& graph,
                                      const Dataset& dataset,
                                      const VamanaBuildConfig& config) {
  lib_assert(graph.num_nodes == dataset.size(),
             "graph and dataset vector counts do not match");
  lib_assert(graph.dim == dataset.dim,
             "graph and dataset dimensions do not match");
  lib_assert(graph.R == config.R,
             "graph degree does not match build configuration");
  lib_assert(graph.medoid < graph.num_nodes,
             "graph medoid is outside the graph");

  vec<u32> neighbors;
  vec<u32> sorted;
  for (size_t node = 0; node < graph.num_nodes; ++node) {
    graph.copy_neighbors(node, neighbors);
    lib_assert(neighbors.size() == graph.degree(node) &&
                   neighbors.size() <= graph.R,
               "graph node has an invalid stored degree");
    sorted = neighbors;
    std::sort(sorted.begin(), sorted.end());
    for (size_t index = 0; index < sorted.size(); ++index) {
      lib_assert(sorted[index] < graph.num_nodes,
                 "graph edge references a node outside the graph");
      lib_assert(sorted[index] != node,
                 "graph contains a self edge");
      lib_assert(index == 0 || sorted[index] != sorted[index - 1],
                 "graph contains a duplicate edge");
    }
  }
}

vec<u32> compute_bfs_partition_graph(const VamanaGraph& graph,
                                     u32 num_parts,
                                     u32 start_node,
                                     PartitionStats* stats) {
  if (num_parts == 0) {
    throw std::runtime_error("BFS partition part count must be > 0");
  }
  const size_t num_nodes = graph.num_nodes;
  vec<u32> parts(num_nodes, 0);
  if (stats != nullptr) {
    stats->input_edges = graph_input_edges(graph);
    stats->unique_edges = stats->input_edges;
    stats->edge_cut = 0;
    stats->partition_cross_shard_ratio = 0.0;
    stats->part_node_counts.assign(num_parts, 0);
  }
  if (num_nodes == 0) return parts;
  if (num_parts == 1) {
    if (stats != nullptr) stats->part_node_counts[0] = num_nodes;
    return parts;
  }

  vec<u32> nbrs;
  const u32 safe_start = (start_node < num_nodes) ? start_node : 0;

  // ── Step 1: Select seeds via farthest-point heuristic (k-center greedy) ──
  vec<u32> seeds;
  seeds.reserve(num_parts);
  seeds.push_back(safe_start);

  const u32 kUnreachable = std::numeric_limits<u32>::max();
  vec<u32> dist(num_nodes, kUnreachable);

  auto bfs_update_distances = [&](u32 source) {
    std::deque<u32> q;
    dist[source] = 0;
    q.push_back(source);
    while (!q.empty()) {
      const u32 node = q.front();
      q.pop_front();
      const u32 ndist = dist[node] + 1;
      graph.copy_neighbors(node, nbrs);
      for (u32 nbr : nbrs) {
        if (nbr < num_nodes && dist[nbr] > ndist) {
          dist[nbr] = ndist;
          q.push_back(nbr);
        }
      }
    }
  };

  // Initial BFS from medoid
  bfs_update_distances(safe_start);

  // Greedy farthest-point selection for remaining seeds
  while (seeds.size() < num_parts) {
    u32 farthest = 0;
    u32 max_dist = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
      if (dist[i] != kUnreachable && dist[i] > max_dist) {
        max_dist = dist[i];
        farthest = static_cast<u32>(i);
      }
    }
    seeds.push_back(farthest);
    // Incremental BFS from new seed — only updates nodes whose distance improves
    bfs_update_distances(farthest);
  }

  // ── Step 2: Multi-source BFS with load balancing ──
  const u32 kUnassigned = std::numeric_limits<u32>::max();
  std::fill(parts.begin(), parts.end(), kUnassigned);

  vec<byte_t> visited(num_nodes, 0);
  vec<u32> shard_size(num_parts, 0);
  const size_t target = num_nodes / num_parts;

  // Single FIFO queue: each entry is (node, preferred_shard)
  std::deque<std::pair<u32, u32>> queue;
  for (u32 s = 0; s < num_parts; ++s) {
    queue.emplace_back(seeds[s], s);
    visited[seeds[s]] = 1;
  }

  while (!queue.empty()) {
    const auto [node, pref_shard] = queue.front();
    queue.pop_front();

    if (parts[node] != kUnassigned) continue;

    // Load balance: if preferred shard is full, redirect to smallest shard
    u32 target_shard = pref_shard;
    if (shard_size[pref_shard] >= target) {
      target_shard = static_cast<u32>(
          std::min_element(shard_size.begin(), shard_size.end()) - shard_size.begin());
    }

    parts[node] = target_shard;
    ++shard_size[target_shard];

    // Tag neighbors with the assigned shard for territory continuity
    graph.copy_neighbors(node, nbrs);
    for (u32 nbr : nbrs) {
      if (nbr < num_nodes && !visited[nbr]) {
        visited[nbr] = 1;
        queue.emplace_back(nbr, target_shard);
      }
    }
  }

  // ── Step 3: Assign leftovers (isolated components never reached by BFS) ──
  for (size_t i = 0; i < num_nodes; ++i) {
    if (parts[i] == kUnassigned) {
      const u32 s = static_cast<u32>(
          std::min_element(shard_size.begin(), shard_size.end()) - shard_size.begin());
      parts[i] = s;
      ++shard_size[s];
    }
  }

  // ── Step 4: Compute stats ──
  if (stats != nullptr) {
    for (u32 s = 0; s < num_parts; ++s) {
      stats->part_node_counts[s] = shard_size[s];
    }
    size_t total_edges = 0;
    size_t cut_edges = 0;
    for (size_t node = 0; node < num_nodes; ++node) {
      graph.copy_neighbors(node, nbrs);
      for (u32 neighbor : nbrs) {
        if (neighbor >= num_nodes) continue;
        ++total_edges;
        if (parts[node] != parts[neighbor]) ++cut_edges;
      }
    }
    stats->edge_cut = cut_edges;
    stats->partition_cross_shard_ratio =
        total_edges == 0 ? 0.0 : static_cast<double>(cut_edges) / static_cast<double>(total_edges);
  }
  return parts;
}

double compute_cross_shard_ratio_graph(const VamanaGraph& graph, const vec<NodePlacement>& placements) {
  size_t total_edges = 0;
  size_t cross_edges = 0;
  vec<u32> nbrs;
  for (size_t i = 0; i < graph.num_nodes; ++i) {
    const u32 source_shard = placements[i].memory_node;
    graph.copy_neighbors(i, nbrs);
    for (u32 neighbor : nbrs) {
      if (neighbor >= placements.size()) continue;
      ++total_edges;
      if (placements[neighbor].memory_node != source_shard) ++cross_edges;
    }
  }
  return total_edges == 0 ? 0.0 : static_cast<double>(cross_edges) / static_cast<double>(total_edges);
}

PlacementResult place_nodes(const VamanaGraph& graph,
                            const VamanaBuildConfig& config,
                            size_t aligned_size) {
  PlacementResult result;
  if (config.partition_strategy == "metis") {
    vec<u64> edges;
    const size_t reserve_edges =
      std::min<size_t>(graph.num_nodes * static_cast<size_t>(config.partition_max_degree),
                       static_cast<size_t>(std::numeric_limits<u32>::max()));
    edges.reserve(reserve_edges);
    vec<u32> nbrs;
    for (size_t i = 0; i < graph.num_nodes; ++i) {
      graph.copy_neighbors(i, nbrs);
      append_partition_edges(static_cast<u32>(i), graph.num_nodes, nbrs,
                             config.partition_max_degree, edges);
    }
    PartitionOptions options;
    options.num_parts = config.num_memory_nodes;
    options.max_degree = config.partition_max_degree;
    options.imbalance = config.partition_imbalance;
    vec<u32> parts = compute_metis_partition(graph.num_nodes, edges, options, &result.stats);
    result.placements = assign_nodes_to_shards_from_partition(parts, config.num_memory_nodes, aligned_size);
  } else if (config.partition_strategy == "bfs") {
    const u32 start_node = graph.medoid < graph.num_nodes ? static_cast<u32>(graph.medoid) : 0;
    vec<u32> parts = compute_bfs_partition_graph(graph, config.num_memory_nodes, start_node, &result.stats);
    result.placements = assign_nodes_to_shards_from_partition(parts, config.num_memory_nodes, aligned_size);
  } else {
    result.placements = assign_nodes_to_shards_balanced(graph.num_nodes, config.num_memory_nodes, aligned_size);
    result.stats.part_node_counts.assign(config.num_memory_nodes, 0);
    for (const auto& placement : result.placements) ++result.stats.part_node_counts[placement.memory_node];
  }
  result.cross_shard_ratio = compute_cross_shard_ratio_graph(graph, result.placements);
  return result;
}

void print_partition_stats(const VamanaBuildConfig& config, const PlacementResult& result) {
  std::cerr << "partition strategy: " << config.partition_strategy
            << " max_degree=" << config.partition_max_degree
            << " imbalance=" << config.partition_imbalance << "\n";
  if (config.partition_strategy == "metis") {
    std::cerr << "METIS partition edges: input=" << result.stats.input_edges
              << " unique=" << result.stats.unique_edges
              << " edge_cut=" << result.stats.edge_cut
              << " partition_cut_ratio=" << result.stats.partition_cross_shard_ratio << "\n";
  } else if (config.partition_strategy == "bfs") {
    std::cerr << "BFS partition (multi-source) edges: input=" << result.stats.input_edges
              << " edge_cut=" << result.stats.edge_cut
              << " partition_cut_ratio=" << result.stats.partition_cross_shard_ratio << "\n";
  }
  std::cerr << "partition node counts:";
  for (size_t count : result.stats.part_node_counts) std::cerr << " " << count;
  std::cerr << "\nactive neighbor cross-shard ratio: " << result.cross_shard_ratio << "\n";
}

struct TemporaryOutputs {
  vec<filepath_t> paths;

  ~TemporaryOutputs() {
    for (const filepath_t& path : paths) {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  }

  filepath_t add(const filepath_t& final_path) {
    const filepath_t temporary{final_path.string() + ".graph-build.tmp"};
    std::error_code error;
    const bool exists = std::filesystem::exists(temporary, error);
    lib_assert(!error,
               "cannot inspect temporary graph output: " +
                 temporary.string() + ": " + error.message());
    if (exists) {
      const bool removed = std::filesystem::remove(temporary, error);
      lib_assert(removed && !error,
                 "cannot remove stale temporary graph output: " +
                   temporary.string() + ": " + error.message());
    }
    paths.push_back(temporary);
    return temporary;
  }
};

struct BuildLock {
  filepath_t path;
  int fd{-1};

  explicit BuildLock(filepath_t lock_path) : path(std::move(lock_path)) {
    fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    lib_assert(fd >= 0,
               "cannot create graph publication lock: " + path.string() +
                 ": " + std::strerror(errno));
    const int lock_result = ::flock(fd, LOCK_EX | LOCK_NB);
    const int lock_error = errno;
    if (lock_result != 0) {
      ::close(fd);
      fd = -1;
    }
    lib_assert(lock_result == 0,
               "another graph publisher is active for this output prefix: " +
                 path.string() + ": " + std::strerror(lock_error));
  }

  ~BuildLock() {
    if (fd >= 0) ::close(fd);
  }

  void release() {
    lib_assert(fd >= 0, "graph publication lock was not held");
    const int unlock_result = ::flock(fd, LOCK_UN);
    const int unlock_error = errno;
    const int close_result = ::close(fd);
    const int close_error = errno;
    fd = -1;
    lib_assert(unlock_result == 0,
               "failed to release graph publication lock: " +
                 path.string() + ": " + std::strerror(unlock_error));
    lib_assert(close_result == 0,
               "failed to close graph publication lock: " + path.string() +
                 ": " + std::strerror(close_error));
  }
};

bool path_exists_checked(const filepath_t& path) {
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  lib_assert(!error, "cannot inspect output path " + path.string() +
                       ": " + error.message());
  return exists;
}

void sync_file(const filepath_t& path) {
  const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  lib_assert(fd >= 0, "cannot open output for fsync: " + path.string() +
                        ": " + std::strerror(errno));
  const int sync_result = ::fsync(fd);
  const int sync_error = errno;
  const int close_result = ::close(fd);
  const int close_error = errno;
  lib_assert(sync_result == 0, "failed to fsync output: " + path.string() +
                                 ": " + std::strerror(sync_error));
  lib_assert(close_result == 0, "failed to close fsynced output: " +
                                  path.string() + ": " +
                                  std::strerror(close_error));
}

void sync_directory(const filepath_t& directory) {
  const filepath_t path = directory.empty() ? filepath_t{"."} : directory;
  const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  lib_assert(fd >= 0, "cannot open output directory for fsync: " +
                        path.string() + ": " + std::strerror(errno));
  const int sync_result = ::fsync(fd);
  const int sync_error = errno;
  const int close_result = ::close(fd);
  const int close_error = errno;
  lib_assert(sync_result == 0,
             "failed to fsync output directory: " + path.string() +
               ": " + std::strerror(sync_error));
  lib_assert(close_result == 0,
             "failed to close output directory: " + path.string() +
               ": " + std::strerror(close_error));
}

void create_sized_file(const filepath_t& path, u64 size) {
  std::ofstream output(path, std::ios::binary | std::ios::out | std::ios::trunc);
  lib_assert(output.good(), "failed to create output shard file: " + path.string());
  if (size > 0) {
    output.seekp(static_cast<std::streamoff>(size - 1));
    output.put(0);
  }
  output.close();
  lib_assert(!output.fail(), "failed to size output shard file: " + path.string());
}

u64 serialized_checksum_initial() {
  return 0x6a09e667f3bcc909ULL;
}

u64 serialized_checksum_update(u64 value, const void* data, size_t size) {
  lib_assert(size % sizeof(u64) == 0,
             "serialized graph checksum input is not word-aligned");
  const auto* bytes = static_cast<const byte_t*>(data);
  for (size_t offset = 0; offset < size; offset += sizeof(u64)) {
    u64 word = 0;
    std::memcpy(&word, bytes + offset, sizeof(word));
    value ^= word + 0x9e3779b97f4a7c15ULL +
      (value << 6) + (value >> 2);
    value = ((value << 27) | (value >> 37)) *
      0x94d049bb133111ebULL;
  }
  return value;
}

u64 checksum_file_range(std::ifstream& input, const filepath_t& path,
                        u64 offset, u64 bytes, const char* description,
                        bool serialized_graph = false) {
  constexpr size_t kChunkBytes = 8ull << 20;
  vec<byte_t> buffer(kChunkBytes);
  input.clear();
  input.seekg(static_cast<std::streamoff>(offset));
  lib_assert(input.good(), "failed to seek while validating " +
                             str{description} + ": " + path.string());
  u64 checksum = serialized_graph
    ? serialized_checksum_initial()
    : vamana::idmap::checksum_initial();
  for (u64 consumed = 0; consumed < bytes;) {
    const size_t chunk = static_cast<size_t>(
      std::min<u64>(buffer.size(), bytes - consumed));
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(chunk));
    lib_assert(input.gcount() == static_cast<std::streamsize>(chunk),
               "short read while validating " + str{description} +
                 ": " + path.string());
    checksum = serialized_graph
      ? serialized_checksum_update(checksum, buffer.data(), chunk)
      : vamana::idmap::checksum_update(checksum, buffer.data(), chunk);
    consumed += chunk;
  }
  return checksum;
}

void validate_temporary_shard(
    const filepath_t& path, u64 expected_size, u64 expected_fingerprint,
    u64 node_offset, u64 node_bytes, u64 expected_node_checksum,
    u64 graph_header_offset, const vamana::hot_graph::Header& expected_header,
    u64 graph_offset, u64 graph_bytes, u64 expected_graph_checksum) {
  std::error_code size_error;
  const uintmax_t actual_size = std::filesystem::file_size(path, size_error);
  lib_assert(!size_error && actual_size == expected_size,
             "temporary graph shard has an incomplete file size: " +
               path.string());
  std::ifstream input(path, std::ios::binary);
  lib_assert(input.good(),
             "cannot reopen temporary graph shard: " + path.string());
  u64 prefix[2]{};
  input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
  lib_assert(input.gcount() == static_cast<std::streamsize>(sizeof(prefix)) &&
               prefix[0] == expected_size &&
               prefix[1] == expected_fingerprint,
             "temporary graph shard has an invalid envelope: " +
               path.string());
  vamana::hot_graph::Header header;
  input.seekg(static_cast<std::streamoff>(graph_header_offset));
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  lib_assert(input.gcount() == static_cast<std::streamsize>(sizeof(header)) &&
               std::memcmp(&header, &expected_header, sizeof(header)) == 0,
             "temporary graph shard has an invalid compact graph header: " +
               path.string());
  const u64 node_checksum = checksum_file_range(
    input, path, node_offset, node_bytes, "fixed-node plane", true);
  lib_assert(node_checksum == expected_node_checksum,
             "temporary graph shard fixed-node checksum mismatch: " +
               path.string());
  const u64 graph_checksum = checksum_file_range(
    input, path, graph_offset, graph_bytes, "compact-graph plane", true);
  lib_assert(graph_checksum == expected_graph_checksum,
             "temporary graph shard compact-graph checksum mismatch: " +
               path.string());
}

void validate_temporary_centroid(
    const filepath_t& path, const vamana::centroid_state::Header& expected,
    span<const byte_t> expected_payload) {
  std::error_code size_error;
  const uintmax_t actual_size = std::filesystem::file_size(path, size_error);
  lib_assert(!size_error &&
               actual_size == sizeof(expected) + expected_payload.size(),
             "temporary centroid sidecar has an incomplete file size: " +
               path.string());
  std::ifstream input(path, std::ios::binary);
  vamana::centroid_state::Header header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  lib_assert(input.gcount() == static_cast<std::streamsize>(sizeof(header)) &&
               std::memcmp(&header, &expected, sizeof(header)) == 0,
             "temporary centroid sidecar header mismatch: " + path.string());
  vec<byte_t> payload(expected_payload.size());
  input.read(reinterpret_cast<char*>(payload.data()),
             static_cast<std::streamsize>(payload.size()));
  lib_assert(input.gcount() == static_cast<std::streamsize>(payload.size()) &&
               std::equal(payload.begin(), payload.end(),
                          expected_payload.begin(), expected_payload.end()) &&
               vamana::centroid_state::checksum(payload) ==
                 header.payload_checksum,
             "temporary centroid sidecar payload mismatch: " + path.string());
}

void finalize_owner_idmap_temporary(
    const filepath_t& path, const filepath_t& temporary,
    std::ofstream& output, vamana::idmap::Header header,
    u64 entry_count, u64 payload_checksum) {
  u64 payload_bytes = 0;
  lib_assert(vamana::idmap::checked_payload_bytes(entry_count, payload_bytes),
             "owner idmap payload size overflow: " + path.string());
  lib_assert(payload_bytes <= std::numeric_limits<size_t>::max() &&
               payload_bytes <= static_cast<u64>(
                 std::numeric_limits<std::streamsize>::max()),
             "owner idmap payload exceeds host I/O limits: " +
               path.string());
  header.entry_count = entry_count;
  header.payload_bytes = payload_bytes;
  header.payload_checksum = payload_checksum;
  header.header_checksum = vamana::idmap::compute_header_checksum(header);

  output.flush();
  lib_assert(output.good(),
             "failed to flush owner idmap payload: " + temporary.string());
  output.seekp(0);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  output.flush();
  lib_assert(output.good(),
             "failed to write complete owner idmap: " + temporary.string());
  output.close();
  lib_assert(!output.fail(),
             "failed to close complete owner idmap: " + temporary.string());
  sync_file(temporary);

  std::error_code error;
  const uintmax_t actual_bytes =
    std::filesystem::file_size(temporary, error);
  lib_assert(!error && actual_bytes == sizeof(header) + payload_bytes,
             "temporary owner idmap has an incomplete payload: " +
               temporary.string());
  std::ifstream input(temporary, std::ios::binary);
  vamana::idmap::Header stored;
  input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
  lib_assert(input.gcount() == static_cast<std::streamsize>(sizeof(stored)) &&
               std::memcmp(&stored, &header, sizeof(stored)) == 0,
             "temporary owner idmap header mismatch: " + temporary.string());
  const u64 stored_checksum = checksum_file_range(
    input, temporary, sizeof(stored), payload_bytes, "owner idmap payload");
  lib_assert(stored_checksum == payload_checksum,
             "temporary owner idmap payload checksum mismatch: " +
               temporary.string());
}

void publish_file(const filepath_t& temporary, const filepath_t& final_path) {
  std::error_code error;
  std::filesystem::rename(temporary, final_path, error);
  lib_assert(!error, "failed to publish graph-build output " +
                       final_path.string() + ": " + error.message());
}

}  // namespace

void validate_vamana_shard_capacity(
    size_t num_vectors, const VamanaBuildConfig& config) {
  lib_assert(num_vectors > 0, "cannot serialize an empty Vamana graph");
  lib_assert(config.num_memory_nodes > 0 &&
               config.num_memory_nodes <= num_vectors,
             "shard count must be in [1,vector_count]");

  const u64 vectors = static_cast<u64>(num_vectors);
  const u64 shards = config.num_memory_nodes;
  u64 maximum_shard_nodes = (vectors + shards - 1) / shards;
  if (config.partition_strategy == "metis") {
    const long double projected = std::ceil(
      static_cast<long double>(vectors) / shards *
      static_cast<long double>(config.partition_imbalance));
    maximum_shard_nodes = projected >= static_cast<long double>(vectors)
      ? vectors : static_cast<u64>(projected);
  }

  const u64 capacity = RemotePtr::BYTE_OFFSET_CAPACITY;
  const u64 node_bytes = VamanaNode::total_size();
  const u64 entry_bytes = VamanaNode::hot_graph_entry_size();
  lib_assert(node_bytes != 0 && entry_bytes != 0,
             "Vamana storage geometry is empty");
  lib_assert(maximum_shard_nodes <= (capacity - 16) / node_bytes,
             "projected fixed nodes exceed the 256 GiB tagged-pointer "
             "capacity of one shard; increase --memory-nodes or reduce the "
             "vector dimension");
  const u64 fixed_end = 16 + maximum_shard_nodes * node_bytes;
  const u64 header_offset = VamanaNode::align_storage(fixed_end);
  lib_assert(header_offset <= capacity - sizeof(vamana::hot_graph::Header),
             "projected hot-graph header exceeds tagged-pointer capacity");
  const u64 entry_offset = VamanaNode::align_storage(
    header_offset + sizeof(vamana::hot_graph::Header));
  lib_assert(maximum_shard_nodes <=
               (capacity - entry_offset) / entry_bytes,
             "projected compact graph exceeds the 256 GiB tagged-pointer "
             "capacity of one shard; increase --memory-nodes or reduce R");
  const u64 dynamic_base = VamanaNode::align_storage(
    entry_offset + maximum_shard_nodes * entry_bytes);
  lib_assert(RemotePtr::representable(0, dynamic_base, 1),
             "projected dynamic-node base exceeds tagged-pointer capacity");
}

void write_vamana_shards(const VamanaGraph& graph,
                         const Dataset& dataset,
                         const VamanaBuildConfig& config,
                         const filepath_t& output_prefix) {
  lib_assert(config.num_memory_nodes > 0 &&
               config.num_memory_nodes <= RemotePtr::MEMORY_NODE_MASK + 1,
             "tagged RemotePtr supports between 1 and 64 physical shards");
  const size_t n = dataset.size();
  const u32 dim = dataset.dim;
  validate_graph_for_serialization(graph, dataset, config);
  VamanaNode::disable_hot_graph();
  const size_t node_size = VamanaNode::total_size();
  const size_t aligned_size = (node_size + 7) & ~7ULL;
  const u64 build_fingerprint = make_build_fingerprint(
    output_prefix, n, dim, config.R, config.num_memory_nodes);

  ProgressReporter progress{"Exporting Vamana shards", n + config.num_memory_nodes};

  PlacementResult placement_result = place_nodes(graph, config, aligned_size);
  print_partition_stats(config, placement_result);
  const auto& placements = placement_result.placements;

  vec<u64> shard_sizes(config.num_memory_nodes, 16);
  vec<u64> shard_entry_counts(config.num_memory_nodes, 0);
  for (const auto& p : placements) {
    lib_assert(RemotePtr::representable(p.memory_node, p.offset, 0),
               "static node placement exceeds tagged RemotePtr capacity");
    shard_sizes[p.memory_node] = std::max<u64>(shard_sizes[p.memory_node], p.offset + aligned_size);
    shard_entry_counts[p.memory_node] = std::max<u64>(
      shard_entry_counts[p.memory_node], (p.offset - 16) / aligned_size + 1);
  }

  const u32 hot_graph_shard_bits = vamana::hot_graph::shard_bits_for(config.num_memory_nodes);
  const u32 hot_graph_entry_size = static_cast<u32>(VamanaNode::hot_graph_entry_size());
  vec<u64> hot_graph_header_offsets(config.num_memory_nodes, 0);
  vec<u64> hot_graph_entry_offsets(config.num_memory_nodes, 0);
  vec<u64> hot_graph_dynamic_base_offsets(config.num_memory_nodes, 0);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    hot_graph_header_offsets[shard] = VamanaNode::align_storage(shard_sizes[shard]);
    hot_graph_entry_offsets[shard] =
      VamanaNode::align_storage(hot_graph_header_offsets[shard] + sizeof(vamana::hot_graph::Header));
    hot_graph_dynamic_base_offsets[shard] = VamanaNode::align_storage(
      hot_graph_entry_offsets[shard] + shard_entry_counts[shard] * hot_graph_entry_size);
    shard_sizes[shard] = hot_graph_dynamic_base_offsets[shard];
    lib_assert(RemotePtr::representable(
                 shard, hot_graph_dynamic_base_offsets[shard], 1),
               "serialized graph leaves no representable dynamic-node base "
               "in this shard");
  }
  VamanaNode::configure_hot_graph(hot_graph_entry_offsets,
                                  shard_entry_counts,
                                  hot_graph_entry_size,
                                  hot_graph_shard_bits,
                                  hot_graph_dynamic_base_offsets,
                                  static_cast<u32>(VamanaNode::dynamic_record_size()),
                                  static_cast<u32>(VamanaNode::total_size()));

  vec<u64> shard_fingerprints(config.num_memory_nodes, 0);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_fingerprints[shard] = mix64(
      build_fingerprint ^ mix64(shard) ^ shard_sizes[shard] ^
      mix64(shard_entry_counts[shard]) ^ hot_graph_entry_offsets[shard]);
    if (shard_fingerprints[shard] == 0) {
      shard_fingerprints[shard] = mix64(build_fingerprint ^ shard ^ 1);
    }
  }

  const filepath_t output_dir = output_prefix.parent_path();
  if (!output_dir.empty()) std::filesystem::create_directories(output_dir);
  const filepath_t metadata_file{output_prefix.string() + ".meta.json"};
  lib_assert(!path_exists_checked(metadata_file),
             "refusing to overwrite a committed index; choose a new output "
             "prefix or explicitly remove the existing index first: " +
               metadata_file.string());
  BuildLock build_lock{
    filepath_t{output_prefix.string() + ".graph-build.lock"}};
  lib_assert(!path_exists_checked(metadata_file),
             "index metadata appeared while acquiring the graph-build lock: " +
               metadata_file.string());
  TemporaryOutputs temporary_outputs;

  vec<filepath_t> shard_paths(config.num_memory_nodes);
  vec<filepath_t> shard_temporary_paths(config.num_memory_nodes);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_paths[shard] = index_path::shard_file(
      output_prefix, shard + 1, config.num_memory_nodes);
    shard_temporary_paths[shard] = temporary_outputs.add(
      shard_paths[shard]);
    create_sized_file(shard_temporary_paths[shard], shard_sizes[shard]);
  }

  vec<vamana::hot_graph::Header> shard_headers(config.num_memory_nodes);
  vec<std::fstream> shard_files(config.num_memory_nodes);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_files[shard].open(shard_temporary_paths[shard],
                            std::ios::binary | std::ios::in | std::ios::out);
    lib_assert(shard_files[shard].good(),
               "failed to open temporary output shard file: " +
                 shard_temporary_paths[shard].string());
    shard_files[shard].seekp(0);
    shard_files[shard].write(reinterpret_cast<const char*>(&shard_sizes[shard]), sizeof(u64));
    shard_files[shard].write(
      reinterpret_cast<const char*>(&shard_fingerprints[shard]),
      sizeof(shard_fingerprints[shard]));
    auto& header = shard_headers[shard];
    header.version = vamana::hot_graph::kVersion3;
    header.entry_bytes = hot_graph_entry_size;
    header.max_degree = config.R;
    header.compact_pointer_shard_bits = hot_graph_shard_bits;
    header.entry_count = shard_entry_counts[shard];
    header.reserved0 = hot_graph_dynamic_base_offsets[shard];
    header.reserved1 = VamanaNode::allocation_size();
    header.reserved2 = static_cast<u32>(VamanaNode::total_size());
    shard_files[shard].seekp(static_cast<std::streamoff>(hot_graph_header_offsets[shard]));
    shard_files[shard].write(reinterpret_cast<const char*>(&header), sizeof(header));
  }

  vec<byte_t> node_buf(aligned_size, 0);
  vec<byte_t> hot_graph_entry(hot_graph_entry_size, 0);
  vec<RemotePtr> hot_neighbors(config.R);
  vec<u64> expected_node_checksums(
    config.num_memory_nodes, serialized_checksum_initial());
  vec<u64> expected_graph_checksums(
    config.num_memory_nodes, serialized_checksum_initial());
  vec<vec<f64>> centroid_sums(
    config.num_memory_nodes, vec<f64>(dim, 0.0));
  vec<u64> centroid_counts(config.num_memory_nodes, 0);
  struct RouteEntryCandidate {
    vamana::routing::CentroidSeedRank rank{};
    vamana::centroid_state::Entry entry{};
  };
  vec<vec<RouteEntryCandidate>> route_entries(config.num_memory_nodes);
  for (auto& entries : route_entries) {
    entries.reserve(vamana::centroid_state::kMaxLiveEntries);
  }
  vec<u32> nbrs;
  for (size_t i = 0; i < n; ++i) {
    std::fill(node_buf.begin(), node_buf.end(), 0);
    byte_t* buf = node_buf.data();

    const u64 node_header = VamanaNode::make_header(
      0, VamanaNode::HEADER_CENTROID_ACCOUNTED);
    const u32 node_id = dataset.id(i);
    std::memcpy(buf, &node_header, sizeof(node_header));
    std::memcpy(buf + VamanaNode::HEADER_SIZE, &node_id, sizeof(node_id));

    graph.copy_neighbors(i, nbrs);
    const u8 edge_count = static_cast<u8>(std::min<size_t>(nbrs.size(), config.R));
    const u32 zero = 0;
    std::memcpy(buf + VamanaNode::offset_generation(), &zero, sizeof(zero));
    std::memcpy(
      buf + VamanaNode::offset_slot_incarnation(), &zero, sizeof(zero));

    std::memcpy(buf + VamanaNode::offset_vector(), dataset.raw_vector(i), dataset.vector_bytes);

    std::fill(hot_neighbors.begin(), hot_neighbors.end(), RemotePtr{});
    for (u8 j = 0; j < edge_count; ++j) {
      const u32 nbr = nbrs[j];
      RemotePtr nbr_ptr{placements[nbr].memory_node, placements[nbr].offset};
      hot_neighbors[j] = nbr_ptr;
    }

    const auto& placement = placements[i];
    for (u32 dimension = 0; dimension < dim; ++dimension) {
      centroid_sums[placement.memory_node][dimension] +=
        static_cast<f64>(vector_component_as_float(
          dataset.raw_vector(i), dataset.dtype, dimension));
    }
    ++centroid_counts[placement.memory_node];
    auto& file = shard_files[placement.memory_node];
    file.seekp(static_cast<std::streamoff>(placement.offset));
    file.write(reinterpret_cast<const char*>(node_buf.data()), static_cast<std::streamsize>(node_buf.size()));
    lib_assert(file.good(), "failed to write temporary output shard node");
    expected_node_checksums[placement.memory_node] =
      serialized_checksum_update(
        expected_node_checksums[placement.memory_node],
        node_buf.data(), node_buf.size());
    VamanaNode::encode_hot_graph_entry(hot_graph_entry.data(),
                                       edge_count,
                                       hot_neighbors.data(),
                                       edge_count,
                                       hot_graph_shard_bits,
                                       0);
    const u64 slot = (placement.offset - 16) / aligned_size;
    const u64 hot_offset = hot_graph_entry_offsets[placement.memory_node] +
      slot * hot_graph_entry_size;
    file.seekp(static_cast<std::streamoff>(hot_offset));
    file.write(reinterpret_cast<const char*>(hot_graph_entry.data()),
               static_cast<std::streamsize>(hot_graph_entry_size));
    lib_assert(file.good(),
               "failed to write temporary compact hot graph entry");
    expected_graph_checksums[placement.memory_node] =
      serialized_checksum_update(
        expected_graph_checksums[placement.memory_node],
        hot_graph_entry.data(), hot_graph_entry_size);
    progress.increment();
  }

  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    shard_files[shard].flush();
    lib_assert(shard_files[shard].good(),
               "failed to flush temporary output shard file: " +
                 shard_temporary_paths[shard].string());
    shard_files[shard].close();
    lib_assert(!shard_files[shard].fail(),
               "failed to close temporary output shard file: " +
                 shard_temporary_paths[shard].string());
    sync_file(shard_temporary_paths[shard]);
    validate_temporary_shard(
      shard_temporary_paths[shard], shard_sizes[shard],
      shard_fingerprints[shard], vamana::hot_graph::kNodeBaseOffset,
      shard_entry_counts[shard] * aligned_size,
      expected_node_checksums[shard], hot_graph_header_offsets[shard],
      shard_headers[shard], hot_graph_entry_offsets[shard],
      shard_entry_counts[shard] * hot_graph_entry_size,
      expected_graph_checksums[shard]);
    progress.increment();
  }

  // Initialize the runtime-maintained live route entries with the real base
  // nodes nearest each physical shard's final compensated FP64 centroid. This
  // second offline pass is O(N*dim), uses O(shards) memory and is
  // dtype-independent; online maintenance later replaces entries from the
  // authoritative live membership under the updated centroid.
  const auto route_less = [](const RouteEntryCandidate& lhs,
                             const RouteEntryCandidate& rhs) {
    return vamana::routing::centroid_seed_rank_less(lhs.rank, rhs.rank);
  };
  for (size_t i = 0; i < n; ++i) {
    const auto& placement = placements[i];
    const u32 shard = placement.memory_node;
    lib_assert(centroid_counts[shard] != 0,
               "centroid seed selection encountered an empty shard");
    long double squared_l2 = 0;
    const f64 inverse_count =
      1.0 / static_cast<f64>(centroid_counts[shard]);
    for (u32 dimension = 0; dimension < dim; ++dimension) {
      const long double component = static_cast<long double>(
        vector_component_as_float(
          dataset.raw_vector(i), dataset.dtype, dimension));
      const long double centroid = static_cast<long double>(
        centroid_sums[shard][dimension] * inverse_count);
      const long double difference = component - centroid;
      squared_l2 += difference * difference;
    }
    const RemotePtr pointer{shard, placement.offset};
    const RouteEntryCandidate candidate{
      .rank = {
        .squared_l2 = squared_l2,
        .pointer_raw = pointer.raw_address,
      },
      .entry = {.remote_node = pointer.raw_address},
    };
    auto& entries = route_entries[shard];
    if (entries.size() < vamana::centroid_state::kMaxLiveEntries) {
      entries.push_back(candidate);
      std::sort(entries.begin(), entries.end(), route_less);
    } else if (route_less(candidate, entries.back())) {
      entries.back() = candidate;
      std::sort(entries.begin(), entries.end(), route_less);
    }
  }

  vec<filepath_t> centroid_paths(config.num_memory_nodes);
  vec<filepath_t> centroid_temporary_paths(config.num_memory_nodes);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    centroid_paths[shard] = index_path::centroid_state_file(
      output_prefix, shard + 1, config.num_memory_nodes);
    centroid_temporary_paths[shard] = temporary_outputs.add(
      centroid_paths[shard]);
    lib_assert(centroid_counts[shard] == shard_entry_counts[shard] &&
                 !route_entries[shard].empty(),
               "physical shard centroid state is incomplete");
    vec<vamana::centroid_state::Entry> entries;
    entries.reserve(route_entries[shard].size());
    for (const RouteEntryCandidate& candidate : route_entries[shard]) {
      entries.push_back(candidate.entry);
    }
    vamana::centroid_state::Header header;
    header.build_fingerprint = build_fingerprint;
    header.shard_fingerprint = shard_fingerprints[shard];
    header.shard = shard;
    header.shard_count = config.num_memory_nodes;
    header.dim = dim;
    header.max_degree = config.R;
    header.entry_count = static_cast<u32>(entries.size());
    header.vector_count = centroid_counts[shard];
    header.node_base_offset = vamana::hot_graph::kNodeBaseOffset;
    header.vector_dtype = static_cast<u32>(dataset.dtype);
    header.vector_component_size = static_cast<u32>(
      vector_dtype_component_size(dataset.dtype));
    header.node_size = static_cast<u32>(node_size);
    header.vector_offset = static_cast<u32>(VamanaNode::offset_vector());
    header.vector_bytes = static_cast<u32>(dataset.vector_bytes);
    header.slot_incarnation_offset = static_cast<u32>(
      VamanaNode::offset_slot_incarnation());
    header.hot_graph_version = vamana::hot_graph::kVersion3;
    header.hot_graph_entry_size = hot_graph_entry_size;
    header.hot_graph_pointer_bytes = vamana::hot_graph::kCompactPointerBytes;
    header.hot_graph_shard_bits = hot_graph_shard_bits;
    header.payload_bytes = vamana::centroid_state::payload_bytes(
      dim, header.entry_count);
    vec<byte_t> payload(static_cast<size_t>(header.payload_bytes));
    std::memcpy(payload.data(), centroid_sums[shard].data(),
                static_cast<size_t>(dim) * sizeof(f64));
    std::memcpy(payload.data() + static_cast<size_t>(dim) * sizeof(f64),
                entries.data(), entries.size() * sizeof(entries.front()));
    header.payload_checksum = vamana::centroid_state::checksum(payload);
    header.header_checksum =
      vamana::centroid_state::compute_header_checksum(header);

    std::ofstream output(
      centroid_temporary_paths[shard],
      std::ios::binary | std::ios::out | std::ios::trunc);
    lib_assert(output.good(),
               "failed to create temporary centroid sidecar: " +
                 centroid_temporary_paths[shard].string());
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    lib_assert(output.good(),
               "failed to write temporary centroid sidecar: " +
                 centroid_temporary_paths[shard].string());
    output.close();
    lib_assert(!output.fail(),
               "failed to close temporary centroid sidecar: " +
                 centroid_temporary_paths[shard].string());
    sync_file(centroid_temporary_paths[shard]);
    validate_temporary_centroid(
      centroid_temporary_paths[shard], header, payload);
  }

  nlohmann::json metadata{
    {"data_file", dataset.source_file.string()},
    {"output_prefix", output_prefix.string()},
    {"distance", "l2"},
    {"num_vectors", n},
    {"dim", dim},
    {"R", config.R},
    {"beam_width", config.beam_width},
    {"beam_width_construction", config.beam_width},
    {"alpha", config.alpha},
    {"num_memory_nodes", config.num_memory_nodes},
    {"node_size", node_size},
    {"node_layout", VamanaNode::layout_name()},
    {"storage_format", VamanaNode::storage_format_name()},
    {"schema_version", 15},
    {"graph_hot_bytes", VamanaNode::graph_hot_bytes()},
    {"vector_offset", VamanaNode::offset_vector()},
    {"slot_incarnation_offset", VamanaNode::offset_slot_incarnation()},
    {"remote_ptr_format", "tagged_inc24_shard6_off34x16_v1"},
    {"vector_storage_bytes", VamanaNode::vector_storage_bytes()},
    {"offline_builder_version", 2},
    {"random_graph_seed_scope", "per_node"},
    {"vector_data_type", vector_dtype_name(dataset.dtype)},
    {"vector_component_size", vector_dtype_component_size(dataset.dtype)},
    {"vector_bytes", dataset.vector_bytes},
    {"partition_strategy", config.partition_strategy},
    {"partition_max_degree", config.partition_max_degree},
    {"partition_imbalance", config.partition_imbalance},
    {"partition_edge_cut", placement_result.stats.edge_cut},
    {"partition_cross_shard_ratio", placement_result.cross_shard_ratio},
    {"navigation_quantizer", ""},
    {"navigation_code_bytes", 0},
    {"pq_subquantizers", 0},
    {"pq_bits", 0},
    {"navigation_model_checksum", 0},
    {"navigation_format", ""},
    {"navigation_code_remote_offsets", vec<u64>{}},
    {"navigation_code_region_bytes", vec<u64>{}},
    {"navigation_code_materialization", ""},
    {"navigation_graph_source", "storage_compact_graph"},
    {"navigation_execution", ""},
  };
  metadata["hot_graph_neighbor_read_bytes"] = hot_graph_entry_size;
  metadata["hot_graph_neighbor_update_bytes"] = hot_graph_entry_size;
  metadata["hot_graph_entry_size"] = hot_graph_entry_size;
  metadata["hot_graph_pointer_bytes"] = vamana::hot_graph::kCompactPointerBytes;
  metadata["hot_graph_shard_bits"] = hot_graph_shard_bits;
  metadata["hot_graph_offsets"] = hot_graph_entry_offsets;
  metadata["hot_graph_header_offsets"] = hot_graph_header_offsets;
  metadata["hot_graph_entry_counts"] = shard_entry_counts;
  metadata["hot_graph_dynamic_base_offsets"] = hot_graph_dynamic_base_offsets;
  metadata["hot_graph_dynamic_record_bytes"] = VamanaNode::allocation_size();
  metadata["hot_graph_dynamic_hot_offset"] = VamanaNode::total_size();
  metadata["allocation_size"] = VamanaNode::allocation_size();
  metadata["idmap_format"] = "owner_sharded_v2_bound";
  metadata["centroid_state_format"] = "physical_shard_centroid_v2_bound";
  metadata["index_build_fingerprint"] = build_fingerprint;
  metadata["shard_build_fingerprints"] = shard_fingerprints;
  metadata["centroid_state_header_bytes"] =
    sizeof(vamana::centroid_state::Header);
  // Stream one pass over the placement table directly into per-owner
  // temporary files. Keeping all idmap entries in vectors would add 24 bytes
  // per base vector (about 24 GB for SIFT1B) to peak builder memory.
  vec<filepath_t> idmap_paths(config.num_memory_nodes);
  vec<filepath_t> idmap_temporary_paths(config.num_memory_nodes);
  vec<std::ofstream> idmap_outputs(config.num_memory_nodes);
  vec<u64> idmap_entry_counts(config.num_memory_nodes, 0);
  vec<u64> idmap_payload_checksums(
    config.num_memory_nodes, vamana::idmap::checksum_initial());
  for (u32 owner = 0; owner < config.num_memory_nodes; ++owner) {
    idmap_paths[owner] = index_path::owner_idmap_file(
      output_prefix, owner + 1, config.num_memory_nodes);
    idmap_temporary_paths[owner] = temporary_outputs.add(
      idmap_paths[owner]);
    idmap_outputs[owner].open(
      idmap_temporary_paths[owner],
      std::ios::binary | std::ios::out | std::ios::trunc);
    lib_assert(idmap_outputs[owner].good(),
               "failed to create temporary owner idmap: " +
                 idmap_temporary_paths[owner].string());
    const vamana::idmap::Header placeholder;
    idmap_outputs[owner].write(
      reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));
    lib_assert(idmap_outputs[owner].good(),
               "failed to reserve temporary owner idmap header: " +
                 idmap_temporary_paths[owner].string());
  }
  for (size_t index = 0; index < n; ++index) {
    const node_t id = dataset.id(index);
    const u32 owner = static_cast<u32>(id % config.num_memory_nodes);
    const vamana::idmap::Entry entry{
      id,
      RemotePtr{placements[index].memory_node,
                placements[index].offset}.raw_address,
      0,
      0,
      0,
    };
    idmap_outputs[owner].write(
      reinterpret_cast<const char*>(&entry), sizeof(entry));
    ++idmap_entry_counts[owner];
    idmap_payload_checksums[owner] = vamana::idmap::checksum_update(
      idmap_payload_checksums[owner], &entry, sizeof(entry));
  }
  for (u32 owner = 0; owner < config.num_memory_nodes; ++owner) {
    lib_assert(idmap_outputs[owner].good(),
               "failed while streaming owner idmap: " +
                 idmap_temporary_paths[owner].string());
    vamana::idmap::Header header;
    header.build_fingerprint = build_fingerprint;
    header.owner_shard_fingerprint = shard_fingerprints[owner];
    header.owner_shard = owner;
    header.shard_count = config.num_memory_nodes;
    header.node_base_offset = vamana::hot_graph::kNodeBaseOffset;
    header.node_size = static_cast<u32>(node_size);
    header.id_offset = static_cast<u32>(VamanaNode::offset_id());
    header.generation_offset =
      static_cast<u32>(VamanaNode::offset_generation());
    header.slot_incarnation_offset =
      static_cast<u32>(VamanaNode::offset_slot_incarnation());
    finalize_owner_idmap_temporary(
      idmap_paths[owner], idmap_temporary_paths[owner],
      idmap_outputs[owner], header, idmap_entry_counts[owner],
      idmap_payload_checksums[owner]);
  }
  // Metadata is itself prepared and validated before any final pathname is
  // changed. The asset renames are individually atomic; the directory fsync
  // makes all of them durable before metadata becomes the commit marker.
  const filepath_t temporary_metadata = temporary_outputs.add(metadata_file);
  {
    std::ofstream metadata_output(temporary_metadata,
                                  std::ios::out | std::ios::trunc);
    lib_assert(metadata_output.good(),
               "failed to create temporary index metadata: " +
                 temporary_metadata.string());
    metadata_output << std::setw(2) << metadata << std::endl;
    metadata_output.close();
    lib_assert(!metadata_output.fail(),
               "failed to write temporary index metadata: " +
                 temporary_metadata.string());
  }
  sync_file(temporary_metadata);
  {
    std::ifstream metadata_input(temporary_metadata);
    nlohmann::json validated_metadata;
    metadata_input >> validated_metadata;
    metadata_input >> std::ws;
    lib_assert(metadata_input.good() || metadata_input.eof(),
               "failed to parse temporary index metadata: " +
                 temporary_metadata.string());
    lib_assert(metadata_input.peek() == std::char_traits<char>::eof() &&
                 validated_metadata == metadata,
               "temporary index metadata failed exact round-trip validation: " +
                 temporary_metadata.string());
  }

  lib_assert(!path_exists_checked(metadata_file),
             "committed metadata appeared before graph publication; refusing "
             "to replace any final output: " + metadata_file.string());
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    publish_file(shard_temporary_paths[shard], shard_paths[shard]);
    publish_file(centroid_temporary_paths[shard], centroid_paths[shard]);
    publish_file(idmap_temporary_paths[shard], idmap_paths[shard]);
  }
  sync_directory(output_dir);
  publish_file(temporary_metadata, metadata_file);
  sync_directory(output_dir);
  // Keep the lock inode permanently. Unlinking a lock file after unlock lets
  // another process acquire the old inode while a third process creates and
  // locks a new inode at the same pathname.
  build_lock.release();
  progress.finish();
}


}  // namespace tools::vamana_offline
