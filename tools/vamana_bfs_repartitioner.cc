#include <cstdlib>
#include <iostream>
#include <string>

#include "tools/vamana_offline/partitioning.hh"
#include "tools/vamana_repartitioner_common.hh"

namespace {

using tools::vamana_repartition::Options;

[[noreturn]] void usage(const char* program, const str& error = {}) {
  if (!error.empty()) std::cerr << "error: " << error << "\n\n";
  std::cerr
    << "Usage: " << program << " --input-prefix OLD_PREFIX --output-prefix NEW_PREFIX\n"
    << "       [--memory-nodes N --dim D --R R]\n"
    << "       [--vector-data-type auto|float32|uint8|int8]\n"
    << "       [--storage-format auto|vamana_aos_v1|vamana_compact_v1]\n"
    << "       [--anchors-per-shard 4096 --anchor-seed 1234]\n"
    << "       [--rabitq-cache-format auto|budget|full]\n"
    << "       [--overwrite]\n\n"
    << "Repartitions a schema-13 Vamana index with multi-source BFS. The output\n"
    << "includes compact hot-graph data when requested, owner idmaps, anchors,\n"
    << "and RFQ5 RaBitQ sidecars when the input has RaBitQ entries.\n";
  std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

str require_value(int& index, int argc, char** argv, const str& option) {
  if (index + 1 >= argc) usage(argv[0], option + " requires a value");
  return argv[++index];
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const str argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      usage(argv[0]);
    } else if (argument == "--input-prefix") {
      options.input_prefix = require_value(i, argc, argv, argument);
    } else if (argument == "--output-prefix") {
      options.output_prefix = require_value(i, argc, argv, argument);
    } else if (argument == "--memory-nodes") {
      options.memory_nodes = static_cast<u32>(
        std::stoul(require_value(i, argc, argv, argument)));
    } else if (argument == "--dim") {
      options.dim = static_cast<u32>(
        std::stoul(require_value(i, argc, argv, argument)));
    } else if (argument == "--R") {
      options.R = static_cast<u32>(
        std::stoul(require_value(i, argc, argv, argument)));
    } else if (argument == "--vector-data-type") {
      const str value = require_value(i, argc, argv, argument);
      if (value == "auto") {
        options.vector_dtype_set = false;
      } else {
        options.vector_dtype = parse_vector_dtype(value);
        options.vector_dtype_set = true;
      }
    } else if (argument == "--storage-format") {
      options.storage_format = require_value(i, argc, argv, argument);
    } else if (argument == "--anchors-per-shard") {
      options.anchors_per_shard = static_cast<u32>(
        std::stoul(require_value(i, argc, argv, argument)));
      options.anchors_per_shard_set = true;
    } else if (argument == "--anchor-seed") {
      options.anchor_seed = std::stoull(require_value(i, argc, argv, argument));
    } else if (argument == "--rabitq-cache-format") {
      options.rabitq_cache_format = require_value(i, argc, argv, argument);
    } else if (argument == "--overwrite") {
      options.overwrite = true;
    } else {
      usage(argv[0], "unknown argument: " + argument);
    }
  }
  if (options.input_prefix.empty()) usage(argv[0], "--input-prefix is required");
  if (options.output_prefix.empty()) {
    options.output_prefix = filepath_t(options.input_prefix.string() + "_bfs");
  }
  return options;
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
    using namespace tools::vamana_repartition;

    Index index(parse_options(argc, argv));
    std::cerr << "input prefix: " << index.options().input_prefix << "\n"
              << "output prefix: " << index.options().output_prefix << "\n"
              << "storage format: " << index.input_storage_format()
              << " -> " << index.output_storage_format() << "\n"
              << "nodes: " << index.node_count()
              << " shards=" << index.options().memory_nodes
              << " dim=" << index.options().dim
              << " R=" << index.options().R << "\n";

    CrossShardStats before_stats;
    vec<vec<u32>> neighbors = index.read_neighbor_lists(&before_stats);
    std::cerr << "before cross-shard ratio: " << before_stats.ratio()
              << " (" << before_stats.cross_edges
              << "/" << before_stats.total_edges << ")\n";

    PartitionStats partition_stats;
    vec<u32> parts = compute_bfs_partition(
      neighbors,
      index.options().memory_nodes,
      index.medoid_vertex(),
      &partition_stats);
    print_part_counts(partition_stats.part_node_counts);
    std::cerr << "BFS partition edges: input=" << partition_stats.input_edges
              << " edge_cut=" << partition_stats.edge_cut
              << " partition_cut_ratio="
              << partition_stats.partition_cross_shard_ratio << "\n";

    const WriteResult result = index.write(
      parts, "bfs", partition_stats, before_stats);
    std::cerr << "after cross-shard ratio: " << result.after_stats.ratio()
              << " (" << result.after_stats.cross_edges
              << "/" << result.after_stats.total_edges << ")\n"
              << "repartition finished: " << result.node_count << " nodes\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "vamana_bfs_repartitioner failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
