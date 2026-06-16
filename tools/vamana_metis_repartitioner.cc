#include <cstdlib>
#include <iostream>
#include <string>

#include "tools/vamana_offline/partitioning.hh"
#include "tools/vamana_repartitioner_common.hh"

namespace {

struct Config {
  tools::vamana_repartition::Options common;
  u32 partition_max_degree{16};
  double partition_imbalance{1.03};
};

[[noreturn]] void usage(const char* program, const str& error = {}) {
  if (!error.empty()) std::cerr << "error: " << error << "\n\n";
  std::cerr
    << "Usage: " << program << " --input-prefix OLD_PREFIX --output-prefix NEW_PREFIX\n"
    << "       [--memory-nodes N --dim D --R R]\n"
    << "       [--vector-data-type auto|float32|uint8|int8]\n"
    << "       [--storage-format auto|vamana_aos_v1|vamana_compact_v1]\n"
    << "       [--anchors-per-shard 4096 --anchor-seed 1234]\n"
    << "       [--partition-max-degree 16 --partition-imbalance 1.03]\n"
    << "       [--overwrite]\n\n"
    << "Repartitions a schema-13 Vamana index with METIS. The output includes\n"
    << "compact hot-graph data when requested, owner idmaps, anchors, and RFQ5\n"
    << "RaBitQ sidecars when the input has RaBitQ entries.\n";
  std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

str require_value(int& index, int argc, char** argv, const str& option) {
  if (index + 1 >= argc) usage(argv[0], option + " requires a value");
  return argv[++index];
}

Config parse_config(int argc, char** argv) {
  Config config;
  auto& options = config.common;
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
    } else if (argument == "--partition-max-degree") {
      config.partition_max_degree = static_cast<u32>(
        std::stoul(require_value(i, argc, argv, argument)));
    } else if (argument == "--partition-imbalance") {
      config.partition_imbalance =
        std::stod(require_value(i, argc, argv, argument));
    } else if (argument == "--overwrite") {
      options.overwrite = true;
    } else {
      usage(argv[0], "unknown argument: " + argument);
    }
  }
  if (options.input_prefix.empty()) usage(argv[0], "--input-prefix is required");
  if (options.output_prefix.empty()) {
    options.output_prefix = filepath_t(options.input_prefix.string() + "_metis");
  }
  if (config.partition_max_degree == 0) {
    usage(argv[0], "--partition-max-degree must be > 0");
  }
  if (config.partition_imbalance < 1.0) {
    usage(argv[0], "--partition-imbalance must be >= 1.0");
  }
  return config;
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

    Config config = parse_config(argc, argv);
    if (!metis_partitioning_available()) {
      throw std::runtime_error(metis_unavailable_reason());
    }
    Index index(std::move(config.common));
    std::cerr << "input prefix: " << index.options().input_prefix << "\n"
              << "output prefix: " << index.options().output_prefix << "\n"
              << "storage format: " << index.input_storage_format()
              << " -> " << index.output_storage_format() << "\n"
              << "nodes: " << index.node_count()
              << " shards=" << index.options().memory_nodes
              << " dim=" << index.options().dim
              << " R=" << index.options().R << "\n"
              << "partition: metis max_degree=" << config.partition_max_degree
              << " imbalance=" << config.partition_imbalance << "\n";

    CrossShardStats before_stats;
    vec<u64> edges = index.read_partition_edges(
      config.partition_max_degree, &before_stats);
    std::cerr << "before cross-shard ratio: " << before_stats.ratio()
              << " (" << before_stats.cross_edges
              << "/" << before_stats.total_edges << ")\n";

    PartitionOptions partition_options;
    partition_options.num_parts = index.options().memory_nodes;
    partition_options.max_degree = config.partition_max_degree;
    partition_options.imbalance = config.partition_imbalance;
    PartitionStats partition_stats;
    vec<u32> parts = compute_metis_partition(
      index.node_count(), edges, partition_options, &partition_stats);
    print_part_counts(partition_stats.part_node_counts);
    std::cerr << "METIS partition edges: input=" << partition_stats.input_edges
              << " unique=" << partition_stats.unique_edges
              << " edge_cut=" << partition_stats.edge_cut
              << " partition_cut_ratio="
              << partition_stats.partition_cross_shard_ratio << "\n";

    const nlohmann::json partition_metadata{
      {"partition_max_degree", config.partition_max_degree},
      {"partition_imbalance", config.partition_imbalance}};
    const WriteResult result = index.write(
      parts, "metis", partition_stats, before_stats, partition_metadata);
    std::cerr << "after cross-shard ratio: " << result.after_stats.ratio()
              << " (" << result.after_stats.cross_edges
              << "/" << result.after_stats.total_edges << ")\n"
              << "repartition finished: " << result.node_count << " nodes\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "vamana_metis_repartitioner failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
