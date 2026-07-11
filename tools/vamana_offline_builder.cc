#include <chrono>
#include <cstdlib>
#include <iostream>

#include <library/utils.hh>

#include "tools/vamana_offline/config.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/progress.hh"
#include "tools/vamana_offline/recall_check.hh"
#include "tools/vamana_offline/shard_writer.hh"
#include "vamana/vamana_node.hh"

using namespace tools::vamana_offline;

int main(int argc, char** argv) {
  const VamanaBuildConfig config = parse_configuration(argc, argv);
  const Dataset dataset = read_dataset(config);
  const filepath_t output_prefix =
      config.output_prefix.empty()
          ? default_vamana_prefix(dataset.source_file, config.R, config.beam_width)
          : config.output_prefix;

  std::cerr << "output prefix: " << output_prefix << "\n";
  std::cerr << "memory nodes: " << config.num_memory_nodes << "\n";
  std::cerr << "threads: " << effective_thread_count(config.threads) << "\n";
  std::cerr << "R=" << config.R << " construction_beam_width=" << config.beam_width
            << " alpha=" << config.alpha
            << " vector_data_type=" << vector_dtype_name(dataset.dtype) << "\n";
  std::cerr << "partition_strategy=" << config.partition_strategy
            << " partition_max_degree=" << config.partition_max_degree
            << " partition_imbalance=" << config.partition_imbalance << "\n";
  std::cerr << "skip_sanity_check=" << (config.skip_sanity_check ? "true" : "false") << "\n";

  const auto build_start = std::chrono::steady_clock::now();

  // Initialize VamanaNode static storage
  VamanaNode::init_static_storage(dataset.dim, config.R, dataset.dtype);

  std::cerr << "offline distance execution: cpu-avx2\n";

  // Step 1: Build Vamana graph
  VamanaGraph graph;
  build_vamana_graph(graph, dataset, config);

  run_optional_recall_check(graph, dataset, config);

  // Step 2: Serialize to shard files
  write_vamana_shards(graph, dataset, config, output_prefix);

  const auto build_end = std::chrono::steady_clock::now();
  const auto seconds = std::chrono::duration_cast<std::chrono::duration<double>>(build_end - build_start).count();
  std::cerr << "offline build finished in " << seconds << " seconds\n";

  return EXIT_SUCCESS;
}
