#include "tools/vamana_offline/config.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>

#include <boost/program_options.hpp>
#include <library/utils.hh>

#include "common/constants.hh"
#include "remote_pointer.hh"

namespace po = boost::program_options;

namespace tools::vamana_offline {

filepath_t default_vamana_prefix(const filepath_t& data_path, u32 R, u32 beam_width) {
  const filepath_t base = std::filesystem::is_regular_file(data_path) ? data_path.parent_path() : data_path;
  return base / "dump" / ("vamana_R" + std::to_string(R) + "_bw" + std::to_string(beam_width));
}

VamanaBuildConfig parse_configuration(int argc, char** argv) {
  VamanaBuildConfig config;

  po::options_description desc{"Vamana offline builder options"};
  desc.add_options()
    ("help,h", "Show help message")
    ("data-path,d", po::value<filepath_t>(&config.data_path), "Path to a dataset file or directory.")
    ("output-prefix,o", po::value<filepath_t>(&config.output_prefix),
     "Output prefix without _nodeX_ofN.dat suffix.")
    ("memory-nodes,n", po::value<u32>(&config.num_memory_nodes)->default_value(config.num_memory_nodes),
     "Number of output shards / memory nodes.")
    ("threads,t", po::value<u32>(&config.threads)->default_value(config.threads),
     "Number of threads. 0 = hardware concurrency.")
    ("export-threads", po::value<u32>(&config.export_threads)->default_value(config.export_threads),
     "Shard export worker count; caps concurrently growing shard files. 0 = --threads.")
    ("R", po::value<u32>(&config.R)->default_value(config.R), "Maximum out-degree.")
    ("beam-width", po::value<u32>(&config.beam_width)->default_value(config.beam_width),
     "Beam width for beam search during offline construction.")
    ("beam-width-construction", po::value<u32>(&config.beam_width),
     "Alias for --beam-width. Offline builder only has a construction beam width.")
    ("ef-construction", po::value<u32>(&config.beam_width),
     "Alias for --beam-width in the offline builder.")
    ("alpha", po::value<f64>(&config.alpha)->default_value(config.alpha), "RobustPrune alpha parameter.")
    ("vector-data-type", po::value<str>(&config.vector_data_type)->default_value(config.vector_data_type),
     "Storage dtype for full vectors: auto, float32, uint8, or int8.")
    ("partition-strategy", po::value<str>(&config.partition_strategy)->default_value(config.partition_strategy),
     "Shard placement strategy: balanced, bfs, or metis.")
    ("partition-max-degree", po::value<u32>(&config.partition_max_degree)->default_value(config.partition_max_degree),
     "Maximum neighbors per node used to build the METIS partition graph.")
    ("partition-kmeans-samples", po::value<u64>(&config.partition_kmeans_samples)->default_value(config.partition_kmeans_samples),
     "Sample size for kmeans vector-space partitioning.")
    ("partition-kmeans-iters", po::value<u32>(&config.partition_kmeans_iters)->default_value(config.partition_kmeans_iters),
     "Lloyd iterations for kmeans vector-space partitioning.")
    ("partition-imbalance", po::value<double>(&config.partition_imbalance)->default_value(config.partition_imbalance),
     "METIS ubvec balance tolerance, e.g. 1.03 allows about 3% imbalance.")
    ("skip-sanity-check", po::bool_switch(&config.skip_sanity_check),
     "Skip the expensive in-memory brute-force recall sanity check after graph construction.")
    ("preflight-only", po::bool_switch(&config.preflight_only),
     "Validate binary capabilities and command-line parameters without reading vectors.")
    ("seed", po::value<i32>(&config.seed)->default_value(config.seed), "PRNG seed.")
    ("consolidation-chunk-bytes", po::value<size_t>(&config.consolidation_chunk_bytes)->default_value(config.consolidation_chunk_bytes),
     "Target bytes per reverse-edge consolidation chunk CSR; 0 = automatic (32 GiB).")
    ("max-vectors", po::value<size_t>(&config.max_vectors)->default_value(config.max_vectors),
     "Maximum number of vectors to read.")
    ("checkpoint-prefix", po::value<filepath_t>(&config.checkpoint_prefix),
     "Enable pause/resume checkpoints under this prefix. Send SIGUSR1 to pause: "
     "the builder commits a checkpoint at the next segment boundary and exits 75; "
     "relaunch with the same flags to resume. Requires a fixed --seed.")
    ("checkpoint-segment-nodes", po::value<size_t>(&config.checkpoint_segment_nodes)
         ->default_value(config.checkpoint_segment_nodes),
     "Main-loop segment size (nodes processed between pause checks).")
    ("checkpoint-interval-segments", po::value<size_t>(&config.checkpoint_interval_segments)
         ->default_value(config.checkpoint_interval_segments),
     "Automatically commit a checkpoint every N segments; 0 disables periodic commits.")
    ("query-path", po::value<filepath_t>(&config.query_path),
     "Path to query file (.fbin) for post-build recall test.")
    ("groundtruth-path", po::value<filepath_t>(&config.groundtruth_path),
     "Path to ground truth file (.bin) for post-build recall test.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  po::notify(vm);

  if (config.data_path.empty()) lib_failure("--data-path is required");
  if (config.num_memory_nodes == 0) lib_failure("--memory-nodes must be > 0");
  if (config.num_memory_nodes > RemotePtr::MEMORY_NODE_MASK + 1)
    lib_failure("--memory-nodes exceeds the 64-shard tagged pointer limit");
  const u32 max_configurable_threads =
    std::max<u32>(1024, std::thread::hardware_concurrency());
  if (config.threads > max_configurable_threads)
    lib_failure("--threads must be zero (automatic) or at most " +
                std::to_string(max_configurable_threads));
  if (config.R == 0) lib_failure("--R must be > 0");
  if (config.beam_width == 0) lib_failure("--beam-width must be > 0");
  if (!std::isfinite(config.alpha) || config.alpha < 1.0)
    lib_failure("--alpha must be finite and >= 1.0");
  if (config.R > kMaxSupportedGraphDegree)
    lib_failure("--R exceeds the system-wide degree limit of " +
                std::to_string(kMaxSupportedGraphDegree));
  if (config.vector_data_type != "auto") {
    try {
      (void)parse_vector_dtype(config.vector_data_type);
    } catch (const std::exception& e) {
      lib_failure(str{"--vector-data-type must be auto, float32, uint8, or int8: "} + e.what());
    }
  }
  if (config.partition_strategy != "balanced" &&
      config.partition_strategy != "bfs" &&
      config.partition_strategy != "metis" &&
      config.partition_strategy != "kmeans")
    lib_failure("--partition-strategy must be balanced, bfs, metis, or kmeans");
  if (config.partition_max_degree == 0)
    lib_failure("--partition-max-degree must be > 0");
  if (config.partition_max_degree > config.R)
    lib_failure("--partition-max-degree must be <= --R");
  if (!std::isfinite(config.partition_imbalance) ||
      config.partition_imbalance < 1.0)
    lib_failure("--partition-imbalance must be finite and >= 1.0");
  if (!config.checkpoint_prefix.empty() &&
      config.checkpoint_segment_nodes == 0)
    lib_failure("--checkpoint-segment-nodes must be > 0");
  return config;
}

}  // namespace tools::vamana_offline
