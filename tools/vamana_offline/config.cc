#include "tools/vamana_offline/config.hh"

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <boost/program_options.hpp>
#include <library/utils.hh>

#include "gpu_search/index_format.hh"
#include "vamana/storage_format.hh"

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
    ("partition-imbalance", po::value<double>(&config.partition_imbalance)->default_value(config.partition_imbalance),
     "METIS ubvec balance tolerance, e.g. 1.03 allows about 3% imbalance.")
    ("skip-sanity-check", po::bool_switch(&config.skip_sanity_check),
     "Skip the expensive in-memory brute-force recall sanity check after graph construction.")
    ("use-rabitq", po::bool_switch(&config.use_rabitq),
     "Store dimension-scaled RaBitQ search entries per node for GPU approximate search.")
    ("rabitq-cache-format",
     po::value<str>(&config.rabitq_cache_format)->default_value(config.rabitq_cache_format),
     "RaBitQ compute-side sidecar format: budget or full.")
    ("storage-format", po::value<str>(&config.storage_format)->default_value(config.storage_format),
     "Storage format to write: vamana_aos_v1 or vamana_compact_v1.")
    ("seed", po::value<i32>(&config.seed)->default_value(config.seed), "PRNG seed.")
    ("max-vectors", po::value<size_t>(&config.max_vectors)->default_value(config.max_vectors),
     "Maximum number of vectors to read.")
    ("ip-dist", po::bool_switch(&config.ip_distance), "Use inner-product distance instead of L2.")
    ("query-path", po::value<filepath_t>(&config.query_path),
     "Path to query file (.fbin) for post-build recall test.")
    ("groundtruth-path", po::value<filepath_t>(&config.groundtruth_path),
     "Path to ground truth file (.bin) for post-build recall test.")
    ("anchor-count-per-shard",
     po::value<u32>(&config.anchor_count_per_shard)->default_value(config.anchor_count_per_shard),
     "Representative anchors written per shard. 0 disables the anchor sidecar.")
    ("gpu-tiered-index", po::bool_switch(&config.build_gpu_tiered_index),
     "Build the small GPU V4 manifest and per-shard RaBitQ code streams.")
    ("gpu-entry-points", po::value<u32>(&config.gpu_entry_points)->default_value(config.gpu_entry_points),
     "Number of deterministic shard-balanced GPU search entry points.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  po::notify(vm);

  if (config.data_path.empty()) lib_failure("--data-path is required");
  if (config.num_memory_nodes == 0) lib_failure("--memory-nodes must be > 0");
  if (config.R == 0) lib_failure("--R must be > 0");
  if (config.vector_data_type != "auto") {
    try {
      (void)parse_vector_dtype(config.vector_data_type);
    } catch (const std::exception& e) {
      lib_failure(str{"--vector-data-type must be auto, float32, uint8, or int8: "} + e.what());
    }
  }
  if (config.partition_strategy != "balanced" &&
      config.partition_strategy != "bfs" &&
      config.partition_strategy != "metis")
    lib_failure("--partition-strategy must be balanced, bfs, or metis");
  if (config.partition_max_degree == 0)
    lib_failure("--partition-max-degree must be > 0");
  if (config.partition_imbalance < 1.0)
    lib_failure("--partition-imbalance must be >= 1.0");
  if (config.use_rabitq && config.ip_distance)
    lib_failure("--use-rabitq currently supports L2 distance only");
  if (config.rabitq_cache_format != "budget" && config.rabitq_cache_format != "full")
    lib_failure("--rabitq-cache-format must be budget or full");
  if (!vamana::parse_storage_format(config.storage_format))
    lib_failure("--storage-format must be vamana_aos_v1 or vamana_compact_v1");
  if (config.build_gpu_tiered_index && !config.use_rabitq)
    lib_failure("--gpu-tiered-index requires --use-rabitq");
  if (config.build_gpu_tiered_index &&
      (config.gpu_entry_points == 0 || config.gpu_entry_points > 512))
    lib_failure("--gpu-entry-points must be in [1, 512]");
  if (config.build_gpu_tiered_index && config.storage_format != "vamana_compact_v1")
    lib_failure("--gpu-tiered-index requires --storage-format=vamana_compact_v1");
  return config;
}

}  // namespace tools::vamana_offline
