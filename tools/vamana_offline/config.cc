#include "tools/vamana_offline/config.hh"

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <boost/program_options.hpp>
#include <library/utils.hh>

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
    ("rabitq-bits", po::value<u32>(&config.rabitq_bits)->default_value(config.rabitq_bits),
     "Bits per dimension for RaBitQ (1, 2, 4, or 8).")
    ("node-layout", po::value<str>(&config.node_layout)->default_value(config.node_layout),
     "Node layout: legacy or rabitq_search_block.")
    ("seed", po::value<i32>(&config.seed)->default_value(config.seed), "PRNG seed.")
    ("max-vectors", po::value<size_t>(&config.max_vectors)->default_value(config.max_vectors),
     "Maximum number of vectors to read.")
    ("ip-dist", po::bool_switch(&config.ip_distance), "Use inner-product distance instead of L2.")
    ("no-gpu", po::bool_switch(&config.no_gpu), "Disable GPU acceleration.")
    ("gpu-device", po::value<i32>(&config.gpu_device)->default_value(config.gpu_device),
     "CUDA device ID (default 0).")
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
  if (config.R == 0) lib_failure("--R must be > 0");
  if (config.rabitq_bits != 1 && config.rabitq_bits != 2 &&
      config.rabitq_bits != 4 && config.rabitq_bits != 8)
    lib_failure("--rabitq-bits must be 1, 2, 4, or 8");
  if (config.node_layout != "legacy" && config.node_layout != "rabitq_search_block")
    lib_failure("--node-layout must be legacy or rabitq_search_block");

  return config;
}

}  // namespace tools::vamana_offline
