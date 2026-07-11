#include <cstdlib>
#include <iostream>

#include <boost/program_options.hpp>

#include "tools/vamana_offline/pq_indexer.hh"

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  tools::vamana_offline::PqIndexOptions options;
  po::options_description description{"DVSTOR OPQ/PQ16 indexer"};
  description.add_options()
    ("help,h", "Show help")
    ("index-prefix", po::value<filepath_t>(&options.index_prefix)->required(),
     "Schema-14 Vamana index prefix")
    ("reuse-model", po::value<filepath_t>(&options.reuse_model),
     "Reuse a compatible .pq16 model instead of training")
    ("subquantizers", po::value<u32>(&options.subquantizers)->default_value(16),
     "Number of 8-bit product quantizers")
    ("train-samples", po::value<u32>(&options.train_samples)->default_value(options.train_samples),
     "Number of sampled vectors used for OPQ/PQ training")
    ("opq-iterations", po::value<u32>(&options.opq_iterations)->default_value(options.opq_iterations),
     "OPQ outer iterations")
    ("pq-iterations", po::value<u32>(&options.pq_iterations)->default_value(options.pq_iterations),
     "PQ k-means iterations")
    ("chunk-vectors", po::value<u32>(&options.chunk_vectors)->default_value(options.chunk_vectors),
     "Vectors encoded per sequential shard chunk")
    ("entry-points", po::value<u32>(&options.entry_points)->default_value(options.entry_points),
     "GPU search entry points")
    ("threads", po::value<u32>(&options.threads)->default_value(0),
     "CPU threads; 0 uses hardware concurrency")
    ("seed", po::value<u64>(&options.seed)->default_value(options.seed), "Training seed")
    ("overwrite", po::bool_switch(&options.overwrite), "Replace existing PQ outputs");
  try {
    po::variables_map variables;
    po::store(po::parse_command_line(argc, argv, description), variables);
    if (variables.count("help")) {
      std::cout << description << '\n';
      return EXIT_SUCCESS;
    }
    po::notify(variables);
    (void)tools::vamana_offline::build_pq_index(options);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "vamana_pq_indexer: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
