#include <cstdlib>
#include <iostream>

#include <boost/program_options.hpp>

#include "tools/legacy_index/migrator.hh"
#include "tools/vamana_offline/pq_indexer.hh"

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  tools::legacy_index::MigrationOptions migration;
  tools::vamana_offline::PqIndexOptions pq;
  bool schema_only = false;
  po::options_description description{"DVSTOR legacy-to-GPU index converter"};
  description.add_options()
    ("help,h", "Show help")
    ("source-prefix", po::value<filepath_t>(&migration.source_prefix)->required(),
     "Schema-13 source index prefix")
    ("output-prefix", po::value<filepath_t>(&migration.output_prefix)->required(),
     "New schema-14 output index prefix")
    ("io-threads", po::value<u32>(&migration.io_threads)->default_value(0),
     "Concurrent shard migration threads; 0 uses available cores")
    ("chunk-nodes", po::value<u32>(&migration.chunk_nodes)->default_value(migration.chunk_nodes),
     "Nodes processed per sequential I/O chunk")
    ("schema-only", po::bool_switch(&schema_only),
     "Only compact the legacy schema; do not train or encode OPQ/PQ16")
    ("reuse-model", po::value<filepath_t>(&pq.reuse_model),
     "Reuse a compatible PQ model")
    ("train-samples", po::value<u32>(&pq.train_samples)->default_value(pq.train_samples),
     "Vectors used for OPQ/PQ training")
    ("opq-iterations", po::value<u32>(&pq.opq_iterations)->default_value(pq.opq_iterations),
     "OPQ outer iterations")
    ("pq-iterations", po::value<u32>(&pq.pq_iterations)->default_value(pq.pq_iterations),
     "PQ k-means iterations")
    ("encode-chunk-vectors", po::value<u32>(&pq.chunk_vectors)->default_value(pq.chunk_vectors),
     "Vectors encoded per sequential PQ chunk")
    ("entry-points", po::value<u32>(&pq.entry_points)->default_value(pq.entry_points),
     "GPU graph entry points")
    ("threads", po::value<u32>(&pq.threads)->default_value(0),
     "PQ training and encoding threads; 0 uses available cores")
    ("seed", po::value<u64>(&pq.seed)->default_value(pq.seed), "Training seed")
    ("overwrite", po::bool_switch(&migration.overwrite), "Replace converter outputs");
  try {
    po::variables_map variables;
    po::store(po::parse_command_line(argc, argv, description), variables);
    if (variables.count("help")) {
      std::cout << description << '\n';
      return EXIT_SUCCESS;
    }
    po::notify(variables);
    const auto result = tools::legacy_index::migrate_schema13_index(migration);
    if (!schema_only) {
      pq.index_prefix = result.output_prefix;
      pq.overwrite = migration.overwrite;
      (void)tools::vamana_offline::build_pq_index(pq);
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "vamana_legacy_index_converter: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
