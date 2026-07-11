#include <cstdlib>
#include <iostream>

#include <boost/program_options.hpp>

#include "tools/legacy_index/migrator.hh"

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  tools::legacy_index::MigrationOptions migration;
  po::options_description description{"DVSTOR schema-13 to schema-14 converter"};
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
    ("overwrite", po::bool_switch(&migration.overwrite), "Replace converter outputs");
  try {
    po::variables_map variables;
    po::store(po::parse_command_line(argc, argv, description), variables);
    if (variables.count("help")) {
      std::cout << description << '\n';
      return EXIT_SUCCESS;
    }
    po::notify(variables);
    (void)tools::legacy_index::migrate_schema13_index(migration);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "vamana_legacy_index_converter: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
