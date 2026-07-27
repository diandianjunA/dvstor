#include <cstdlib>
#include <iostream>

#include <boost/program_options.hpp>

#include "tools/vamana_offline/graph_extent_indexer.hh"

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  tools::vamana_offline::GraphExtentIndexOptions options;
  po::options_description description{
    "Build the global-ordinal GPU graph live-extent sidecar"};
  description.add_options()
    ("help,h", "Show help")
    ("index-prefix",
     po::value<filepath_t>(&options.index_prefix)->required(),
     "Existing schema-16 tagged Vamana index prefix")
    ("output", po::value<filepath_t>(&options.output),
     "Output path; defaults to <index-prefix>.gextent8")
    ("chunk-records",
     po::value<u32>(&options.chunk_records)
       ->default_value(options.chunk_records),
     "Complete graph records validated per sequential I/O chunk")
    ("overwrite", po::bool_switch(&options.overwrite),
     "Atomically replace an existing graph extent sidecar");
  try {
    po::variables_map variables;
    po::store(
      po::parse_command_line(argc, argv, description), variables);
    if (variables.count("help")) {
      std::cout << description << '\n';
      return EXIT_SUCCESS;
    }
    po::notify(variables);
    const auto result =
      tools::vamana_offline::build_graph_extent_index(options);
    std::cout << "GPU graph extent sidecar built: output="
              << result.output
              << " nodes=" << result.node_count
              << " payload_bytes=" << result.payload_bytes
              << " graph_bytes_validated="
              << result.graph_bytes_validated
              << " maximum_class=" << result.maximum_class
              << " payload_checksum=" << result.payload_checksum
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "vamana_graph_extent_indexer: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
