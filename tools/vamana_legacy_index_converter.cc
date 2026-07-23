#include <cstdlib>
#include <iostream>

#include <boost/program_options.hpp>

#include "tools/vamana_offline/legacy_index_converter.hh"

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  tools::vamana_offline::LegacyIndexConvertOptions options;
  po::options_description description{
    "Convert an immutable schema-15 compact-v1 index to tagged-v2"};
  description.add_options()
    ("help,h", "Show help")
    ("input-prefix", po::value<filepath_t>(&options.input_prefix)->required(),
     "Existing schema-15 vamana_compact_v1 index prefix")
    ("output-prefix", po::value<filepath_t>(&options.output_prefix)->required(),
     "New output prefix (in-place conversion is forbidden)")
    ("reuse-model", po::value<filepath_t>(&options.reuse_model),
     "Legacy OPQ/PQ model; defaults to metadata or <input>.pq<M>")
    ("subquantizers", po::value<u32>(&options.subquantizers)->default_value(0),
     "PQ subquantizers; 0 infers the legacy value")
    ("chunk-vectors", po::value<u32>(&options.chunk_nodes)
       ->default_value(options.chunk_nodes),
     "Vectors per PQ re-encoding chunk")
    ("threads", po::value<u32>(&options.threads)->default_value(0),
     "PQ encoder CPU threads; 0 uses the indexer default")
    ("dry-run", po::bool_switch(&options.dry_run),
     "Validate every fixed record and edge without writing")
    ("graph-only", po::bool_switch(&options.graph_only),
     "Stop at the schema-15 tagged-v2 intermediate index");

  try {
    po::variables_map variables;
    po::store(po::parse_command_line(argc, argv, description), variables);
    if (variables.count("help")) {
      std::cout << description << '\n';
      return EXIT_SUCCESS;
    }
    po::notify(variables);
    const auto result =
      tools::vamana_offline::convert_legacy_index(options);
    std::cout << (options.dry_run ? "Legacy index validation passed" :
                  "Legacy index conversion passed")
              << ": nodes=" << result.node_count
              << " edges=" << result.edge_count
              << " shards=" << result.shards
              << " input_bytes=" << result.input_bytes
              << " output_bytes=" << result.output_bytes
              << " graph_written=" << (result.wrote_graph ? "yes" : "no")
              << " pq_built=" << (result.built_pq ? "yes" : "no")
              << '\n';
    if (!options.dry_run) {
      std::cout << "Output metadata: " << result.metadata_file << '\n';
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "vamana_legacy_index_converter: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
