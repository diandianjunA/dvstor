#include <cstdlib>
#include <iostream>

#include <boost/program_options.hpp>

#include "tools/vamana_offline/metis_repartitioner.hh"

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  tools::vamana_offline::MetisRepartitionOptions options;
  po::options_description description{
      "Repartition a complete schema-16 balanced index with 64-bit METIS"};
  description.add_options()("help,h", "Show help")(
      "input-prefix", po::value<filepath_t>(&options.input_prefix)->required(),
      "Existing complete schema-16 balanced index prefix")(
      "output-prefix",
      po::value<filepath_t>(&options.output_prefix)->required(),
      "New METIS index prefix (in-place conversion is forbidden)")(
      "data-path", po::value<filepath_t>(&options.data_path),
      "Base vector file; defaults to source metadata data_file")(
      "reuse-model", po::value<filepath_t>(&options.reuse_model),
      "OPQ/PQ model; defaults to the source schema-16 model")(
      "partition-max-degree",
      po::value<u32>(&options.partition_max_degree)->default_value(0),
      "Edges per node exposed to METIS; 0 reuses source metadata")(
      "partition-imbalance",
      po::value<f64>(&options.partition_imbalance)->default_value(0.0),
      "METIS imbalance >=1; 0 reuses source metadata")(
      "threads", po::value<u32>(&options.threads)->default_value(16),
      "PQ encoder threads in [1,32]")(
      "pq-chunk-vectors",
      po::value<u32>(&options.pq_chunk_vectors)->default_value(32768),
      "Vectors per PQ re-encoding chunk")(
      "graph-only", po::bool_switch(&options.graph_only),
      "Stop after publishing the resumable schema-15 METIS graph")(
      "validate-only", po::bool_switch(&options.validate_only),
      "Fully validate and reconstruct the source graph without writing");

  try {
    po::variables_map variables;
    po::store(po::parse_command_line(argc, argv, description), variables);
    if (variables.count("help")) {
      std::cout << description << '\n';
      return EXIT_SUCCESS;
    }
    po::notify(variables);
    const auto result =
        tools::vamana_offline::repartition_schema16_index(options);
    std::cout << (options.validate_only ? "Source validation passed"
                                        : "METIS repartition passed")
              << ": nodes=" << result.node_count
              << " edges=" << result.edge_count << " shards=" << result.shards
              << " graph_written=" << (result.graph_written ? "yes" : "no")
              << " pq_built=" << (result.pq_built ? "yes" : "no")
              << " extent_built=" << (result.extent_built ? "yes" : "no")
              << " resumed=" << (result.resumed ? "yes" : "no") << '\n';
    if (!options.validate_only) {
      std::cout << "Output metadata: " << result.metadata_file << '\n';
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "vamana_metis_repartitioner: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
