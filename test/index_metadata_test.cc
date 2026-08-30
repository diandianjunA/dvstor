#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <unistd.h>

#include "gpu_search/index_format.hh"
#include "nlohmann/json.hh"
#include "service/index_metadata.hh"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

struct TemporaryDirectory {
  fs::path path = fs::temp_directory_path() /
    ("dvstor-index-metadata-test-" + std::to_string(::getpid()));

  TemporaryDirectory() {
    std::error_code error;
    fs::remove_all(path, error);
    fs::create_directories(path);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    fs::remove_all(path, error);
  }
};

json valid_metadata() {
  // dim=8 int8 -> 32-byte nodes. R=16 reserves two provisional slots,
  // yielding 160-byte graph records. Each shard contains three nodes.
  constexpr u64 graph_offset = 192;
  constexpr u64 static_dynamic_base = 704;
  constexpr u64 control_offset = 704;
  constexpr u64 code_offset = 4800;
  constexpr u64 code_region_bytes = 12;
  constexpr u64 dynamic_node_base = 4864;
  return {
    {"schema_version", 16u},
    {"distance", "l2"},
    {"dim", 8u},
    {"R", 16u},
    {"beam_width_construction", 32u},
    {"partition_max_degree", 8u},
    {"partition_cross_shard_ratio", 0.25},
    {"num_vectors", 6ull},
    {"num_memory_nodes", 2u},
    {"node_size", 32u},
    {"node_layout", "plain"},
    {"storage_format", "vamana_tagged_v2"},
    {"graph_hot_bytes", 24u},
    {"vector_offset", 24u},
    {"slot_incarnation_offset", 16u},
    {"remote_ptr_format", "tagged_inc24_shard6_off34x16_v1"},
    {"vector_data_type", "int8"},
    {"vector_component_size", 1u},
    {"vector_bytes", 8u},
    {"navigation_quantizer", "opq_pq"},
    {"navigation_code_bytes", 4u},
    {"pq_subquantizers", 4u},
    {"pq_bits", 8u},
    {"navigation_model_checksum", 77ull},
    {"hot_graph_entry_size", 160u},
    {"hot_graph_pointer_bytes", 8u},
    {"hot_graph_shard_bits", 1u},
    {"hot_graph_offsets", {graph_offset, graph_offset}},
    {"hot_graph_entry_counts", {3ull, 3ull}},
    {"hot_graph_dynamic_base_offsets",
     {static_dynamic_base, static_dynamic_base}},
    {"storage_control_remote_offsets", {control_offset, control_offset}},
    {"dynamic_node_base_offsets", {dynamic_node_base, dynamic_node_base}},
    {"hot_graph_dynamic_record_bytes", 208u},
    {"hot_graph_dynamic_hot_offset", 32u},
    {"dynamic_navigation_code_offset", 192u},
    {"dynamic_navigation_code_validation_bytes", 4u},
    {"dynamic_navigation_code_checksum_bytes", 4u},
    {"allocation_size", 208u},
    {"idmap_format", "owner_sharded_v2_bound"},
    {"centroid_state_format", "physical_shard_centroid_v2_bound"},
    {"index_build_fingerprint", 123ull},
    {"shard_build_fingerprints", {111ull, 222ull}},
    {"navigation_format", "opq_pq_graph_v1"},
    {"navigation_code_remote_offsets", {code_offset, code_offset}},
    {"navigation_code_region_bytes", {code_region_bytes, code_region_bytes}},
  };
}

void write_document(const fs::path& prefix, const std::string& document) {
  std::ofstream output(prefix.string() + ".meta.json",
                       std::ios::binary | std::ios::trunc);
  output.write(document.data(), static_cast<std::streamsize>(document.size()));
  output.close();
  assert(output.good());
}

void expect_rejected(const fs::path& prefix, const std::string& document) {
  write_document(prefix, document);
  service::index_metadata::Metadata output;
  output.dim = 777;
  std::string error;
  assert(!service::index_metadata::load_metadata(prefix, output, &error));
  assert(!error.empty());
  // Failure is transactional: no partially parsed fields escape to callers.
  assert(output.dim == 777);
}

void run_regressions() {
  TemporaryDirectory temporary;
  const fs::path prefix = temporary.path / "index";
  const json good = valid_metadata();
  write_document(prefix, good.dump(2) + "\n");

  service::index_metadata::Metadata loaded;
  std::string error;
  assert(service::index_metadata::load_metadata(prefix, loaded, &error));
  assert(error.empty());
  assert(loaded.schema_version == 16 && loaded.dim == 8 &&
         loaded.vector_dtype == VectorDType::int8 &&
         loaded.hot_graph_entry_counts == vec<u64>({3, 3}));
  gpu_search::format::View view;
  assert(gpu_search::format::synthesize_distributed_view(
    prefix, view, &error));
  assert(view.layout.num_nodes == 6 && view.shards.size() == 2);

  json corrupt = good;
  corrupt["dim"] = 8.0;
  expect_rejected(prefix, corrupt.dump());

  corrupt = good;
  corrupt["hot_graph_offsets"][0] = -1;
  expect_rejected(prefix, corrupt.dump());

  corrupt = good;
  corrupt["hot_graph_offsets"].push_back(192ull);
  expect_rejected(prefix, corrupt.dump());

  corrupt = good;
  corrupt["hot_graph_entry_counts"][0] =
    std::numeric_limits<u64>::max();
  expect_rejected(prefix, corrupt.dump());
  write_document(prefix, corrupt.dump());
  error.clear();
  assert(!gpu_search::format::synthesize_distributed_view(
    prefix, view, &error));
  assert(!error.empty());

  corrupt = good;
  corrupt["navigation_code_remote_offsets"][0] =
    std::numeric_limits<u64>::max();
  expect_rejected(prefix, corrupt.dump());

  corrupt = good;
  corrupt.erase("dynamic_node_base_offsets");
  expect_rejected(prefix, corrupt.dump());

  corrupt = good;
  corrupt["vector_data_type"] = "float32";
  expect_rejected(prefix, corrupt.dump());

  const std::string serialized = good.dump();
  expect_rejected(prefix, "{\"dim\":8," + serialized.substr(1));
  expect_rejected(prefix, serialized.substr(0, serialized.size() / 2));
  expect_rejected(prefix, serialized + " trailing-garbage");
  expect_rejected(prefix, "[]");
  expect_rejected(prefix, std::string((4u << 20) + 1, ' '));
}

}  // namespace

int main(int argc, char** argv) {
  run_regressions();
  for (int argument = 1; argument < argc; ++argument) {
    service::index_metadata::Metadata metadata;
    std::string error;
    if (!service::index_metadata::load_metadata(argv[argument], metadata,
                                                &error)) {
      std::cerr << argv[argument] << ": " << error << '\n';
      return 1;
    }
    gpu_search::format::View view;
    if (!gpu_search::format::synthesize_distributed_view(
          argv[argument], view, &error)) {
      std::cerr << argv[argument] << ": " << error << '\n';
      return 1;
    }
    std::cout << "validated schema=" << metadata.schema_version
              << " dim=" << metadata.dim << " R=" << metadata.R
              << " vectors=" << metadata.num_vectors
              << " shards=" << metadata.num_memory_nodes << '\n';
  }
  return 0;
}
