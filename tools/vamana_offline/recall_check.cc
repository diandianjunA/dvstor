#include "tools/vamana_offline/recall_check.hh"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_set>

#include <library/utils.hh>

namespace tools::vamana_offline {
namespace {

struct QuerySet {
  u32 count{0};
  u32 dim{0};
  vec<float> vectors;
};

struct GroundTruth {
  u32 query_count{0};
  u32 topk{0};
  vec<u32> ids;
};

QuerySet read_queries(const filepath_t& path, u32 expected_dim) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    lib_failure("cannot open query file: " + path.string());
  }

  QuerySet queries;
  if (!input.read(reinterpret_cast<char*>(&queries.count),
                  sizeof(queries.count)) ||
      !input.read(reinterpret_cast<char*>(&queries.dim),
                  sizeof(queries.dim))) {
    lib_failure("query file header is truncated: " + path.string());
  }
  if (queries.count == 0 || queries.dim == 0 ||
      queries.dim != expected_dim) {
    lib_failure("query file has an invalid count or dimension: " +
                path.string());
  }
  const auto dtype = infer_vector_dtype_from_path(path);
  if (!dtype.has_value()) {
    lib_failure(
      "query suffix must be .fbin, .u8bin, or .i8bin: " + path.string());
  }
  const u64 component_bytes = vector_dtype_component_size(*dtype);
  const u64 components = static_cast<u64>(queries.count) * queries.dim;
  if (queries.dim != 0 &&
      components / queries.dim != queries.count) {
    lib_failure("query component count overflows");
  }
  if (components > (std::numeric_limits<u64>::max() - 8) /
                     component_bytes) {
    lib_failure("query file byte count overflows");
  }
  const u64 payload_bytes = components * component_bytes;
  const u64 expected_bytes = 8 + payload_bytes;
  std::error_code size_error;
  const u64 actual_bytes = std::filesystem::file_size(path, size_error);
  if (size_error || actual_bytes != expected_bytes) {
    lib_failure("query file size does not match its header: " +
                path.string());
  }
  if (payload_bytes > std::numeric_limits<size_t>::max() ||
      payload_bytes >
        static_cast<u64>(std::numeric_limits<std::streamsize>::max())) {
    lib_failure("query payload exceeds host I/O limits");
  }
  if (components > std::numeric_limits<size_t>::max() / sizeof(float)) {
    lib_failure("decoded query matrix exceeds host address space");
  }

  vec<byte_t> raw(static_cast<size_t>(payload_bytes));
  if (!input.read(reinterpret_cast<char*>(raw.data()),
                  static_cast<std::streamsize>(payload_bytes))) {
    lib_failure("query payload is truncated: " + path.string());
  }
  queries.vectors.resize(static_cast<size_t>(components));
  const size_t raw_row_bytes =
    vector_dtype_bytes(*dtype, queries.dim);
  for (u32 row = 0; row < queries.count; ++row) {
    const byte_t* source =
      raw.data() + static_cast<size_t>(row) * raw_row_bytes;
    float* destination =
      queries.vectors.data() + static_cast<size_t>(row) * queries.dim;
    decode_storage_vector_to_float(
      source, *dtype, queries.dim, destination);
    for (u32 dimension = 0; dimension < queries.dim; ++dimension) {
      if (!floating_value_is_finite(destination[dimension])) {
        lib_failure("query file contains a non-finite component: " +
                    path.string());
      }
    }
  }
  return queries;
}

GroundTruth read_groundtruth(const filepath_t& path, u32 expected_queries,
                             size_t vector_count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    lib_failure("cannot open groundtruth file: " + path.string());
  }

  GroundTruth groundtruth;
  if (!input.read(reinterpret_cast<char*>(&groundtruth.query_count),
                  sizeof(groundtruth.query_count)) ||
      !input.read(reinterpret_cast<char*>(&groundtruth.topk),
                  sizeof(groundtruth.topk))) {
    lib_failure("groundtruth header is truncated: " + path.string());
  }
  if (groundtruth.query_count == 0 || groundtruth.topk == 0 ||
      groundtruth.query_count != expected_queries) {
    lib_failure("groundtruth count/top-k mismatch: " + path.string());
  }
  const u64 id_count =
    static_cast<u64>(groundtruth.query_count) * groundtruth.topk;
  if (groundtruth.topk != 0 &&
      id_count / groundtruth.topk != groundtruth.query_count) {
    lib_failure("groundtruth ID count overflows");
  }
  if (id_count > (std::numeric_limits<u64>::max() - 8) / sizeof(u32)) {
    lib_failure("groundtruth file byte count overflows");
  }
  const u64 payload_bytes = id_count * sizeof(u32);
  const u64 expected_bytes = 8 + payload_bytes;
  std::error_code size_error;
  const u64 actual_bytes = std::filesystem::file_size(path, size_error);
  if (size_error || actual_bytes != expected_bytes) {
    lib_failure("groundtruth file size does not match its header: " +
                path.string());
  }
  if (id_count > std::numeric_limits<size_t>::max() ||
      payload_bytes >
        static_cast<u64>(std::numeric_limits<std::streamsize>::max())) {
    lib_failure("groundtruth payload exceeds host I/O limits");
  }

  groundtruth.ids.resize(static_cast<size_t>(id_count));
  if (!input.read(reinterpret_cast<char*>(groundtruth.ids.data()),
                  static_cast<std::streamsize>(payload_bytes))) {
    lib_failure("groundtruth payload is truncated: " + path.string());
  }
  for (u32 id : groundtruth.ids) {
    if (id >= vector_count) {
      lib_failure(
        "groundtruth contains an ID outside the indexed vector range");
    }
  }
  return groundtruth;
}

size_t count_hits(VamanaGraph& graph,
                  const Dataset& dataset,
                  const QuerySet& queries,
                  const GroundTruth& groundtruth,
                  u32 eval_k,
                  u32 search_beam) {
  size_t total_hits = 0;
  for (u32 qi = 0; qi < queries.count; ++qi) {
    const float* qvec = queries.vectors.data() + static_cast<size_t>(qi) * queries.dim;
    const auto results = beam_search_float_query(graph, dataset, qvec, search_beam);
    const u32* gt_row = groundtruth.ids.data() + static_cast<size_t>(qi) * groundtruth.topk;
    const std::unordered_set<u32> gt_set(gt_row, gt_row + eval_k);

    for (size_t i = 0; i < eval_k && i < results.size(); ++i) {
      if (gt_set.count(results[i].second)) {
        ++total_hits;
      }
    }
  }
  return total_hits;
}

}  // namespace

void preflight_optional_recall_inputs(
    const Dataset& dataset, const VamanaBuildConfig& config) {
  if (config.query_path.empty() != config.groundtruth_path.empty()) {
    lib_failure(
      "--query-path and --groundtruth-path must be supplied together");
  }
  if (config.query_path.empty()) return;
  const QuerySet queries = read_queries(config.query_path, dataset.dim);
  (void)read_groundtruth(
    config.groundtruth_path, queries.count, dataset.size());
  std::cerr << "offline recall inputs passed exact format/range preflight\n";
}

void run_optional_recall_check(VamanaGraph& graph,
                               const Dataset& dataset,
                               const VamanaBuildConfig& config) {
  if (config.query_path.empty() != config.groundtruth_path.empty()) {
    lib_failure(
      "--query-path and --groundtruth-path must be supplied together");
  }
  if (config.query_path.empty()) return;

  std::cerr << "\n=== Recall Test ===\n";
  const QuerySet queries = read_queries(config.query_path, dataset.dim);
  std::cerr << "queries: " << queries.count << " x " << queries.dim << "\n";

  const GroundTruth groundtruth = read_groundtruth(
    config.groundtruth_path, queries.count, dataset.size());
  std::cerr << "ground truth: " << groundtruth.query_count << " queries x top-" << groundtruth.topk << "\n";

  for (u32 eval_k : {1u, 5u, 10u}) {
    if (eval_k > groundtruth.topk) {
      continue;
    }

    const size_t total_hits =
        count_hits(graph, dataset, queries, groundtruth, eval_k, config.beam_width);
    const double recall = static_cast<double>(total_hits) / (static_cast<double>(queries.count) * eval_k);
    std::cerr << "recall@" << eval_k << " = " << std::fixed << std::setprecision(4) << recall
              << " (" << total_hits << "/" << (queries.count * eval_k) << ")\n";
  }
  std::cerr << "=== End Recall Test ===\n\n";
}

}  // namespace tools::vamana_offline
