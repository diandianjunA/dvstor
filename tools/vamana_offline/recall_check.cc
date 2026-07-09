#include "tools/vamana_offline/recall_check.hh"

#include <fstream>
#include <iomanip>
#include <iostream>
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
  input.read(reinterpret_cast<char*>(&queries.count), sizeof(queries.count));
  input.read(reinterpret_cast<char*>(&queries.dim), sizeof(queries.dim));
  if (queries.dim != expected_dim) {
    lib_failure("query dim mismatch");
  }

  queries.vectors.resize(static_cast<size_t>(queries.count) * queries.dim);
  input.read(reinterpret_cast<char*>(queries.vectors.data()),
             static_cast<std::streamsize>(queries.vectors.size() * sizeof(float)));
  return queries;
}

GroundTruth read_groundtruth(const filepath_t& path, u32 expected_queries) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    lib_failure("cannot open groundtruth file: " + path.string());
  }

  GroundTruth groundtruth;
  input.read(reinterpret_cast<char*>(&groundtruth.query_count), sizeof(groundtruth.query_count));
  input.read(reinterpret_cast<char*>(&groundtruth.topk), sizeof(groundtruth.topk));
  if (groundtruth.query_count != expected_queries) {
    lib_failure("groundtruth count mismatch");
  }

  groundtruth.ids.resize(static_cast<size_t>(groundtruth.query_count) * groundtruth.topk);
  input.read(reinterpret_cast<char*>(groundtruth.ids.data()),
             static_cast<std::streamsize>(groundtruth.ids.size() * sizeof(u32)));
  return groundtruth;
}

size_t count_hits(VamanaGraph& graph,
                  const Dataset& dataset,
                  const QuerySet& queries,
                  const GroundTruth& groundtruth,
                  u32 eval_k,
                  u32 search_beam,
                  bool ip_distance) {
  size_t total_hits = 0;
  for (u32 qi = 0; qi < queries.count; ++qi) {
    const float* qvec = queries.vectors.data() + static_cast<size_t>(qi) * queries.dim;
    const auto results = beam_search_float_query(graph, dataset, qvec, search_beam, ip_distance);
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

void run_optional_recall_check(VamanaGraph& graph,
                               const Dataset& dataset,
                               const VamanaBuildConfig& config) {
  if (config.query_path.empty() || config.groundtruth_path.empty()) {
    return;
  }

  std::cerr << "\n=== Recall Test ===\n";
  const QuerySet queries = read_queries(config.query_path, dataset.dim);
  std::cerr << "queries: " << queries.count << " x " << queries.dim << "\n";

  const GroundTruth groundtruth = read_groundtruth(config.groundtruth_path, queries.count);
  std::cerr << "ground truth: " << groundtruth.query_count << " queries x top-" << groundtruth.topk << "\n";

  for (u32 eval_k : {1u, 5u, 10u}) {
    if (eval_k > groundtruth.topk) {
      continue;
    }

    const size_t total_hits =
        count_hits(graph, dataset, queries, groundtruth, eval_k, config.beam_width, config.ip_distance);
    const double recall = static_cast<double>(total_hits) / (static_cast<double>(queries.count) * eval_k);
    std::cerr << "recall@" << eval_k << " = " << std::fixed << std::setprecision(4) << recall
              << " (" << total_hits << "/" << (queries.count * eval_k) << ")\n";
  }
  std::cerr << "=== End Recall Test ===\n\n";
}

}  // namespace tools::vamana_offline
