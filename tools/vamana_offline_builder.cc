#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_set>

#include <library/utils.hh>

#include "gpu/gpu_kernel_launcher.hh"
#include "tools/vamana_offline/config.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/progress.hh"
#include "tools/vamana_offline/rabitq.hh"
#include "tools/vamana_offline/shard_writer.hh"
#include "vamana/vamana_node.hh"

using namespace tools::vamana_offline;

int main(int argc, char** argv) {
  const VamanaBuildConfig config = parse_configuration(argc, argv);
  const Dataset dataset = read_dataset(config);
  const filepath_t output_prefix =
      config.output_prefix.empty()
          ? default_vamana_prefix(dataset.source_file, config.R, config.beam_width)
          : config.output_prefix;

  std::cerr << "output prefix: " << output_prefix << "\n";
  std::cerr << "memory nodes: " << config.num_memory_nodes << "\n";
  std::cerr << "threads: " << effective_thread_count(config.threads) << "\n";
  std::cerr << "R=" << config.R << " construction_beam_width=" << config.beam_width
            << " alpha=" << config.alpha << " rabitq_bits=" << config.rabitq_bits
            << " node_layout=" << config.node_layout << "\n";

  const auto build_start = std::chrono::steady_clock::now();

  // Initialize VamanaNode static storage
  VamanaNode::init_static_storage(
      dataset.dim, config.R, config.rabitq_bits, VamanaNode::parse_layout(config.node_layout));

  // Select distance function
  DistFn dist_fn = config.ip_distance ? ip_distance : l2_squared;

  // GPU initialization
  const size_t num_threads = effective_thread_count(config.threads);
  std::unique_ptr<BuilderGpuContext[]> gpu_contexts;
  size_t num_gpu_contexts = 0;

  if (!config.no_gpu && !config.ip_distance) {
    gpu::gpu_init(config.gpu_device);
    num_gpu_contexts = num_threads;
    gpu_contexts = std::make_unique<BuilderGpuContext[]>(num_gpu_contexts);
    for (size_t i = 0; i < num_gpu_contexts; ++i) {
      gpu_contexts[i].init(dataset.dim, config.beam_width, config.R);
    }
    std::cerr << "GPU: device " << config.gpu_device
              << " (" << num_gpu_contexts << " streams)\n";
  } else {
    std::cerr << "GPU: disabled"
              << (config.ip_distance ? " (IP distance not supported on GPU)" : "")
              << "\n";
  }

  // Step 1: Build Vamana graph
  VamanaGraph graph;
  build_vamana_graph(graph, dataset, config, dist_fn,
                     gpu_contexts.get(), num_gpu_contexts);

  // Optional: recall test with external queries and ground truth
  if (!config.query_path.empty() && !config.groundtruth_path.empty()) {
    std::cerr << "\n=== Recall Test ===\n";

    // Read queries (.fbin format: u32 num_queries, u32 dim, then float32 data)
    std::ifstream qfile(config.query_path, std::ios::binary);
    if (!qfile) lib_failure("cannot open query file: " + config.query_path.string());
    u32 n_queries, q_dim;
    qfile.read(reinterpret_cast<char*>(&n_queries), 4);
    qfile.read(reinterpret_cast<char*>(&q_dim), 4);
    if (q_dim != dataset.dim) lib_failure("query dim mismatch");
    vec<float> query_vecs(static_cast<size_t>(n_queries) * q_dim);
    qfile.read(reinterpret_cast<char*>(query_vecs.data()), query_vecs.size() * sizeof(float));
    qfile.close();
    std::cerr << "queries: " << n_queries << " x " << q_dim << "\n";

    // Read ground truth (.bin format: u32 n_queries, u32 topk, then u32 IDs)
    std::ifstream gtfile(config.groundtruth_path, std::ios::binary);
    if (!gtfile) lib_failure("cannot open groundtruth file: " + config.groundtruth_path.string());
    u32 gt_n, gt_k;
    gtfile.read(reinterpret_cast<char*>(&gt_n), 4);
    gtfile.read(reinterpret_cast<char*>(&gt_k), 4);
    if (gt_n != n_queries) lib_failure("groundtruth count mismatch");
    vec<u32> gt_ids(static_cast<size_t>(gt_n) * gt_k);
    gtfile.read(reinterpret_cast<char*>(gt_ids.data()), gt_ids.size() * sizeof(u32));
    gtfile.close();
    std::cerr << "ground truth: " << gt_n << " queries x top-" << gt_k << "\n";

    // Run beam_search for each query and compute recall
    const u32 search_beam = config.beam_width;
    for (u32 eval_k : {1u, 5u, 10u}) {
      if (eval_k > gt_k) continue;

      size_t total_hits = 0;
      for (u32 qi = 0; qi < n_queries; ++qi) {
        const float* qvec = query_vecs.data() + static_cast<size_t>(qi) * q_dim;
        auto results = beam_search(graph, dataset, qvec, search_beam, dist_fn);

        // Ground truth for this query
        const u32* gt_row = gt_ids.data() + static_cast<size_t>(qi) * gt_k;
        std::unordered_set<u32> gt_set(gt_row, gt_row + eval_k);

        size_t hits = 0;
        for (size_t i = 0; i < eval_k && i < results.size(); ++i) {
          if (gt_set.count(results[i].second)) ++hits;
        }
        total_hits += hits;
      }

      double recall = static_cast<double>(total_hits) / (static_cast<double>(n_queries) * eval_k);
      std::cerr << "recall@" << eval_k << " = " << std::fixed << std::setprecision(4) << recall
                << " (" << total_hits << "/" << (n_queries * eval_k) << ")\n";
    }
    std::cerr << "=== End Recall Test ===\n\n";
  }

  // Step 2: Initialize RaBitQ and quantize all vectors
  RaBitQState rabitq_state = init_rabitq(dataset, config.rabitq_bits, config.seed);

  vec<vec<byte_t>> rabitq_data(dataset.ids.size());
  {
    ProgressReporter progress{"Quantizing vectors (RaBitQ)", dataset.ids.size()};
    parallel_for(0, dataset.ids.size(), config.threads,
                 [&](size_t i, size_t) {
                   rabitq_data[i].resize(rabitq_state.total_rabitq_bytes);
                   rabitq_quantize_vector(dataset.vector(i), rabitq_state, rabitq_data[i].data());
                   progress.increment();
                 });
    progress.finish();
  }

  // Step 3: Serialize to shard files
  write_vamana_shards(graph, dataset, config, rabitq_state, rabitq_data, output_prefix);

  // GPU cleanup
  for (size_t i = 0; i < num_gpu_contexts; ++i) gpu_contexts[i].destroy();
  gpu_contexts.reset();
  if (num_gpu_contexts > 0) gpu::gpu_shutdown();

  const auto build_end = std::chrono::steady_clock::now();
  const auto seconds = std::chrono::duration_cast<std::chrono::duration<double>>(build_end - build_start).count();
  std::cerr << "offline build finished in " << seconds << " seconds\n";

  return EXIT_SUCCESS;
}
