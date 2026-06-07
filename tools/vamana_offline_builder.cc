#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_set>

#include <cuda_runtime.h>

#include <library/utils.hh>

#include "gpu/gpu_kernel_launcher.hh"
#include "tools/vamana_offline/config.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/progress.hh"
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
            << " alpha=" << config.alpha
            << " vector_data_type=" << vector_dtype_name(dataset.dtype) << "\n";
  std::cerr << "partition_strategy=" << config.partition_strategy
            << " partition_max_degree=" << config.partition_max_degree
            << " partition_imbalance=" << config.partition_imbalance << "\n";
  std::cerr << "skip_sanity_check=" << (config.skip_sanity_check ? "true" : "false") << "\n";

  const auto build_start = std::chrono::steady_clock::now();

  // Initialize VamanaNode static storage
  VamanaNode::init_static_storage(dataset.dim, config.R, dataset.dtype);

  // GPU initialization
  const size_t num_threads = effective_thread_count(config.threads);
  std::unique_ptr<BuilderGpuContext[]> gpu_contexts;
  size_t num_gpu_contexts = 0;
  void* d_base_vectors = nullptr;

  if (!config.no_gpu && !config.ip_distance) {
    gpu::gpu_init(config.gpu_device);
    const size_t gpu_budget_bytes = static_cast<size_t>(config.gpu_memory_gb * 1024.0 * 1024.0 * 1024.0);
    const size_t raw_bytes = dataset.raw_vectors.size();
    const size_t workspace_streams = config.build_gpu_streams == 0
        ? std::min<size_t>(num_threads, 32)
        : static_cast<size_t>(config.build_gpu_streams);
    const size_t workspace_bytes = workspace_streams *
        (static_cast<size_t>(config.beam_width) * (sizeof(u32) + sizeof(float)) + 4096);
    if (raw_bytes + workspace_bytes <= gpu_budget_bytes) {
      d_base_vectors = gpu::gpu_malloc(raw_bytes);
      cudaStream_t upload_stream = gpu::gpu_stream_create();
      gpu::gpu_memcpy_h2d_async(d_base_vectors, dataset.raw_vectors.data(), raw_bytes, upload_stream);
      gpu::gpu_stream_synchronize(upload_stream);
      gpu::gpu_stream_destroy(upload_stream);
      num_gpu_contexts = workspace_streams;
      gpu_contexts = std::make_unique<BuilderGpuContext[]>(num_gpu_contexts);
      for (size_t i = 0; i < num_gpu_contexts; ++i) {
        gpu_contexts[i].init(dataset.dim, config.beam_width, dataset.dtype, d_base_vectors);
      }
      std::cerr << "GPU: device " << config.gpu_device
                << " resident_raw_base=true streams=" << num_gpu_contexts
                << " raw_bytes=" << raw_bytes
                << " budget_bytes=" << gpu_budget_bytes << "\n";
    } else {
      std::cerr << "GPU: disabled (raw dataset + workspace exceeds --gpu-memory-gb budget: raw="
                << raw_bytes << " workspace=" << workspace_bytes
                << " budget=" << gpu_budget_bytes << ")\n";
      gpu::gpu_shutdown();
    }
  } else {
    std::cerr << "GPU: disabled"
              << (config.ip_distance ? " (IP distance not supported on GPU)" : "")
              << "\n";
  }

  // Step 1: Build Vamana graph
  VamanaGraph graph;
  build_vamana_graph(graph, dataset, config, gpu_contexts.get(), num_gpu_contexts);

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
        auto results = beam_search_float_query(graph, dataset, qvec, search_beam, config.ip_distance);

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

  // Step 2: Serialize to shard files
  write_vamana_shards(graph, dataset, config, output_prefix);

  // GPU cleanup
  for (size_t i = 0; i < num_gpu_contexts; ++i) gpu_contexts[i].destroy();
  gpu_contexts.reset();
  if (d_base_vectors) gpu::gpu_free(d_base_vectors);
  if (num_gpu_contexts > 0 || d_base_vectors) gpu::gpu_shutdown();

  const auto build_end = std::chrono::steady_clock::now();
  const auto seconds = std::chrono::duration_cast<std::chrono::duration<double>>(build_end - build_start).count();
  std::cerr << "offline build finished in " << seconds << " seconds\n";

  return EXIT_SUCCESS;
}
