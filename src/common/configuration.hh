#pragma once

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <library/configuration.hh>

#include "constants.hh"
#include "index_path.hh"
#include "types.hh"
#include "vector_dtype.hh"

namespace configuration {

// struct used for sending serialized from CN to MN
struct Parameters {
  u32 num_threads{};
  bool gpu_persistent{};
  bool routing{};
  u32 qp_pool_size{1};
  u32 gpu_rdma_qps{};
};

class IndexConfiguration : public Configuration {
public:
  filepath_t data_path{};
  filepath_t index_prefix{};
  filepath_t server_index_file{};
  str query_suffix{};
  u32 num_threads{};
  u32 num_coroutines{};
  i32 seed{};
  bool disable_thread_pinning{};
  str label{};  // for labeling benchmarks

  // Vamana parameters
  u32 R{};                // max out-degree
  u32 beam_width{};       // beam width for search (replaces ef_search)
  u32 beam_width_construction{}; // beam width for insert (replaces ef_construction)
  f64 alpha{};            // RobustPrune alpha parameter
  u32 k{};
  u32 gpu_device{};       // CUDA device ID
  bool gpudirect_rdma{};  // Enable GPUDirect RDMA (read vectors directly into GPU buffers)
  bool enable_breakdown{true};  // Per-request fine-grained timing/counter collection
  bool observe_device_utilization{};  // CUDA-event instrumentation for motivation experiments
  u32 expansion_batch{1};   // Batch K beam expansions per iteration (1=serial)
  bool credit_aware_expansion{};  // Treat expansion_batch as a hardware-credit-driven upper bound
  u32 credit_aware_min_k{1};
  u32 credit_aware_max_k{};
  u32 credit_aware_target_candidates{};
  u32 credit_aware_max_lookahead{};
  bool credit_aware_cost_guard{};
  f64 credit_aware_cost_max_extra_ratio{1.05};
  u32 credit_aware_cost_probe_rounds{4};
  u32 rdma_qp_pool_size{};   // QPs per memory node per SharedContext (0=auto)
  str rdma_read_batch_mode{"adaptive"};
  u32 rdma_read_chain_size{};
  u32 rdma_read_max_inflight_wrs{};
  u32 query_batch_size{1};  // Fuse GPU across N queries (1=single query)
  str search_engine{"legacy"};
  str gpu_rdma_backend{"gpunetio"};
  u32 query_batch_min{16};
  u32 query_batch_target{64};
  u32 query_batch_max{256};
  u32 query_batch_wait_us{20};
  u32 gpu_page_cache_mb{};
  f64 gpu_page_cache_ratio{0.35};
  u32 gpu_hot_degree{16};
  u32 gpu_cold_expansions{8};
  u32 gpu_rdma_qps{4};
  u32 update_visibility_us{10000};
  f64 delta_max_ratio{0.01};
  u32 delta_budget_mb{1024};
  u32 merge_period_ms{60000};
  bool use_rabitq{};        // Use the local RaBitQ gate before exact beam insertion
  u32 rabitq_gate_width{18};
  u32 rabitq_gate_max_width{36};
  f64 rabitq_gate_margin{0.08};
  u32 rabitq_dynamic_budget_mb{64};
  u32 rabitq_coalesce_target{64};
  u32 rabitq_coalesce_min{32};
  u32 rabitq_coalesce_wait_us{6};
  u32 rabitq_warmup_exact_expansions{6};
  u32 rabitq_audit_period{12};
  bool rabitq_strict_recall{true};
  str vector_data_type{"auto"};
  str insert_execution{"compute"};
  u32 insert_workers{};
  u32 query_workers{};
  u32 insert_coroutines{};
  u32 query_coroutines{};
  u32 storage_id{0};
  vec<str> storage_peers;
  u32 storage_owner_batch_max{16};
  u32 storage_owner_batch_wait_us{250};
  u32 storage_owner_peer_rdma_tokens{8};
  u32 storage_owner_rpc_depth{8};
  u32 storage_owner_rpc_timeout_ms{30000};
  u32 storage_owner_construction_beam_width{128};
  u32 storage_owner_search_snapshot_batch{64};
  u32 storage_owner_prune_max_candidates{128};
  str storage_owner_update_mode{"exact"};
  u32 storage_owner_anchor_hints{4};
  u32 storage_owner_anchor_beam_width{64};
  u32 storage_owner_anchor_expand_cap{16};
  u32 storage_owner_anchor_remote_rescue_cap{4};
  bool storage_owner_local_stitch_sync_fast_path{true};
  str storage_owner_maintenance_mode{"off"};
  u32 storage_owner_maintenance_workers{0};
  str storage_owner_reverse_mode{"async"};
  u32 storage_owner_reverse_queue_depth{65536};
  u32 storage_owner_reverse_flush_us{200};
  u32 storage_owner_reverse_coalesce_max{256};

  // Legacy aliases for compatibility
  u32& ef_search = beam_width;
  u32& ef_construction = beam_width_construction;
  u32& m = R;

  bool store_index{};  // memory servers store the index; index is constructed from scratch; location is data_path
  bool load_index{};  // memory servers load index from file; cannot be used with store_index; location is data_path
  bool no_recall{};  // does not calculate the recall and thus requires no groundtruth
  bool ip_distance{};  // use the inner product distance rather than squared L2 norm

  bool routing{};

  u32 dim{};
  u32 max_vectors{1000000};

  // Memory size parameters (in GB)
  u32 cn_memory_gb{10};
  u32 mn_memory_gb{10};

public:
  IndexConfiguration(int argc, char** argv) {
    add_options();
    process_program_options(argc, argv);
    insert_execution = normalize_mode(insert_execution);
    storage_owner_reverse_mode = normalize_mode(storage_owner_reverse_mode);
    storage_owner_update_mode = normalize_mode(storage_owner_update_mode);
    storage_owner_maintenance_mode = normalize_mode(storage_owner_maintenance_mode);
    rdma_read_batch_mode = normalize_mode(rdma_read_batch_mode);
    search_engine = normalize_mode(search_engine);
    gpu_rdma_backend = normalize_mode(gpu_rdma_backend);

    validate_common_options(argv);
    if (!is_server) {
      validate_compute_node_options(argv);
    }

    operator<<(std::cerr, *this);
  }

private:
  void validate_common_options(char** argv) const {
    if (dim == 0 || R == 0 || max_vectors == 0 || num_coroutines == 0) {
      std::cerr << "[ERROR]: Parameters dim, R, max-vectors, and coroutines must be > 0" << std::endl;
      exit_with_help_message(argv);
    }
    if (R > std::numeric_limits<u8>::max()) {
      std::cerr << "[ERROR]: --R must be <= 255 because the on-wire node format stores edge_count in one byte"
                << std::endl;
      exit_with_help_message(argv);
    }
    if (beam_width_construction == 0 || query_batch_size == 0) {
      std::cerr << "[ERROR]: --beam-width-construction and --query-batch-size must be > 0" << std::endl;
      exit_with_help_message(argv);
    }
    if (cn_memory_gb == 0 || mn_memory_gb == 0) {
      std::cerr << "[ERROR]: --cn-memory and --mn-memory must be > 0" << std::endl;
      exit_with_help_message(argv);
    }
  }

  void add_options() {
    desc.add_options()("data-path,d",
                       po::value<filepath_t>(&data_path),
                       "Path to input directory containing the base vectors (\"base.fvecs\") and the \"query\" "
                       "directory (which contains the query and the groundtruth file).")(
      "index-prefix",
      po::value<filepath_t>(&index_prefix),
      "Path prefix of index shard files without the _nodeX_ofN.dat suffix. If omitted, the prefix is derived from "
      "data-path, M, and ef-construction.")(
      "server-index-file",
      po::value<filepath_t>(&server_index_file),
      "Path to a local DVSTOR index shard file that a memory node should load during startup.")(
      "threads,t", po::value<u32>(&num_threads), "Number of threads per compute node.")(
      "coroutines,C", po::value<u32>(&num_coroutines)->default_value(4), "Number of coroutines per compute thread.")(
      "disable-thread-pinning,p",
      po::bool_switch(&disable_thread_pinning)->default_value(false),
      "Disables pinning compute threads to physical cores if set.")(
      "seed", po::value<i32>(&seed)->default_value(1234), "Seed for PRNG; setting to -1 uses std::random_device.")(
      "label", po::value<str>(&label), "Optional label to identify benchmarks.")(
      "query-suffix,q", po::value<str>(&query_suffix), "Filename suffix for the query file.")(
      "store-index,s",
      po::bool_switch(&store_index),
      "Construct the index from scratch and the memory servers store the index to a file.")(
      "load-index,l",
      po::bool_switch(&load_index),
      "The index is not built, the memory servers load the index from a file.")(
      "routing", po::bool_switch(&routing), "Activate adaptive query routing.")(
      "no-recall", po::bool_switch(&no_recall), "No recall computation, ground truth file can be omitted.")(
      "ip-dist", po::bool_switch(&ip_distance), "Use the inner product distance rather than the squared L2 norm.")(
      "beam-width", po::value<u32>(&beam_width), "Beam width during search (replaces ef-search).")(
      "ef-search", po::value<u32>(&beam_width), "Alias for --beam-width.")(
      "beam-width-construction", po::value<u32>(&beam_width_construction)->default_value(200),
      "Beam width during construction (replaces ef-construction).")(
      "ef-construction", po::value<u32>(&beam_width_construction), "Alias for --beam-width-construction.")(
      "k,k", po::value<u32>(&k), "Number of k nearest neighbors.")(
      "R", po::value<u32>(&R)->default_value(64), "Maximum out-degree of Vamana graph.")(
      "m,m", po::value<u32>(&R), "Alias for --R (max out-degree).")(
      "alpha", po::value<f64>(&alpha)->default_value(1.2), "RobustPrune diversity factor.")
      ("vector-data-type", po::value<str>(&vector_data_type)->default_value(vector_data_type),
      "Storage dtype for full vectors: auto, float32, uint8, or int8.")(
      "insert-execution", po::value<str>(&insert_execution)->default_value(insert_execution),
      "Insert execution mode: compute or storage_owner.")(
      "insert-workers", po::value<u32>(&insert_workers)->default_value(0),
      "Dedicated insert worker threads. 0 keeps the built-in split.")(
      "query-workers", po::value<u32>(&query_workers)->default_value(0),
      "Dedicated query worker threads. 0 keeps the built-in split.")(
      "insert-coroutines", po::value<u32>(&insert_coroutines)->default_value(0),
      "Coroutines per insert worker. 0 uses the global coroutines value.")(
      "query-coroutines", po::value<u32>(&query_coroutines)->default_value(0),
      "Coroutines per query worker. 0 uses the built-in query default.")(
      "storage-id", po::value<u32>(&storage_id)->default_value(0),
      "Storage-node id used by storage_owner insert execution.")(
      "storage-peers", po::value<vec<str>>(&storage_peers)->multitoken(),
      "Ordered list of storage-peer endpoints used for storage_owner insert execution.")(
      "storage-owner-batch-max", po::value<u32>(&storage_owner_batch_max)->default_value(storage_owner_batch_max),
      "Maximum number of inserts grouped into one storage_owner batch.")(
      "storage-owner-batch-wait-us", po::value<u32>(&storage_owner_batch_wait_us)->default_value(storage_owner_batch_wait_us),
      "Maximum micro-batch wait in microseconds for storage_owner inserts.")(
      "storage-owner-peer-rdma-tokens",
      po::value<u32>(&storage_owner_peer_rdma_tokens)->default_value(storage_owner_peer_rdma_tokens),
      "Maximum storage-owner peer RDMA reads allowed per peer QP. Capped by the memory-node safety limit.")(
      "storage-owner-rpc-depth",
      po::value<u32>(&storage_owner_rpc_depth)->default_value(storage_owner_rpc_depth),
      "Maximum in-flight storage_owner insert batches per storage node.")(
      "storage-owner-rpc-timeout-ms",
      po::value<u32>(&storage_owner_rpc_timeout_ms)->default_value(storage_owner_rpc_timeout_ms),
      "Maximum time to wait for one storage_owner insert RPC response.")(
      "storage-owner-construction-beam-width",
      po::value<u32>(&storage_owner_construction_beam_width)->default_value(storage_owner_construction_beam_width),
      "Storage-owner online construction beam width. 0 uses --beam-width-construction unchanged.")(
      "storage-owner-search-snapshot-batch",
      po::value<u32>(&storage_owner_search_snapshot_batch)->default_value(storage_owner_search_snapshot_batch),
      "Maximum node snapshots read concurrently during storage-owner search/prune.")(
      "storage-owner-prune-max-candidates",
      po::value<u32>(&storage_owner_prune_max_candidates)->default_value(storage_owner_prune_max_candidates),
      "Maximum candidates considered by storage-owner robust-prune. 0 disables the cap.")(
      "storage-owner-update-mode",
      po::value<str>(&storage_owner_update_mode)->default_value(storage_owner_update_mode),
      "Storage-owner update search: exact or local_stitch.")(
      "storage-owner-anchor-hints",
      po::value<u32>(&storage_owner_anchor_hints)->default_value(storage_owner_anchor_hints),
      "Anchor entry points attached to each storage-owner mutation.")(
      "storage-owner-anchor-beam-width",
      po::value<u32>(&storage_owner_anchor_beam_width)->default_value(storage_owner_anchor_beam_width),
      "Maximum beam width for local-stitch storage-owner search.")(
      "storage-owner-anchor-expand-cap",
      po::value<u32>(&storage_owner_anchor_expand_cap)->default_value(storage_owner_anchor_expand_cap),
      "Maximum graph expansions in local-stitch foreground search.")(
      "storage-owner-anchor-remote-rescue-cap",
      po::value<u32>(&storage_owner_anchor_remote_rescue_cap)->default_value(storage_owner_anchor_remote_rescue_cap),
      "Maximum remote-node expansions in local-stitch foreground search.")(
      "storage-owner-local-stitch-sync-fast-path",
      po::value<bool>(&storage_owner_local_stitch_sync_fast_path)->default_value(storage_owner_local_stitch_sync_fast_path),
      "Use a synchronous worker-local foreground path for local-stitch storage-owner mutations with anchors.")(
      "storage-owner-maintenance-mode",
      po::value<str>(&storage_owner_maintenance_mode)->default_value(storage_owner_maintenance_mode),
      "Storage-owner background graph-quality maintenance: off or finalize.")(
      "storage-owner-maintenance-workers",
      po::value<u32>(&storage_owner_maintenance_workers)->default_value(storage_owner_maintenance_workers),
      "Background exact-finalization worker threads. 0 disables maintenance.")(
      "storage-owner-reverse-mode",
      po::value<str>(&storage_owner_reverse_mode)->default_value(storage_owner_reverse_mode),
      "Reverse-update completion mode for storage_owner inserts: async or sync.")(
      "storage-owner-reverse-queue-depth",
      po::value<u32>(&storage_owner_reverse_queue_depth)->default_value(storage_owner_reverse_queue_depth),
      "Maximum queued peer reverse-update requests per memory node.")(
      "storage-owner-reverse-flush-us",
      po::value<u32>(&storage_owner_reverse_flush_us)->default_value(storage_owner_reverse_flush_us),
      "Maximum worker-side coalescing wait for peer reverse updates in microseconds.")(
      "storage-owner-reverse-coalesce-max",
      po::value<u32>(&storage_owner_reverse_coalesce_max)->default_value(storage_owner_reverse_coalesce_max),
      "Maximum reverse-update operations coalesced by one peer worker batch.")(
      "gpu-device", po::value<u32>(&gpu_device)->default_value(0), "CUDA device ID.")(
      "gpudirect-rdma", po::bool_switch(&gpudirect_rdma)->default_value(false),
      "Enable GPUDirect RDMA on compute nodes (direct RDMA reads into GPU memory).")(
      "enable-breakdown", po::value<bool>(&enable_breakdown)->default_value(enable_breakdown),
      "Enable per-request fine-grained breakdown collection. Disable for performance-only runs.")(
      "observe-device-utilization",
      po::bool_switch(&observe_device_utilization)->default_value(false),
      "Collect CUDA-kernel and RDMA-wait utilization metrics (benchmark instrumentation).")(
      "expansion-batch,K", po::value<u32>(&expansion_batch)->default_value(1),
      "Number of beam nodes expanded per iteration.")(
      "credit-aware-expansion",
      po::bool_switch(&credit_aware_expansion)->default_value(false),
      "Enable resource-credit-driven per-query expansion scheduling; --expansion-batch becomes the hard cap.")(
      "credit-aware-min-k",
      po::value<u32>(&credit_aware_min_k)->default_value(credit_aware_min_k),
      "Minimum expansions issued by the credit-aware scheduler.")(
      "credit-aware-max-k",
      po::value<u32>(&credit_aware_max_k)->default_value(credit_aware_max_k),
      "Maximum expansions issued by the credit-aware scheduler. 0 uses --expansion-batch.")(
      "credit-aware-target-candidates",
      po::value<u32>(&credit_aware_target_candidates)->default_value(credit_aware_target_candidates),
      "Target generated frontier candidates per expansion round. 0 derives it from graph degree and max K.")(
      "credit-aware-max-lookahead",
      po::value<u32>(&credit_aware_max_lookahead)->default_value(credit_aware_max_lookahead),
      "Maximum pre-commit speculative neighbor reads. 0 derives it from max K.")(
      "credit-aware-cost-guard",
      po::bool_switch(&credit_aware_cost_guard)->default_value(false),
      "Use a query-local work-per-expansion guard to reject unprofitable K growth.")(
      "credit-aware-cost-max-extra-ratio",
      po::value<f64>(&credit_aware_cost_max_extra_ratio)->default_value(credit_aware_cost_max_extra_ratio),
      "Maximum work-per-expansion over the query-local min-K baseline before shrinking K.")(
      "credit-aware-cost-probe-rounds",
      po::value<u32>(&credit_aware_cost_probe_rounds)->default_value(credit_aware_cost_probe_rounds),
      "Minimum min-K rounds used to learn the query-local cost baseline.")(
      "rdma-qp-pool-size", po::value<u32>(&rdma_qp_pool_size)->default_value(rdma_qp_pool_size),
      "QPs per memory node per SharedContext. 0 selects the automatic pool size.")(
      "rdma-read-batch-mode",
      po::value<str>(&rdma_read_batch_mode)->default_value(rdma_read_batch_mode),
      "Vector RDMA batch scheduling mode: adaptive or legacy.")(
      "rdma-read-chain-size",
      po::value<u32>(&rdma_read_chain_size)->default_value(rdma_read_chain_size),
      "Maximum RDMA READ WRs per signaled chain. 0 derives it from device capabilities.")(
      "rdma-read-max-inflight-wrs",
      po::value<u32>(&rdma_read_max_inflight_wrs)->default_value(rdma_read_max_inflight_wrs),
      "Maximum outstanding bulk READ WRs per QP. 0 derives it from the send queue capacity.")(
      "query-batch-size", po::value<u32>(&query_batch_size)->default_value(1),
      "Fuse GPU/D2H across N queries processed in lockstep (1=disabled, 2-4=batch).")(
      "search-engine", po::value<str>(&search_engine)->default_value(search_engine),
      "Query execution engine: legacy or gpu_persistent.")(
      "gpu-rdma-backend", po::value<str>(&gpu_rdma_backend)->default_value(gpu_rdma_backend),
      "Remote fetch backend for gpu_persistent: gpunetio, verbs_proxy, or local.")(
      "query-batch-min", po::value<u32>(&query_batch_min)->default_value(query_batch_min),
      "Minimum query batch admitted to the persistent GPU scheduler under load.")(
      "query-batch-target", po::value<u32>(&query_batch_target)->default_value(query_batch_target),
      "Target query batch for the persistent GPU scheduler.")(
      "query-batch-max", po::value<u32>(&query_batch_max)->default_value(query_batch_max),
      "Maximum active query slots owned by the persistent GPU scheduler.")(
      "query-batch-wait-us", po::value<u32>(&query_batch_wait_us)->default_value(query_batch_wait_us),
      "Maximum admission wait before launching an under-filled GPU query batch.")(
      "gpu-page-cache-mb", po::value<u32>(&gpu_page_cache_mb)->default_value(gpu_page_cache_mb),
      "GPU graph-page cache size in MiB. 0 derives it from --gpu-page-cache-ratio.")(
      "gpu-page-cache-ratio", po::value<f64>(&gpu_page_cache_ratio)->default_value(gpu_page_cache_ratio),
      "Fraction of currently free GPU memory reserved for graph pages when cache MiB is zero.")(
      "gpu-hot-degree", po::value<u32>(&gpu_hot_degree)->default_value(gpu_hot_degree),
      "Per-node navigation edges resident on the GPU.")(
      "gpu-cold-expansions", po::value<u32>(&gpu_cold_expansions)->default_value(gpu_cold_expansions),
      "Maximum expanded nodes per query whose full adjacency page may be fetched remotely.")(
      "gpu-rdma-qps", po::value<u32>(&gpu_rdma_qps)->default_value(gpu_rdma_qps),
      "GPU data-path QPs created per memory node.")(
      "update-visibility-us", po::value<u32>(&update_visibility_us)->default_value(update_visibility_us),
      "Maximum mutation micro-batch visibility delay for the GPU delta index.")(
      "delta-max-ratio", po::value<f64>(&delta_max_ratio)->default_value(delta_max_ratio),
      "Trigger base consolidation when delta entries reach this fraction of base nodes.")(
      "delta-budget-mb", po::value<u32>(&delta_budget_mb)->default_value(delta_budget_mb),
      "GPU delta-index memory budget in MiB.")(
      "merge-period-ms", po::value<u32>(&merge_period_ms)->default_value(merge_period_ms),
      "Maximum interval between delta consolidation attempts.")(
      "use-rabitq", po::bool_switch(&use_rabitq)->default_value(false),
      "Use the local RaBitQ CPU gate; only exact distances enter the beam.")(
      "rabitq-gate-width", po::value<u32>(&rabitq_gate_width)->default_value(rabitq_gate_width),
      "Minimum cached candidates exactified per expansion.")(
      "rabitq-gate-max-width",
      po::value<u32>(&rabitq_gate_max_width)->default_value(rabitq_gate_max_width),
      "Maximum cached candidates exactified after margin expansion.")(
      "rabitq-gate-margin", po::value<f64>(&rabitq_gate_margin)->default_value(rabitq_gate_margin),
      "Relative margin around the gate-width cutoff.")(
      "rabitq-dynamic-budget-mb",
      po::value<u32>(&rabitq_dynamic_budget_mb)->default_value(rabitq_dynamic_budget_mb),
      "Fixed RaBitQ dynamic overlay budget in MiB.")(
      "rabitq-coalesce-target",
      po::value<u32>(&rabitq_coalesce_target)->default_value(rabitq_coalesce_target),
      "Target candidates per RaBitQ exactification flush.")(
      "rabitq-coalesce-min",
      po::value<u32>(&rabitq_coalesce_min)->default_value(rabitq_coalesce_min),
      "Minimum RDMA-friendly candidates exactified before strict recall widening stops.")(
      "rabitq-coalesce-wait-us",
      po::value<u32>(&rabitq_coalesce_wait_us)->default_value(rabitq_coalesce_wait_us),
      "Maximum RaBitQ coalescer wait in microseconds.")(
      "rabitq-warmup-exact-expansions",
      po::value<u32>(&rabitq_warmup_exact_expansions)->default_value(rabitq_warmup_exact_expansions),
      "Exactify all candidates for this many initial RaBitQ graph expansions.")(
      "rabitq-audit-period",
      po::value<u32>(&rabitq_audit_period)->default_value(rabitq_audit_period),
      "Exactify one full RaBitQ frontier every N graph expansions after warmup. 0 disables audit.")(
      "rabitq-strict-recall",
      po::value<bool>(&rabitq_strict_recall)->default_value(rabitq_strict_recall),
      "Widen uncertain small RaBitQ gates so recall is protected.")(
      "dim", po::value<u32>(&dim), "Vector dimension")(
      "max-vectors", po::value<u32>(&max_vectors)->default_value(1000000), "Max vectors capacity")(
      "cn-memory", po::value<u32>(&cn_memory_gb)->default_value(10), "Compute node local buffer size in GB")(
      "mn-memory", po::value<u32>(&mn_memory_gb)->default_value(10), "Memory node buffer size in GB");
  }

  void validate_compute_node_options(char** argv) const {
    if (num_threads == 0 || beam_width == 0 || k == 0 || dim == 0) {
      std::cerr << "[ERROR]: Parameters threads, beam-width (ef-search), k, and dim are required" << std::endl;
      exit_with_help_message(argv);
    }
    if (beam_width < k) {
      std::cerr << "[ERROR]: --beam-width must be >= k" << std::endl;
      exit_with_help_message(argv);
    }

    if (store_index && load_index) {
      std::cerr << "[ERROR]: --store-index and --load-index cannot be used in conjunction" << std::endl;
      exit_with_help_message(argv);
    }

    if (rdma_read_batch_mode != "adaptive" && rdma_read_batch_mode != "legacy") {
      std::cerr << "[ERROR]: --rdma-read-batch-mode must be adaptive or legacy" << std::endl;
      exit_with_help_message(argv);
    }

    if (search_engine != "legacy" && search_engine != "gpu_persistent") {
      std::cerr << "[ERROR]: --search-engine must be legacy or gpu_persistent" << std::endl;
      exit_with_help_message(argv);
    }
    if (gpu_rdma_backend != "gpunetio" && gpu_rdma_backend != "verbs_proxy" &&
        gpu_rdma_backend != "local") {
      std::cerr << "[ERROR]: --gpu-rdma-backend must be gpunetio, verbs_proxy, or local" << std::endl;
      exit_with_help_message(argv);
    }
    if (search_engine == "gpu_persistent" &&
        (query_batch_min == 0 || query_batch_min > query_batch_target ||
         query_batch_target > query_batch_max || query_batch_max > 4096 ||
         gpu_hot_degree == 0 || gpu_hot_degree > R || gpu_hot_degree > 32 ||
         gpu_cold_expansions > beam_width ||
         gpu_page_cache_ratio <= 0.0 || gpu_page_cache_ratio >= 0.9 ||
         gpu_rdma_qps == 0 || gpu_rdma_qps > 32 || update_visibility_us == 0 ||
         delta_max_ratio <= 0.0 || delta_max_ratio > 0.5 || delta_budget_mb == 0 ||
         merge_period_ms == 0 || beam_width > 256 || rabitq_gate_max_width > 64 ||
         !use_rabitq || ip_distance || insert_execution != "storage_owner" || routing)) {
      std::cerr << "[ERROR]: invalid gpu_persistent engine configuration" << std::endl;
      exit_with_help_message(argv);
    }
    if (search_engine == "gpu_persistent" && !load_index) {
      std::cerr << "[ERROR]: --search-engine=gpu_persistent currently requires --load-index" << std::endl;
      exit_with_help_message(argv);
    }

    if ((store_index || load_index) && index_prefix.empty() && data_path.empty()) {
      std::cerr << "[ERROR]: --data-path or --index-prefix is required when --load-index or --store-index is set"
                << std::endl;
      exit_with_help_message(argv);
    }

    if (vector_data_type != "auto") {
      try {
        (void)parse_vector_dtype(vector_data_type);
      } catch (const std::exception& e) {
        std::cerr << "[ERROR]: --vector-data-type must be auto, float32, uint8, or int8: "
                  << e.what() << std::endl;
        exit_with_help_message(argv);
      }
    }

    if (use_rabitq && ip_distance) {
      std::cerr << "[ERROR]: --use-rabitq currently supports L2 distance only" << std::endl;
      exit_with_help_message(argv);
    }
    if (rabitq_gate_width == 0 || rabitq_gate_max_width < rabitq_gate_width ||
        rabitq_gate_margin < 0.0 ||
        rabitq_coalesce_min == 0 || rabitq_coalesce_target < rabitq_coalesce_min) {
      std::cerr << "[ERROR]: invalid RaBitQ gate configuration" << std::endl;
      exit_with_help_message(argv);
    }

    if (expansion_batch == 0 ||
        credit_aware_min_k == 0 ||
        (credit_aware_max_k != 0 && credit_aware_max_k < credit_aware_min_k) ||
        (credit_aware_max_k != 0 && credit_aware_max_k > expansion_batch) ||
        credit_aware_max_lookahead > expansion_batch ||
        credit_aware_cost_max_extra_ratio < 1.0 ||
        credit_aware_cost_probe_rounds == 0) {
      std::cerr << "[ERROR]: invalid credit-aware expansion configuration" << std::endl;
      exit_with_help_message(argv);
    }

    if (insert_execution != "compute" && insert_execution != "storage_owner") {
      std::cerr << "[ERROR]: --insert-execution must be compute or storage_owner" << std::endl;
      exit_with_help_message(argv);
    }

    if (insert_workers > num_threads || query_workers > num_threads) {
      std::cerr << "[ERROR]: --insert-workers and --query-workers cannot exceed --threads" << std::endl;
      exit_with_help_message(argv);
    }

    if (insert_workers > 0 && query_workers > 0 && insert_workers + query_workers != num_threads) {
      std::cerr << "[ERROR]: --insert-workers + --query-workers must equal --threads when both are set" << std::endl;
      exit_with_help_message(argv);
    }

    if (insert_coroutines > num_coroutines || query_coroutines > num_coroutines) {
      std::cerr << "[ERROR]: --insert-coroutines and --query-coroutines cannot exceed --coroutines" << std::endl;
      exit_with_help_message(argv);
    }

    if (insert_execution == "storage_owner") {
      if (routing) {
        std::cerr << "[ERROR]: storage-side insert execution is not compatible with --routing in the current implementation"
                  << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_batch_max == 0) {
        std::cerr << "[ERROR]: --storage-owner-batch-max must be > 0" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_batch_max > std::numeric_limits<u32>::max() / R) {
        std::cerr << "[ERROR]: --storage-owner-batch-max is too large for R; response invalidation capacity "
                     "must fit in u32" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_peer_rdma_tokens == 0) {
        std::cerr << "[ERROR]: --storage-owner-peer-rdma-tokens must be > 0" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_rpc_depth == 0) {
        std::cerr << "[ERROR]: --storage-owner-rpc-depth must be > 0" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_rpc_timeout_ms == 0) {
        std::cerr << "[ERROR]: --storage-owner-rpc-timeout-ms must be > 0" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_search_snapshot_batch == 0) {
        std::cerr << "[ERROR]: --storage-owner-search-snapshot-batch must be > 0" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_update_mode != "exact" &&
          storage_owner_update_mode != "local_stitch") {
        std::cerr << "[ERROR]: --storage-owner-update-mode must be exact or local_stitch" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_update_mode == "local_stitch" &&
          (ip_distance || storage_owner_anchor_hints == 0 ||
           storage_owner_anchor_beam_width == 0 || storage_owner_anchor_expand_cap == 0)) {
        std::cerr << "[ERROR]: invalid local_stitch storage-owner configuration; L2 is required" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_reverse_mode != "async" && storage_owner_reverse_mode != "sync") {
        std::cerr << "[ERROR]: --storage-owner-reverse-mode must be async or sync" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_maintenance_mode != "off" && storage_owner_maintenance_mode != "finalize") {
        std::cerr << "[ERROR]: --storage-owner-maintenance-mode must be off or finalize" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_update_mode == "local_stitch" &&
          storage_owner_maintenance_mode != "finalize") {
        std::cerr << "[ERROR]: --storage-owner-update-mode=local_stitch requires "
                     "--storage-owner-maintenance-mode=finalize" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_maintenance_mode == "finalize" &&
          storage_owner_maintenance_workers == 0) {
        std::cerr << "[ERROR]: --storage-owner-maintenance-workers must be > 0 when finalize mode is enabled" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_reverse_queue_depth == 0) {
        std::cerr << "[ERROR]: --storage-owner-reverse-queue-depth must be > 0" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_owner_reverse_coalesce_max == 0) {
        std::cerr << "[ERROR]: --storage-owner-reverse-coalesce-max must be > 0" << std::endl;
        exit_with_help_message(argv);
      }
      if (storage_peers.size() != num_server_nodes()) {
        std::cerr << "[ERROR]: --storage-peers must list exactly one endpoint per storage node when "
                     "--insert-execution=storage_owner"
                  << std::endl;
        exit_with_help_message(argv);
      }
    }
  }

public:
  VectorDType resolved_vector_dtype() const {
    if (vector_data_type == "auto") {
      return VectorDType::float32;
    }
    return parse_vector_dtype(vector_data_type);
  }

private:
  static str normalize_mode(str value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    return value;
  }

public:
  filepath_t resolved_index_prefix() const {
    return index_path::resolve_prefix(data_path, index_prefix, R, beam_width_construction);
  }

  bool use_storage_owner_insert() const { return insert_execution == "storage_owner"; }
  bool use_gpu_persistent_search() const { return search_engine == "gpu_persistent"; }
  u32 effective_rdma_qp_pool_size() const {
    if (rdma_qp_pool_size != 0) return rdma_qp_pool_size;
    return rdma_read_batch_mode == "legacy" ? 1 : MAX_QPS;
  }

  friend std::ostream& operator<<(std::ostream& os, const IndexConfiguration& config) {
    os << static_cast<const Configuration&>(config);

    if (config.is_initiator) {
      constexpr i32 width = 30;
      constexpr i32 max_width = width * 2;

      os << std::left << std::setfill(' ');
      os << std::setw(width) << "data path: " << config.data_path << std::endl;
      if (!config.index_prefix.empty()) {
        os << std::setw(width) << "index prefix: " << config.index_prefix << std::endl;
      }
      os << std::setw(width) << "query suffix: " << config.query_suffix << std::endl;
      os << std::setw(width) << "number of threads: " << config.num_threads << std::endl;
      os << std::setw(width) << "number of coroutines: " << config.num_coroutines << std::endl;
      os << std::setw(width) << "threads pinned: " << (config.disable_thread_pinning ? "false" : "true") << std::endl;
      os << std::setw(width) << "seed: " << config.seed << std::endl;
      os << std::setw(width) << "dimension: " << config.dim << std::endl;
      os << std::setw(width) << "max vectors: " << config.max_vectors << std::endl;
      os << std::setw(width) << "CN memory (GB): " << config.cn_memory_gb << std::endl;
      os << std::setfill('-') << std::setw(max_width) << "" << std::endl;
      os << std::left << std::setfill(' ');
      os << std::setw(width) << "K: " << config.k << std::endl;
      os << std::setw(width) << "R (max degree): " << config.R << std::endl;
      os << std::setw(width) << "beam width (search): " << config.beam_width << std::endl;
      os << std::setw(width) << "beam width (construction): " << config.beam_width_construction << std::endl;
      os << std::setw(width) << "alpha: " << config.alpha << std::endl;
      os << std::setw(width) << "vector data type: " << config.vector_data_type << std::endl;
      os << std::setw(width) << "insert execution: " << config.insert_execution << std::endl;
      if (!config.storage_peers.empty()) {
        os << std::setw(width) << "storage id: " << config.storage_id << std::endl;
        os << std::setw(width) << "storage batch max: " << config.storage_owner_batch_max << std::endl;
        os << std::setw(width) << "storage batch wait(us): " << config.storage_owner_batch_wait_us << std::endl;
        os << std::setw(width) << "storage peer RDMA tokens: " << config.storage_owner_peer_rdma_tokens << std::endl;
        os << std::setw(width) << "storage RPC depth: " << config.storage_owner_rpc_depth << std::endl;
        os << std::setw(width) << "storage RPC timeout(ms): " << config.storage_owner_rpc_timeout_ms << std::endl;
        os << std::setw(width) << "storage construction beam: "
           << config.storage_owner_construction_beam_width << std::endl;
        os << std::setw(width) << "storage snapshot batch: "
           << config.storage_owner_search_snapshot_batch << std::endl;
        os << std::setw(width) << "storage prune max candidates: "
           << config.storage_owner_prune_max_candidates << std::endl;
        os << std::setw(width) << "storage update mode: " << config.storage_owner_update_mode << std::endl;
        if (config.storage_owner_update_mode == "local_stitch") {
          os << std::setw(width) << "anchor hints: " << config.storage_owner_anchor_hints << std::endl;
          os << std::setw(width) << "anchor beam width: " << config.storage_owner_anchor_beam_width << std::endl;
          os << std::setw(width) << "anchor expand cap: " << config.storage_owner_anchor_expand_cap << std::endl;
          os << std::setw(width) << "anchor remote cap: "
             << config.storage_owner_anchor_remote_rescue_cap << std::endl;
          os << std::setw(width) << "local stitch sync fast path: "
             << std::boolalpha << config.storage_owner_local_stitch_sync_fast_path
             << std::noboolalpha << std::endl;
        }
        os << std::setw(width) << "storage maintenance mode: "
           << config.storage_owner_maintenance_mode << std::endl;
        os << std::setw(width) << "storage maintenance workers: "
           << config.storage_owner_maintenance_workers << std::endl;
        os << std::setw(width) << "storage reverse mode: " << config.storage_owner_reverse_mode << std::endl;
        os << std::setw(width) << "storage reverse queue depth: "
           << config.storage_owner_reverse_queue_depth << std::endl;
        os << std::setw(width) << "storage reverse flush(us): "
           << config.storage_owner_reverse_flush_us << std::endl;
        os << std::setw(width) << "storage reverse coalesce max: "
           << config.storage_owner_reverse_coalesce_max << std::endl;
        os << std::setw(width) << "storage peers: " << "[";
        for (const str& node : config.storage_peers) {
          os << node << ", ";
        }
        os << "\b\b]" << std::endl;
      }
      os << std::setw(width) << "insert workers: " << config.insert_workers << std::endl;
      os << std::setw(width) << "query workers: " << config.query_workers << std::endl;
      os << std::setw(width) << "insert coroutines: " << config.insert_coroutines << std::endl;
      os << std::setw(width) << "query coroutines: " << config.query_coroutines << std::endl;
      os << std::setw(width) << "GPU device: " << config.gpu_device << std::endl;
      os << std::setw(width) << "Search engine: " << config.search_engine << std::endl;
      if (config.use_gpu_persistent_search()) {
        os << std::setw(width) << "GPU RDMA backend: " << config.gpu_rdma_backend << std::endl;
        os << std::setw(width) << "GPU query batch min/target/max: "
           << config.query_batch_min << "/" << config.query_batch_target << "/"
           << config.query_batch_max << std::endl;
        os << std::setw(width) << "GPU query batch wait(us): "
           << config.query_batch_wait_us << std::endl;
        os << std::setw(width) << "GPU hot degree: " << config.gpu_hot_degree << std::endl;
        os << std::setw(width) << "GPU cold expansion budget: "
           << config.gpu_cold_expansions << std::endl;
        os << std::setw(width) << "Update visibility(us): "
           << config.update_visibility_us << std::endl;
      }
      os << std::setw(width) << "GPUDirect RDMA: " << (config.gpudirect_rdma ? "true" : "false") << std::endl;
      os << std::setw(width) << "Fine-grained breakdown: "
         << (config.enable_breakdown ? "true" : "false") << std::endl;
      os << std::setw(width) << "Observe device utilization: "
         << (config.observe_device_utilization ? "true" : "false") << std::endl;
      os << std::setw(width) << "Expansion Batch (K): " << config.expansion_batch << std::endl;
      os << std::setw(width) << "Credit-aware expansion: "
         << (config.credit_aware_expansion ? "true" : "false") << std::endl;
      os << std::setw(width) << "Credit-aware min K: " << config.credit_aware_min_k << std::endl;
      os << std::setw(width) << "Credit-aware max K: "
         << (config.credit_aware_max_k == 0 ? config.expansion_batch : config.credit_aware_max_k)
         << std::endl;
      os << std::setw(width) << "Credit-aware target candidates: "
         << config.credit_aware_target_candidates << std::endl;
      os << std::setw(width) << "Credit-aware max lookahead: "
         << config.credit_aware_max_lookahead << std::endl;
      os << std::setw(width) << "Credit-aware cost guard: "
         << (config.credit_aware_cost_guard ? "true" : "false") << std::endl;
      os << std::setw(width) << "Credit-aware cost max extra ratio: "
         << config.credit_aware_cost_max_extra_ratio << std::endl;
      os << std::setw(width) << "Credit-aware cost probe rounds: "
         << config.credit_aware_cost_probe_rounds << std::endl;
      os << std::setw(width) << "RDMA QP Pool Size: "
         << config.effective_rdma_qp_pool_size();
      if (config.rdma_qp_pool_size == 0) os << " (auto)";
      os << std::endl;
      os << std::setw(width) << "RDMA read batch mode: " << config.rdma_read_batch_mode << std::endl;
      os << std::setw(width) << "RDMA read chain size: " << config.rdma_read_chain_size << std::endl;
      os << std::setw(width) << "RDMA read max inflight WRs: "
         << config.rdma_read_max_inflight_wrs << std::endl;
      os << std::setw(width) << "Query Batch Size: " << config.query_batch_size << std::endl;
      os << std::setw(width) << "Use RaBitQ: " << (config.use_rabitq ? "true" : "false") << std::endl;
      os << std::setw(width) << "RaBitQ mode: cpu_gate" << std::endl;
      os << std::setw(width) << "RaBitQ gate width: " << config.rabitq_gate_width << std::endl;
      os << std::setw(width) << "RaBitQ gate max width: " << config.rabitq_gate_max_width << std::endl;
      os << std::setw(width) << "RaBitQ gate margin: " << config.rabitq_gate_margin << std::endl;
      os << std::setw(width) << "RaBitQ dynamic budget MB: " << config.rabitq_dynamic_budget_mb << std::endl;
      os << std::setw(width) << "RaBitQ coalesce target: " << config.rabitq_coalesce_target << std::endl;
      os << std::setw(width) << "RaBitQ coalesce min: " << config.rabitq_coalesce_min << std::endl;
      os << std::setw(width) << "RaBitQ coalesce wait(us): " << config.rabitq_coalesce_wait_us << std::endl;
      os << std::setw(width) << "RaBitQ warmup exact expansions: "
         << config.rabitq_warmup_exact_expansions << std::endl;
      os << std::setw(width) << "RaBitQ audit period: " << config.rabitq_audit_period << std::endl;
      os << std::setw(width) << "RaBitQ strict recall: "
         << (config.rabitq_strict_recall ? "true" : "false") << std::endl;
      os << std::setfill('=') << std::setw(max_width) << "" << std::endl;
    } else if (config.is_server && !config.server_index_file.empty()) {
      os << std::left << std::setfill(' ');
      os << std::setw(30) << "server index file: " << config.server_index_file << std::endl;
      os << std::setfill('=') << std::setw(60) << "" << std::endl;
    }
    return os;
  }
};

}  // namespace configuration
