#pragma once

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <library/configuration.hh>

#include "index_path.hh"
#include "types.hh"

namespace configuration {

// struct used for sending serialized from CN to MN
struct Parameters {
  u32 num_threads{};
  bool use_cache{};
  bool routing{};
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
  u32 rabitq_bits{};      // bits per dimension for RaBitQ quantization
  u32 k{};
  u32 gpu_device{};       // CUDA device ID
  bool gpudirect_rdma{};  // Enable GPUDirect RDMA (read vectors directly into GPU buffers)
  u32 neighbor_cache_mb{0};
  u32 neighbor_cache_invalidation_ms{0};
  u32 neighbor_cache_invalidation_inserts{1};
  u32 gpu_rabitq_cache_mb{0};
  str rabitq_cache_mode{"slot_clock"};
  u32 gentile_tile_slots{32};
  double gentile_nursery_ratio{0.25};
  u32 gentile_promotion_threshold{2};
  bool gentile_enable_promotion{false};
  bool gentile_enable_value_bin{false};
  bool gentile_enable_hit_tile_grouping{true};
  str search_mode{"exact_gpu"};
  str insert_execution{"compute"};
  u32 insert_workers{};
  u32 query_workers{};
  u32 insert_coroutines{};
  u32 query_coroutines{};
  u32 storage_id{0};
  vec<str> storage_peers;
  u32 storage_owner_batch_max{16};
  u32 storage_owner_batch_wait_us{250};
  u32 storage_owner_cache_mb{0};
  u32 storage_owner_peer_rdma_tokens{8};
  u32 storage_owner_rpc_depth{8};
  u32 storage_owner_rpc_timeout_ms{30000};
  u32 storage_owner_construction_beam_width{128};
  u32 storage_owner_search_snapshot_batch{64};
  u32 storage_owner_prune_max_candidates{128};
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

  u32 cache_size_ratio{};  // in %
  bool use_cache{};
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
    search_mode = normalize_search_mode(search_mode);
    insert_execution = normalize_search_mode(insert_execution);
    storage_owner_reverse_mode = normalize_search_mode(storage_owner_reverse_mode);
    rabitq_cache_mode = normalize_search_mode(rabitq_cache_mode);

    if (!is_server) {
      validate_compute_node_options(argv);
    }

    operator<<(std::cerr, *this);
  }

private:
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
      "cache", po::bool_switch(&use_cache), "Activate cache on CNs.")(
      "routing", po::bool_switch(&routing), "Activate adaptive query routing.")(
      "cache-ratio",
      po::value<u32>(&cache_size_ratio)->default_value(5),
      "Cache size ratio relative to the index size in %.")(
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
      "alpha", po::value<f64>(&alpha)->default_value(1.2), "RobustPrune diversity factor.")(
      "rabitq-bits", po::value<u32>(&rabitq_bits)->default_value(1),
      "Bits per dimension for RaBitQ quantization (1, 2, 4, or 8).")(
      "search-mode", po::value<str>(&search_mode)->default_value(search_mode),
      "Search mode for the query path: exact_gpu or rabitq_gpu.")(
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
      "storage-owner-cache-mb", po::value<u32>(&storage_owner_cache_mb)->default_value(storage_owner_cache_mb),
      "Per-memory-node storage_owner local metadata/vector/neighbor cache size in MB. 0 disables it.")(
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
      "neighbor-cache-mb", po::value<u32>(&neighbor_cache_mb)->default_value(0),
      "CPU neighbor-list cache size per compute node in MB. 0 disables it.")(
      "neighbor-cache-invalidation-ms",
      po::value<u32>(&neighbor_cache_invalidation_ms)->default_value(neighbor_cache_invalidation_ms),
      "Minimum time between neighbor-cache epoch invalidations after inserts. 0 disables time batching.")(
      "neighbor-cache-invalidation-inserts",
      po::value<u32>(&neighbor_cache_invalidation_inserts)->default_value(neighbor_cache_invalidation_inserts),
      "Minimum successful inserts between neighbor-cache epoch invalidations.")(
      "gpu-rabitq-cache-mb", po::value<u32>(&gpu_rabitq_cache_mb)->default_value(0),
      "GPU RaBitQ cache size per compute node in MB. 0 disables it.")(
      "rabitq-cache-mode", po::value<str>(&rabitq_cache_mode)->default_value(rabitq_cache_mode),
      "GPU RaBitQ cache mode: off, slot_clock, or gentile.")(
      "gentile-tile-slots", po::value<u32>(&gentile_tile_slots)->default_value(gentile_tile_slots),
      "GenTile cache slots per tile.")(
      "gentile-nursery-ratio", po::value<double>(&gentile_nursery_ratio)->default_value(gentile_nursery_ratio),
      "Fraction of GenTile cache tiles reserved for nursery allocation.")(
      "gentile-promotion-threshold",
      po::value<u32>(&gentile_promotion_threshold)->default_value(gentile_promotion_threshold),
      "GenTile nursery hit credit threshold before promotion. Reserved for later stages.")(
      "gentile-enable-promotion", po::value<bool>(&gentile_enable_promotion)->default_value(gentile_enable_promotion),
      "Enable GenTile nursery-to-hot promotion. Reserved for later stages.")(
      "gentile-enable-value-bin", po::value<bool>(&gentile_enable_value_bin)->default_value(gentile_enable_value_bin),
      "Enable GenTile tile-level value-bin replacement. Reserved for later stages.")(
      "gentile-enable-hit-tile-grouping",
      po::value<bool>(&gentile_enable_hit_tile_grouping)->default_value(gentile_enable_hit_tile_grouping),
      "Enable GenTile grouped-by-tile cached distance kernel.")(
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

    if (store_index && load_index) {
      std::cerr << "[ERROR]: --store-index and --load-index cannot be used in conjunction" << std::endl;
      exit_with_help_message(argv);
    }

    if ((store_index || load_index) && index_prefix.empty() && data_path.empty()) {
      std::cerr << "[ERROR]: --data-path or --index-prefix is required when --load-index or --store-index is set"
                << std::endl;
      exit_with_help_message(argv);
    }

    if (use_cache && cache_size_ratio == 0) {
      std::cerr << "[ERROR]: If --cache is set, --cache-ratio must be > 0" << std::endl;
      exit_with_help_message(argv);
    }

    if (search_mode != "exact_gpu" && search_mode != "rabitq_gpu") {
      std::cerr << "[ERROR]: --search-mode must be exact_gpu or rabitq_gpu" << std::endl;
      exit_with_help_message(argv);
    }

    if (neighbor_cache_invalidation_inserts == 0) {
      std::cerr << "[ERROR]: --neighbor-cache-invalidation-inserts must be > 0" << std::endl;
      exit_with_help_message(argv);
    }

    if (rabitq_cache_mode != "off" && rabitq_cache_mode != "slot_clock" && rabitq_cache_mode != "gentile") {
      std::cerr << "[ERROR]: --rabitq-cache-mode must be off, slot_clock, or gentile" << std::endl;
      exit_with_help_message(argv);
    }
    if (gentile_tile_slots == 0) {
      std::cerr << "[ERROR]: --gentile-tile-slots must be > 0" << std::endl;
      exit_with_help_message(argv);
    }
    if (gentile_nursery_ratio <= 0.0 || gentile_nursery_ratio > 1.0) {
      std::cerr << "[ERROR]: --gentile-nursery-ratio must be in (0, 1]" << std::endl;
      exit_with_help_message(argv);
    }

    if (insert_execution != "compute" && insert_execution != "storage_owner") {
      std::cerr << "[ERROR]: --insert-execution must be compute or storage_owner" << std::endl;
      exit_with_help_message(argv);
    }

    if (rabitq_bits != 1 && rabitq_bits != 2 && rabitq_bits != 4 && rabitq_bits != 8) {
      std::cerr << "[ERROR]: --rabitq-bits must be 1, 2, 4, or 8" << std::endl;
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
      if (storage_owner_reverse_mode != "async" && storage_owner_reverse_mode != "sync") {
        std::cerr << "[ERROR]: --storage-owner-reverse-mode must be async or sync" << std::endl;
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

  static str normalize_search_mode(str value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    return value;
  }

public:
  filepath_t resolved_index_prefix() const {
    return index_path::resolve_prefix(data_path, index_prefix, R, beam_width_construction);
  }

  bool use_rabitq_search() const { return search_mode == "rabitq_gpu"; }
  bool use_storage_owner_insert() const { return insert_execution == "storage_owner"; }

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
      os << std::setw(width) << "insert execution: " << config.insert_execution << std::endl;
      if (!config.storage_peers.empty()) {
        os << std::setw(width) << "storage id: " << config.storage_id << std::endl;
        os << std::setw(width) << "storage batch max: " << config.storage_owner_batch_max << std::endl;
        os << std::setw(width) << "storage batch wait(us): " << config.storage_owner_batch_wait_us << std::endl;
        os << std::setw(width) << "storage cache (MB): " << config.storage_owner_cache_mb << std::endl;
        os << std::setw(width) << "storage peer RDMA tokens: " << config.storage_owner_peer_rdma_tokens << std::endl;
        os << std::setw(width) << "storage RPC depth: " << config.storage_owner_rpc_depth << std::endl;
        os << std::setw(width) << "storage RPC timeout(ms): " << config.storage_owner_rpc_timeout_ms << std::endl;
        os << std::setw(width) << "storage construction beam: "
           << config.storage_owner_construction_beam_width << std::endl;
        os << std::setw(width) << "storage snapshot batch: "
           << config.storage_owner_search_snapshot_batch << std::endl;
        os << std::setw(width) << "storage prune max candidates: "
           << config.storage_owner_prune_max_candidates << std::endl;
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
      os << std::setw(width) << "RaBitQ bits: " << config.rabitq_bits << std::endl;
      os << std::setw(width) << "search mode: " << config.search_mode << std::endl;
      os << std::setw(width) << "insert workers: " << config.insert_workers << std::endl;
      os << std::setw(width) << "query workers: " << config.query_workers << std::endl;
      os << std::setw(width) << "insert coroutines: " << config.insert_coroutines << std::endl;
      os << std::setw(width) << "query coroutines: " << config.query_coroutines << std::endl;
      os << std::setw(width) << "GPU device: " << config.gpu_device << std::endl;
      os << std::setw(width) << "GPUDirect RDMA: " << (config.gpudirect_rdma ? "true" : "false") << std::endl;
      os << std::setw(width) << "neighbor cache (MB): " << config.neighbor_cache_mb << std::endl;
      os << std::setw(width) << "neighbor cache invalidation(ms): "
         << config.neighbor_cache_invalidation_ms << std::endl;
      os << std::setw(width) << "neighbor cache invalidation inserts: "
         << config.neighbor_cache_invalidation_inserts << std::endl;
      os << std::setw(width) << "GPU RaBitQ cache (MB): " << config.gpu_rabitq_cache_mb << std::endl;
      os << std::setw(width) << "RaBitQ cache mode: " << config.rabitq_cache_mode << std::endl;
      os << std::setw(width) << "GenTile tile slots: " << config.gentile_tile_slots << std::endl;
      os << std::setw(width) << "GenTile nursery ratio: " << config.gentile_nursery_ratio << std::endl;
      os << std::setw(width) << "GenTile promotion threshold: " << config.gentile_promotion_threshold << std::endl;
      os << std::setw(width) << "GenTile promotion: " << (config.gentile_enable_promotion ? "true" : "false") << std::endl;
      os << std::setw(width) << "GenTile value-bin: " << (config.gentile_enable_value_bin ? "true" : "false") << std::endl;
      os << std::setw(width) << "GenTile hit grouping: "
         << (config.gentile_enable_hit_tile_grouping ? "true" : "false") << std::endl;
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
