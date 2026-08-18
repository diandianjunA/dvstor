#pragma once

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>

#include <library/configuration.hh>

#include "constants.hh"
#include "gpu_search/adaptive_frontier.hh"
#include "types.hh"
#include "vector_dtype.hh"

namespace configuration {

struct Parameters {
  u32 num_threads{};
  u32 gpu_rdma_qps{};
};

class IndexConfiguration : public Configuration {
public:
  filepath_t index_prefix{};
  filepath_t server_index_file{};
  u32 num_threads{};
  i32 seed{1234};
  bool disable_thread_pinning{};

  u32 dim{};
  u32 max_vectors{1'000'000};
  // Exclusive upper bound for logical node IDs. Zero during option parsing
  // means "derive from max_vectors" for backward-compatible configurations.
  u32 vector_id_namespace_size{};
  u32 k{};
  u32 R{64};
  u32 beam_width_construction{200};
  f64 alpha{1.2};
  str vector_data_type{"auto"};

  u32 gpu_device{};
  bool enable_breakdown{true};
  u32 gpu_query_slots{256};
  u32 gpu_memory_limit_gb{40};
  u32 gpu_memory_reserve_gb{4};
  u32 gpu_bootstrap_window_mb{64};
  u32 gpu_bootstrap_windows{2};
  u32 gpu_graph_prefetch_depth{32};
  // Zero preserves legacy configurations by deriving the authoritative
  // commit width from gpu_graph_prefetch_depth after option parsing.
  u32 gpu_graph_commit_width{};
  // Zero selects the workspace/beam-limited speculative issue cap.
  u32 gpu_graph_issue_width{};
  str gpu_query_graph_read_policy{"fixed"};
  // Keep dynamic-node extent prediction independently ablatable from the
  // base-node Live-Extent sidecar. Fixed graph reads ignore this switch.
  bool gpu_dynamic_graph_extent{true};
  str gpu_query_beam_merge_policy{"legacy"};
  str query_rdma_trace_mode{"off"};
  u32 query_rdma_trace_sample_rate{1000};
  filepath_t query_rdma_trace_output{};
  u32 query_rdma_trace_events_per_query{1024};
  u32 gpu_traversal_beam_width{128};
  u32 gpu_final_rerank_width{64};
  u32 gpu_max_expansions{384};
  u32 gpu_rdma_qps{4};
  // Deprecated compatibility input. Dynamic PQ now uses a fixed one-to-one
  // physical-slot arena sized from storage metadata; this value is ignored.
  u32 gpu_dynamic_code_cache_entries{1u << 20};
  // A CQE can be delayed well beyond ordinary read latency when GPU reads and
  // CPU-posted mutation traffic share the NIC.  This is a liveness bound, not
  // a latency target: transport errors still fail immediately.
  u32 gpu_direct_timeout_ms{250};
  u32 gpu_persistent_blocks_per_sm{4};
  // Compute-side mutation support is optional for query-only deployments.
  // Keep it enabled by default so existing service configurations preserve
  // their insert/upsert/erase behavior.
  bool enable_updates{true};

  u32 storage_id{};
  vec<str> storage_peers;
  u32 storage_owner_batch_max{16};
  // Give synchronous writers enough time to form one useful foreground
  // microbatch. A full batch is sent immediately; this only bounds additional
  // coalescing while an RPC slot is available, not remote service time.
  u32 storage_owner_batch_max_wait_us{2'000};
  // Stage2 is query-invisible background maintenance. Its compaction horizon
  // must not be tied to the foreground Stage1 batching latency.
  // Stage2 batching is coordinated across already-pending work.  A short
  // horizon still forms useful waves without adding millisecond-scale queue
  // delay when update load is sparse or uneven across homes.
  u32 storage_owner_stage2_batch_max_wait_us{50};
  // Fresh inserts may publish the sorted local Beam as their provisional
  // adjacency and defer the exact local RobustPrune to Stage2. Stage2 then
  // reconstructs the same prune seed before the final global prune, so this
  // changes foreground placement of work, not durable graph quality.
  bool storage_owner_defer_stage1_prune{false};
  bool storage_owner_stage2_score_many{false};
  // Maximum ordered graph issue width. Width one is the exact legacy path;
  // larger values are promotion-gated and only fill spare items in an RPC
  // that an authoritative expansion already requires.
  u32 storage_owner_stage2_graph_issue_width{16};
  u32 storage_owner_peer_qps_per_peer{8};
  u32 storage_owner_peer_rdma_tokens{16};
  u32 storage_owner_rpc_depth{8};
  u32 storage_owner_rpc_timeout_ms{30'000};
  u32 storage_owner_search_snapshot_batch{256};
  // Stage2 is part of the only supported update protocol, so at least one
  // maintenance executor must exist even when the caller does not tune it.
  u32 storage_owner_maintenance_workers{1};
  u32 storage_owner_maintenance_queue_depth{65'536};
  u32 storage_owner_reverse_queue_depth{65'536};
  u32 storage_owner_reverse_coalesce_max{256};

  u32 mn_memory_gb{10};

  IndexConfiguration(int argc, char** argv) {
    add_options();
    process_program_options(argc, argv);
    if (vector_id_namespace_size == 0) {
      vector_id_namespace_size = max_vectors;
    }
    resolve_graph_frontier_widths();
    vector_data_type = normalize_mode(vector_data_type);
    gpu_query_graph_read_policy =
      normalize_mode(gpu_query_graph_read_policy);
    gpu_query_beam_merge_policy =
      normalize_mode(gpu_query_beam_merge_policy);
    query_rdma_trace_mode = normalize_mode(query_rdma_trace_mode);
    validate(argv);
    operator<<(std::cerr, *this);
  }

  filepath_t resolved_index_prefix() const {
    return index_prefix;
  }

  VectorDType resolved_vector_dtype() const {
    return vector_data_type == "auto"
      ? VectorDType::float32 : parse_vector_dtype(vector_data_type);
  }

  u32 resolved_storage_owner_construction_width() const {
    return beam_width_construction;
  }

private:
  void resolve_graph_frontier_widths() {
    if (gpu_graph_commit_width == 0) {
      gpu_graph_commit_width = gpu_graph_prefetch_depth;
    }
    if (gpu_graph_issue_width == 0) {
      gpu_graph_issue_width =
        gpu_search::adaptive_frontier::automatic_max_issue_width(
          gpu_traversal_beam_width);
      // A historical configuration could legally use a prefetch depth larger
      // than its traversal beam. Keep that configuration valid and preserve
      // its authoritative width instead of silently narrowing the search.
      if (gpu_graph_issue_width < gpu_graph_commit_width) {
        gpu_graph_issue_width = gpu_graph_commit_width;
      }
    }
  }

  static str normalize_mode(str value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    return value;
  }

  void add_options() {
    desc.add_options()
      ("index-prefix", po::value<filepath_t>(&index_prefix),
       "Prefix shared by metadata, graph shards, PQ model, and PQ code shards.")
      ("server-index-file", po::value<filepath_t>(&server_index_file),
       "Local graph shard loaded by this storage node before serving requests.")
      ("threads,t", po::value<u32>(&num_threads),
       "CPU control/update threads represented by this process.")
      ("disable-thread-pinning,p",
       po::bool_switch(&disable_thread_pinning)->default_value(false),
       "Disable CPU thread pinning.")
      ("seed", po::value<i32>(&seed)->default_value(seed),
       "Deterministic random seed.")
      ("dim", po::value<u32>(&dim), "Vector dimension.")
      ("max-vectors", po::value<u32>(&max_vectors)->default_value(max_vectors),
       "Number of immutable base vectors in the loaded index.")
      ("vector-id-namespace-size",
       po::value<u32>(&vector_id_namespace_size)
         ->default_value(vector_id_namespace_size),
       "Exclusive upper bound for base and dynamically assigned vector IDs.")
      ("k", po::value<u32>(&k), "Requested nearest-neighbor count.")
      ("R", po::value<u32>(&R)->default_value(R), "Maximum graph out-degree.")
      ("beam-width-construction",
       po::value<u32>(&beam_width_construction)->default_value(beam_width_construction),
       "Beam width used by storage-side online graph maintenance.")
      ("alpha", po::value<f64>(&alpha)->default_value(alpha),
       "RobustPrune diversity factor.")
      ("vector-data-type",
       po::value<str>(&vector_data_type)->default_value(vector_data_type),
       "Exact-vector storage type: auto, float32, uint8, or int8.")

      ("gpu-device", po::value<u32>(&gpu_device)->default_value(gpu_device),
       "CUDA device used by the persistent query engine.")
      ("enable-breakdown",
       po::value<bool>(&enable_breakdown)->default_value(enable_breakdown),
       "Collect per-request breakdown samples.")
      ("gpu-query-slots",
       po::value<u32>(&gpu_query_slots)->default_value(gpu_query_slots),
       "Maximum concurrent GPU query slots.")
      ("gpu-memory-limit-gb",
       po::value<u32>(&gpu_memory_limit_gb)->default_value(gpu_memory_limit_gb),
       "Hard limit for explicit query-engine GPU allocations.")
      ("gpu-memory-reserve-gb",
       po::value<u32>(&gpu_memory_reserve_gb)->default_value(gpu_memory_reserve_gb),
       "GPU memory reserved for CUDA and transport runtime state.")
      ("gpu-bootstrap-window-mb",
       po::value<u32>(&gpu_bootstrap_window_mb)->default_value(gpu_bootstrap_window_mb),
       "Maximum one-time PQ bootstrap RDMA read size.")
      ("gpu-bootstrap-windows",
       po::value<u32>(&gpu_bootstrap_windows)->default_value(gpu_bootstrap_windows),
       "Concurrent one-time PQ bootstrap reads.")
      ("gpu-graph-prefetch-depth",
       po::value<u32>(&gpu_graph_prefetch_depth)->default_value(gpu_graph_prefetch_depth),
       "Legacy graph fetch/expansion width and default commit width.")
      ("gpu-graph-commit-width",
       po::value<u32>(&gpu_graph_commit_width)->default_value(gpu_graph_commit_width),
       "Authoritative graph expansion width; zero derives from legacy prefetch depth.")
      ("gpu-graph-issue-width",
       po::value<u32>(&gpu_graph_issue_width)->default_value(gpu_graph_issue_width),
       "Maximum speculative graph read width; zero derives from frontier capacity and traversal beam.")
      ("gpu-query-graph-read-policy",
       po::value<str>(&gpu_query_graph_read_policy)
         ->default_value(gpu_query_graph_read_policy),
       "GPU graph-record transfer policy: fixed or live-extent.")
      ("gpu-dynamic-graph-extent",
       po::value<bool>(&gpu_dynamic_graph_extent)
         ->default_value(gpu_dynamic_graph_extent),
       "Use incarnation-tagged dynamic-node extent hints when Live-Extent "
       "graph reads are enabled.")
      ("gpu-query-beam-merge-policy",
       po::value<str>(&gpu_query_beam_merge_policy)
         ->default_value(gpu_query_beam_merge_policy),
       "GPU query Beam merge policy: legacy or stable-run.")
      ("query-rdma-trace-mode",
       po::value<str>(&query_rdma_trace_mode)->default_value(query_rdma_trace_mode),
       "Shard-batch RDMA trace mode: off, sampled, or full.")
      ("query-rdma-trace-sample-rate",
       po::value<u32>(&query_rdma_trace_sample_rate)
         ->default_value(query_rdma_trace_sample_rate),
       "In sampled mode, trace one of every N request IDs.")
      ("query-rdma-trace-output",
       po::value<filepath_t>(&query_rdma_trace_output)
         ->default_value(query_rdma_trace_output),
       "JSONL output for detailed shard-batch RDMA trace events.")
      ("query-rdma-trace-events-per-query",
       po::value<u32>(&query_rdma_trace_events_per_query)
         ->default_value(query_rdma_trace_events_per_query),
       "Preallocated per-query-slot trace event capacity.")
      ("gpu-traversal-beam-width",
       po::value<u32>(&gpu_traversal_beam_width)->default_value(gpu_traversal_beam_width),
       "OPQ/PQ beam width for GPU graph navigation.")
      ("gpu-final-rerank-width",
       po::value<u32>(&gpu_final_rerank_width)->default_value(gpu_final_rerank_width),
       "Exact vectors fetched for final reranking.")
      ("gpu-max-expansions",
       po::value<u32>(&gpu_max_expansions)->default_value(gpu_max_expansions),
       "Maximum graph expansions per query.")
      ("gpu-rdma-qps",
       po::value<u32>(&gpu_rdma_qps)->default_value(gpu_rdma_qps),
       "GPU-initiated GPUNetIO QPs per storage node.")
      ("gpu-direct-timeout-ms",
       po::value<u32>(&gpu_direct_timeout_ms)->default_value(gpu_direct_timeout_ms),
       "Maximum wait for one GPU-initiated RDMA completion before fail-stop.")
      ("gpu-persistent-blocks-per-sm",
       po::value<u32>(&gpu_persistent_blocks_per_sm)->default_value(gpu_persistent_blocks_per_sm),
       "Maximum unified persistent CTAs per GPU SM; hardware occupancy may be lower.")
      ("enable-updates",
       po::value<bool>(&enable_updates)->default_value(enable_updates),
       "Enable compute-side insert, upsert, and erase submission.")
      ("gpu-dynamic-code-cache-entries",
       po::value<u32>(&gpu_dynamic_code_cache_entries)
         ->default_value(gpu_dynamic_code_cache_entries),
       "Deprecated compatibility option; the dynamic-PQ arena is metadata-sized.")

      ("storage-id", po::value<u32>(&storage_id)->default_value(storage_id),
       "Zero-based storage shard identifier.")
      ("storage-peers", po::value<vec<str>>(&storage_peers)->multitoken(),
       "Ordered storage-node endpoints.")
      ("storage-owner-batch-max",
       po::value<u32>(&storage_owner_batch_max)->default_value(storage_owner_batch_max),
       "Maximum mutations in one storage RPC batch.")
      ("storage-owner-batch-max-wait-us",
       po::value<u32>(&storage_owner_batch_max_wait_us)
         ->default_value(storage_owner_batch_max_wait_us),
       "Maximum foreground wait while an announced initial mutation batch is "
       "still being published; zero runs that partial batch immediately.")
      ("storage-owner-stage2-batch-max-wait-us",
       po::value<u32>(&storage_owner_stage2_batch_max_wait_us)
         ->default_value(storage_owner_stage2_batch_max_wait_us),
       "Maximum background Stage2 compaction wait for a partial batch; zero "
       "runs partial Stage2 batches immediately.")
      ("storage-owner-defer-stage1-prune",
       po::value<bool>(&storage_owner_defer_stage1_prune)
         ->default_value(storage_owner_defer_stage1_prune),
       "Publish a bounded nearest-first provisional adjacency for fresh "
       "inserts and execute the exact local RobustPrune in Stage2.")
      ("storage-owner-stage2-score-many",
       po::value<bool>(&storage_owner_stage2_score_many)
         ->default_value(storage_owner_stage2_score_many),
       "Score remote Stage2 candidates at their physical home using a "
       "query-deduplicated exact score-many RPC.")
      ("storage-owner-stage2-graph-issue-width",
       po::value<u32>(&storage_owner_stage2_graph_issue_width)
         ->default_value(storage_owner_stage2_graph_issue_width),
       "Maximum ordered speculative Stage2 graph issue width; one disables "
       "speculation and preserves the legacy request path.")
      ("storage-owner-peer-rdma-tokens",
       po::value<u32>(&storage_owner_peer_rdma_tokens)->default_value(storage_owner_peer_rdma_tokens),
       "Outstanding peer reads allowed per storage data QP.")
      ("storage-owner-peer-qps-per-peer",
       po::value<u32>(&storage_owner_peer_qps_per_peer)
         ->default_value(storage_owner_peer_qps_per_peer),
       "Storage-to-storage RC QPs per peer, including one ordered control QP.")
      ("storage-owner-rpc-depth",
       po::value<u32>(&storage_owner_rpc_depth)->default_value(storage_owner_rpc_depth),
       "In-flight mutation batches per storage node.")
      ("storage-owner-rpc-timeout-ms",
       po::value<u32>(&storage_owner_rpc_timeout_ms)->default_value(storage_owner_rpc_timeout_ms),
       "Mutation RPC timeout.")
      ("storage-owner-search-snapshot-batch",
       po::value<u32>(&storage_owner_search_snapshot_batch)
         ->default_value(storage_owner_search_snapshot_batch),
       "Concurrent node snapshots during update search.")
      ("storage-owner-maintenance-workers",
       po::value<u32>(&storage_owner_maintenance_workers)
         ->default_value(storage_owner_maintenance_workers),
       "Background exact-finalization workers.")
      ("storage-owner-maintenance-queue-depth",
       po::value<u32>(&storage_owner_maintenance_queue_depth)
         ->default_value(storage_owner_maintenance_queue_depth),
       "Bounded graph-maintenance backlog; writers backpressure at the limit.")
      ("storage-owner-reverse-queue-depth",
       po::value<u32>(&storage_owner_reverse_queue_depth)
         ->default_value(storage_owner_reverse_queue_depth),
       "Maximum queued reverse updates.")
      ("storage-owner-reverse-coalesce-max",
       po::value<u32>(&storage_owner_reverse_coalesce_max)
         ->default_value(storage_owner_reverse_coalesce_max),
       "Maximum reverse updates per coalesced batch.")
      ("mn-memory", po::value<u32>(&mn_memory_gb)->default_value(mn_memory_gb),
       "Storage-node registered-memory capacity in GiB.");
  }

  void validate(char** argv) const {
    const auto fail = [&](const str& message) {
      std::cerr << "[ERROR]: " << message << std::endl;
      exit_with_help_message(argv);
    };

    if (index_prefix.empty()) fail("--index-prefix is required");
    if (num_threads == 0 || dim == 0 || max_vectors == 0 || k == 0 ||
        R == 0 || beam_width_construction == 0 || mn_memory_gb == 0) {
      fail("threads, dim, max-vectors, k, R, beam-width-construction, and mn-memory must be > 0");
    }
    if (vector_id_namespace_size < max_vectors) {
      fail("--vector-id-namespace-size must be >= --max-vectors");
    }
    if (R > kMaxSupportedGraphDegree) {
      fail("--R must be <= " + std::to_string(kMaxSupportedGraphDegree));
    }
    if (k > gpu_final_rerank_width) {
      fail("--k must not exceed --gpu-final-rerank-width");
    }
    try {
      if (vector_data_type != "auto") (void)parse_vector_dtype(vector_data_type);
    } catch (const std::exception& error) {
      fail(str{"invalid --vector-data-type: "} + error.what());
    }

    if (query_rdma_trace_mode != "off" &&
        query_rdma_trace_mode != "sampled" &&
        query_rdma_trace_mode != "full") {
      fail("--query-rdma-trace-mode must be off, sampled, or full");
    }
    if (query_rdma_trace_sample_rate == 0 ||
        query_rdma_trace_events_per_query == 0 ||
        query_rdma_trace_events_per_query > 65'536) {
      fail("invalid query RDMA trace sampling rate or event capacity");
    }
    if (query_rdma_trace_mode != "off" && query_rdma_trace_output.empty()) {
      fail("--query-rdma-trace-output is required when tracing is enabled");
    }
    if (gpu_query_graph_read_policy != "fixed" &&
        gpu_query_graph_read_policy != "live-extent") {
      fail("--gpu-query-graph-read-policy must be fixed or live-extent");
    }
    if (gpu_query_beam_merge_policy != "legacy" &&
        gpu_query_beam_merge_policy != "stable-run") {
      fail("--gpu-query-beam-merge-policy must be legacy or stable-run");
    }
    if (gpu_query_slots == 0 || gpu_query_slots > 4096 ||
        gpu_memory_limit_gb == 0 ||
        gpu_memory_reserve_gb >= gpu_memory_limit_gb ||
        gpu_bootstrap_window_mb == 0 || gpu_bootstrap_windows == 0 ||
        gpu_bootstrap_windows > 16 ||
        gpu_graph_prefetch_depth == 0 ||
        gpu_graph_prefetch_depth > 32 ||
        gpu_graph_commit_width == 0 ||
        gpu_graph_commit_width > gpu_graph_issue_width ||
        gpu_graph_issue_width >
          gpu_search::adaptive_frontier::kFrontierCapacity ||
        gpu_traversal_beam_width < k || gpu_traversal_beam_width > 256 ||
        gpu_final_rerank_width < k || gpu_final_rerank_width > 256 ||
        gpu_max_expansions < gpu_traversal_beam_width ||
        gpu_max_expansions > 4096 ||
        gpu_rdma_qps == 0 || gpu_rdma_qps > 32 ||
        gpu_direct_timeout_ms < 20 || gpu_direct_timeout_ms > 5'000 ||
        gpu_persistent_blocks_per_sm == 0 ||
        gpu_persistent_blocks_per_sm > 16) {
      fail("invalid persistent GPU query configuration");
    }

    if (storage_peers.size() != num_server_nodes()) {
      fail("--storage-peers must list exactly one endpoint per storage node");
    }
    if (storage_id >= num_server_nodes() ||
        storage_owner_batch_max == 0 ||
        storage_owner_peer_qps_per_peer == 0 ||
        storage_owner_peer_qps_per_peer > kMaxPeerQps ||
        storage_owner_peer_rdma_tokens == 0 ||
        storage_owner_rpc_depth == 0 ||
        storage_owner_rpc_timeout_ms == 0 ||
        storage_owner_search_snapshot_batch == 0 ||
        storage_owner_stage2_graph_issue_width == 0 ||
        storage_owner_stage2_graph_issue_width > storage_owner_batch_max ||
        storage_owner_maintenance_workers == 0 ||
        storage_owner_maintenance_queue_depth == 0 ||
        storage_owner_reverse_queue_depth == 0 ||
        storage_owner_reverse_coalesce_max == 0) {
      fail("invalid storage-side update configuration");
    }
    if (storage_owner_batch_max > std::numeric_limits<u32>::max() / R) {
      fail("storage-owner batch invalidation capacity exceeds u32");
    }
    if (static_cast<u64>(storage_owner_maintenance_queue_depth) <
        static_cast<u64>(storage_owner_batch_max) * 2) {
      fail("stage2 maintenance queue depth must cover two intents per RPC batch");
    }
    if (is_server && server_index_file.empty()) {
      fail("storage node requires --server-index-file");
    }
  }

public:
  friend std::ostream& operator<<(
      std::ostream& output, const IndexConfiguration& config) {
    output << static_cast<const Configuration&>(config);
    constexpr i32 width = 34;
    constexpr i32 line_width = 68;
    output << std::left << std::setfill(' ');

    if (config.is_initiator) {
      output << std::setw(width) << "index prefix: " << config.index_prefix << '\n';
      output << std::setw(width) << "threads: " << config.num_threads << '\n';
      output << std::setw(width) << "dimension: " << config.dim << '\n';
      output << std::setw(width) << "max vectors: " << config.max_vectors << '\n';
      output << std::setw(width) << "vector ID namespace [0,N): "
             << config.vector_id_namespace_size << '\n';
      output << std::setw(width) << "K / R: "
             << config.k << " / " << config.R << '\n';
      output << std::setw(width) << "vector data type: "
             << config.vector_data_type << '\n';
      output << std::setfill('-') << std::setw(line_width) << "" << '\n';
      output << std::setfill(' ');
      output << std::setw(width) << "query engine: "
             << "persistent_gpu_opq_pq" << '\n';
      output << std::setw(width) << "remote transport: "
             << "GPU-initiated GPUNetIO" << '\n';
      output << std::setw(width) << "GPU device: " << config.gpu_device << '\n';
      output << std::setw(width) << "GPU concurrent query slots: "
             << config.gpu_query_slots << '\n';
      output << std::setw(width) << "GPU memory limit/reserve GiB: "
             << config.gpu_memory_limit_gb << "/"
             << config.gpu_memory_reserve_gb << '\n';
      output << std::setw(width) << "GPU traversal/rerank width: "
             << config.gpu_traversal_beam_width << "/"
             << config.gpu_final_rerank_width << '\n';
      output << std::setw(width) << "GPU max expansions: "
             << config.gpu_max_expansions << '\n';
      output << std::setw(width) << "GPU graph commit/issue width: "
             << config.gpu_graph_commit_width << "/"
             << config.gpu_graph_issue_width << '\n';
      output << std::setw(width) << "GPU graph read policy: "
             << config.gpu_query_graph_read_policy << '\n';
      output << std::setw(width) << "GPU Beam merge policy: "
             << config.gpu_query_beam_merge_policy << '\n';
      output << std::setw(width) << "query RDMA trace mode/rate: "
             << config.query_rdma_trace_mode << "/"
             << config.query_rdma_trace_sample_rate << '\n';
      output << std::setw(width) << "query RDMA trace output: "
             << config.query_rdma_trace_output << '\n';
      output << std::setw(width) << "GPU RDMA QPs per storage node: "
             << config.gpu_rdma_qps << '\n';
      output << std::setw(width) << "GPU direct CQ timeout ms: "
             << config.gpu_direct_timeout_ms << '\n';
      output << std::setw(width) << "GPU persistent blocks/SM cap: "
             << config.gpu_persistent_blocks_per_sm << '\n';
      output << std::setw(width) << "compute updates enabled: "
             << std::boolalpha << config.enable_updates << '\n';
      output << std::setw(width) << "storage update protocol: "
             << "centroid-home two-stage" << '\n';
      output << std::setw(width) << "storage RPC depth/batch: "
             << config.storage_owner_rpc_depth << "/"
             << config.storage_owner_batch_max << '\n';
      output << std::setw(width) << "storage batch max wait us: "
             << config.storage_owner_batch_max_wait_us << '\n';
      output << std::setw(width) << "storage Stage2 batch max wait us: "
             << config.storage_owner_stage2_batch_max_wait_us << '\n';
      output << std::setw(width) << "defer fresh-insert Stage1 prune: "
             << config.storage_owner_defer_stage1_prune << '\n';
      output << std::setw(width) << "Stage2 exact score-many: "
             << config.storage_owner_stage2_score_many << '\n';
      output << std::setw(width) << "Stage2 graph issue width: "
             << config.storage_owner_stage2_graph_issue_width << '\n';
      output << std::setw(width) << "storage peer QPs per peer: "
             << config.storage_owner_peer_qps_per_peer << '\n';
      output << std::setw(width) << "storage stage2 maintenance: "
             << config.storage_owner_maintenance_workers << " workers, backlog "
             << config.storage_owner_maintenance_queue_depth << '\n';
      output << std::setfill('=') << std::setw(line_width) << "" << '\n';
    } else if (config.is_server) {
      output << std::setw(width) << "index prefix: " << config.index_prefix << '\n';
      output << std::setw(width) << "storage shard: "
             << config.server_index_file << '\n';
      output << std::setw(width) << "storage id: " << config.storage_id << '\n';
      output << std::setw(width) << "base vectors: " << config.max_vectors << '\n';
      output << std::setw(width) << "vector ID namespace [0,N): "
             << config.vector_id_namespace_size << '\n';
      output << std::setw(width) << "registered memory GiB: "
             << config.mn_memory_gb << '\n';
      output << std::setfill('=') << std::setw(line_width) << "" << '\n';
    }
    return output;
  }
};

}  // namespace configuration
