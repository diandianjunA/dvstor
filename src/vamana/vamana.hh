#pragma once

/**
 * Vamana Index: GPU-accelerated beam-search graph index with RDMA disaggregated memory.
 *
 * Replaces HNSW with a single-layer directed graph using:
 *  - Beam search (instead of multi-layer greedy descent)
 *  - RobustPrune (alpha-based diversity pruning instead of HNSW heuristic)
 *  - Exact L2 distances for search (GPU-accelerated)
 *  - Full-precision L2 distances for insert/prune (GPU-accelerated)
 *  - Coroutine-based RDMA overlap (from DVSTOR baseline)
 */

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <type_traits>

#include "common/constants.hh"
#include "common/distance.hh"
#include "common/types.hh"
#include "compute_thread.hh"
#include "coroutine.hh"
#include "gpu/gpu_awaitable.hh"
#include "gpu/gpu_kernel_launcher.hh"
#include "rdma/vamana_rdma_operations.hh"
#include "remote_pointer.hh"
#include "vamana/rabitq_cache.hh"
#include "vamana/vamana_neighborlist.hh"
#include "vamana/vamana_node.hh"

namespace vamana {

template <class Distance>
inline distance_t distance_to_stored_vector(const span<const element_t> query, const byte_t* stored) {
    return typed_distance_float_query(query,
                                      stored,
                                      VamanaNode::vector_dtype(),
                                      VamanaNode::DIM,
                                      std::is_same_v<Distance, IPDistance>);
}

template <class Distance>
inline distance_t distance_to_stored_vector(const byte_t* query, VectorDType query_dtype, const byte_t* stored) {
    if constexpr (std::is_same_v<Distance, IPDistance>) {
        return typed_ip_distance(query, query_dtype, stored, VamanaNode::vector_dtype(), VamanaNode::DIM);
    } else {
        return typed_l2_distance(query, query_dtype, stored, VamanaNode::vector_dtype(), VamanaNode::DIM);
    }
}

template <class Distance>
class Vamana {
public:
    Vamana(u32 R, u32 beam_width, u32 beam_width_construction, f64 alpha,
           u32 k, u32 dim, VectorDType vector_dtype)
        : R_(R),
          beam_width_(beam_width),
          beam_width_construction_(beam_width_construction),
          alpha_(static_cast<f32>(alpha)),
          k_(k),
          dim_(dim),
          direct_node_reads_(true) {
        lib_assert(beam_width_ >= k_, "beam_width must be >= k");
        VamanaNode::init_static_storage(dim, R, vector_dtype);
    }


    // =========================================================================
    // Search (knn)
    // =========================================================================

#include "vamana/vamana_search.ipp"
#include "vamana/vamana_insert.ipp"
#include "vamana/vamana_helpers.ipp"

public:
    void set_expansion_batch(u32 k) { expansion_batch_ = k; }
    u32 expansion_batch() const { return expansion_batch_; }
    void set_query_batch_size(u32 q) {
        query_batch_size_ = q;
    }
    u32 query_batch_size() const { return use_rabitq_ ? 1 : query_batch_size_; }
    bool use_rabitq() const { return use_rabitq_; }
    void set_rabitq_cache(const rabitq::Cache* cache) { rabitq_cache_ = cache; }
    void set_rabitq_gate(u32 width, u32 max_width, f32 margin) {
        rabitq_gate_width_ = width;
        rabitq_gate_max_width_ = max_width;
        rabitq_gate_margin_ = std::max(margin, 0.0f);
    }
    void set_rabitq_runtime(u32 coalesce_min, u32 warmup_exact_expansions,
                            bool strict_recall) {
        rabitq_coalesce_min_ = std::max<u32>(1, coalesce_min);
        rabitq_warmup_exact_expansions_ = warmup_exact_expansions;
        rabitq_strict_recall_ = strict_recall;
    }
    void set_use_rabitq(bool v) {
        use_rabitq_ = v;
        if (!v) return;
        rabitq::validate_dimension();
        VamanaNode::enable_rabitq();
    }

private:
    u32 expansion_batch_{1};
    u32 query_batch_size_{1};
    bool use_rabitq_{false};
    const rabitq::Cache* rabitq_cache_{nullptr};
    u32 rabitq_gate_width_{16};
    u32 rabitq_gate_max_width_{24};
    f32 rabitq_gate_margin_{0.05f};
    u32 rabitq_coalesce_min_{32};
    u32 rabitq_warmup_exact_expansions_{4};
    bool rabitq_strict_recall_{true};
    // Retained only for compilation of the unreachable legacy branch below the v2 gate.
    f32 rabitq_confidence_epsilon_{1.9f};
    u32 rabitq_exact_batch_{0};
    u32 rabitq_exact_budget_{0};
    const u32 R_;
    const u32 beam_width_;
    const u32 beam_width_construction_;
    const f32 alpha_;
    const u32 k_;
    const u32 dim_;
    const bool direct_node_reads_;
};

}  // namespace vamana
