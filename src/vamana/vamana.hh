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

#include "cache/cache.hh"
#include "common/constants.hh"
#include "common/debug.hh"
#include "common/types.hh"
#include "compute_thread.hh"
#include "coroutine.hh"
#include "gpu/gpu_awaitable.hh"
#include "gpu/gpu_kernel_launcher.hh"
#include "rdma/vamana_rdma_operations.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_neighborlist.hh"
#include "vamana/vamana_node.hh"

namespace vamana {

constexpr u32 kRabitqSearchBeamSlack = 64;

template <class Distance>
class Vamana {
public:
    Vamana(u32 R, u32 beam_width, u32 beam_width_construction, f64 alpha,
           u32 k, u32 rabitq_bits, u32 dim, bool use_cache, bool use_rabitq_search)
        : R_(R),
          beam_width_(beam_width),
          beam_width_construction_(beam_width_construction),
          alpha_(static_cast<f32>(alpha)),
          k_(k),
          rabitq_bits_(rabitq_bits),
          dim_(dim),
          use_cache_(use_cache),
          use_rabitq_search_(use_rabitq_search) {
        lib_assert(beam_width_ >= k_, "beam_width must be >= k");
        VamanaNode::init_static_storage(dim, R, rabitq_bits);
    }

    // =========================================================================
    // Search (knn)
    // =========================================================================

#include "vamana/vamana_search.ipp"
#include "vamana/vamana_insert.ipp"
#include "vamana/vamana_helpers.ipp"

private:
    const u32 R_;
    const u32 beam_width_;
    const u32 beam_width_construction_;
    const f32 alpha_;
    const u32 k_;
    const u32 rabitq_bits_;
    const u32 dim_;
    const bool use_cache_;
    const bool use_rabitq_search_;
};

}  // namespace vamana
