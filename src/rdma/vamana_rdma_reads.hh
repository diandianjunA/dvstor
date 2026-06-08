#pragma once

/**
 * RDMA read operations for the Vamana index.
 */

#include <cstring>

#include "common/neighbor_cache.hh"
#include "compute_thread.hh"
#include "coroutine.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_neighborlist.hh"
#include "vamana/vamana_node.hh"

namespace rdma::vamana {

struct VectorBatchReadResult {
    vec<byte_t*> host_buffers;
    bool direct_to_gpu{false};
};

struct BatchReadDestination {
    u64 local_addr{};
    u32 lkey{};
    byte_t* host_buffer{nullptr};
    bool gpu_destination{false};
};

inline void track_total_rdma_read(const u_ptr<ComputeThread>& thread, size_t bytes, size_t ops = 1) {
    thread->stats.rdma_reads_in_bytes += bytes;
    thread->stats.rdma_read_ops += ops;
    if (thread->is_query_worker()) {
        thread->stats.query_rdma_reads_in_bytes += bytes;
        thread->stats.query_rdma_read_ops += ops;
    } else if (thread->is_insert_worker()) {
        thread->stats.build_rdma_reads_in_bytes += bytes;
        thread->stats.build_rdma_read_ops += ops;
    }
}

inline void track_neighbor_rdma_read(const u_ptr<ComputeThread>& thread, size_t bytes, size_t ops = 1) {
    track_total_rdma_read(thread, bytes, ops);
    if (thread->is_query_worker()) {
        thread->stats.query_neighbor_rdma_reads_in_bytes += bytes;
        thread->stats.query_neighbor_rdma_read_ops += ops;
    } else if (thread->is_insert_worker()) {
        thread->stats.build_neighbor_rdma_reads_in_bytes += bytes;
        thread->stats.build_neighbor_rdma_read_ops += ops;
    }
}

inline void track_vector_rdma_read(const u_ptr<ComputeThread>& thread, size_t bytes, size_t ops = 1) {
    track_total_rdma_read(thread, bytes, ops);
    if (thread->is_query_worker()) {
        thread->stats.query_vector_rdma_reads_in_bytes += bytes;
        thread->stats.query_vector_rdma_read_ops += ops;
    } else if (thread->is_insert_worker()) {
        thread->stats.build_vector_rdma_reads_in_bytes += bytes;
        thread->stats.build_vector_rdma_read_ops += ops;
    }
}

inline auto read_vamana_node(RemotePtr rptr, const u_ptr<ComputeThread>& thread) {
    const size_t read_size = VamanaNode::size_until_vector_end();
    byte_t* node_ptr = thread->buffer_allocator.allocate_buffer(read_size);

    track_vector_rdma_read(thread, read_size);
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(node_ptr),
                  read_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(rptr.memory_node()),
                  rptr.byte_offset(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        RemotePtr rptr;
        byte_t* node_ptr;
        size_t read_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        s_ptr<VamanaNode> await_resume() {
            return std::make_shared<VamanaNode>(node_ptr, read_size, rptr, thread.get());
        }
    };

    return awaitable{rptr, node_ptr, read_size, thread};
}

inline auto read_vamana_node_full(RemotePtr rptr, const u_ptr<ComputeThread>& thread) {
    const size_t read_size = VamanaNode::total_size();
    byte_t* node_ptr = thread->buffer_allocator.allocate_vamana_node(thread->get_id());

    track_vector_rdma_read(thread, read_size);
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(node_ptr),
                  read_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(rptr.memory_node()),
                  rptr.byte_offset(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        RemotePtr rptr;
        byte_t* node_ptr;
        size_t read_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        s_ptr<VamanaNode> await_resume() {
            return std::make_shared<VamanaNode>(node_ptr, read_size, rptr, thread.get());
        }
    };

    return awaitable{rptr, node_ptr, read_size, thread};
}


inline auto read_vamana_id(RemotePtr rptr, const u_ptr<ComputeThread>& thread) {
    node_t* id_ptr = reinterpret_cast<node_t*>(thread->buffer_allocator.allocate_buffer(sizeof(node_t)));

    track_total_rdma_read(thread, sizeof(node_t));
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(id_ptr),
                  sizeof(node_t),
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(rptr.memory_node()),
                  rptr.byte_offset() + VamanaNode::offset_id(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        node_t* id_ptr;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        node_t await_resume() {
            const node_t id = *id_ptr;
            thread->buffer_allocator.free_buffer(reinterpret_cast<byte_t*>(id_ptr), sizeof(node_t));
            return id;
        }
    };

    return awaitable{id_ptr, thread};
}

inline auto read_vamana_neighbors(RemotePtr node_rptr, const u_ptr<ComputeThread>& thread) {
    const size_t read_size = sizeof(u8) + VamanaNode::NEIGHBORS_SIZE;
    byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(read_size);

    const QP& qp = thread->ctx->qps[node_rptr.memory_node()]->qp;
    track_neighbor_rdma_read(thread, read_size, 2);
    thread->track_post();
    qp->post_send(reinterpret_cast<u64>(local_buffer),
                  sizeof(u8),
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node_rptr.memory_node()),
                  node_rptr.byte_offset() + VamanaNode::offset_edge_count(),
                  0,
                  thread->create_wr_id());

    thread->track_post();
    qp->post_send(reinterpret_cast<u64>(local_buffer + sizeof(u8)),
                  VamanaNode::NEIGHBORS_SIZE,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node_rptr.memory_node()),
                  node_rptr.byte_offset() + VamanaNode::offset_neighbors(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        byte_t* local_buffer;
        size_t read_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        s_ptr<VamanaNeighborlist> await_resume() {
            return std::make_shared<VamanaNeighborlist>(local_buffer, read_size, thread.get());
        }
    };

    return awaitable{local_buffer, read_size, thread};
}

// Cached version: checks NeighborCache before issuing RDMA reads.
// On cache hit, returns immediately without suspending the coroutine.
//
// When mutable_access = false (default, search path): cache-hit nlist borrows the
// thread-local scratch buffer (no buffer_allocator allocation, no extra memcpy).
// When mutable_access = true (insert path, reverse-edge update): cache-hit nlist
// gets its own allocated buffer because the caller may call nlist->add().
inline auto read_vamana_neighbors_cached(RemotePtr node_rptr,
                                          const u_ptr<ComputeThread>& thread,
                                          NeighborCache* cache,
                                          bool mutable_access = false) {
    const size_t read_size = sizeof(u8) + VamanaNode::NEIGHBORS_SIZE;

    // Single awaitable type for both cache-hit and cache-miss paths
    struct awaitable {
        byte_t* local_buffer{nullptr};
        size_t read_size{0};
        const u_ptr<ComputeThread>* thread_ptr{nullptr};
        s_ptr<VamanaNeighborlist> cached_result;
        RemotePtr node_rptr;
        NeighborCache* cache{nullptr};
        bool is_cache_hit{false};

        bool await_ready() const {
            return is_cache_hit;  // cache hit: no suspension; miss: suspend
        }
        static void await_suspend(std::coroutine_handle<>) {}
        s_ptr<VamanaNeighborlist> await_resume() {
            if (is_cache_hit) {
                return std::move(cached_result);
            }
            auto nlist = std::make_shared<VamanaNeighborlist>(local_buffer, read_size,
                                                               thread_ptr->get());
            if (cache) {
                (*thread_ptr)->stats.neighbor_cache_misses++;
                cache->insert(node_rptr, nlist->num_neighbors(), nlist->view().data());
            }
            return nlist;
        }
    };

    if (cache) {
        // Lazy-init per-thread scratch buffer in VamanaNeighborlist format:
        //   [count(1B)][neighbors(R*8B)]
        auto& scratch = thread->cached_neighbor_data;
        if (scratch.empty()) {
            scratch.resize(VamanaNeighborlist::buffer_size());
        }

        u8 cached_count = 0;
        // find() writes neighbors into scratch+1, returns hit and count
        if (cache->find(node_rptr,
                         reinterpret_cast<RemotePtr*>(scratch.data() + sizeof(u8)),
                         cached_count)) {
            thread->stats.neighbor_cache_hits++;

            if (!mutable_access) {
                // Fast path (read-only): borrow thread-local scratch buffer.
                // No buffer_allocator allocation, no second memcpy.
                scratch[0] = static_cast<byte_t>(cached_count);
                auto nlist = std::make_shared<VamanaNeighborlist>(
                    scratch.data(), read_size, thread.get(),
                    false /* owns_buffer = false */);
                return awaitable{nullptr, read_size, &thread, std::move(nlist),
                                 RemotePtr{}, nullptr, true};
            }

            // Mutable path: caller may modify nlist (e.g., add()).
            // Allocate a dedicated buffer and copy from scratch.
            byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(read_size);
            *reinterpret_cast<u8*>(local_buffer) = cached_count;
            std::memcpy(local_buffer + sizeof(u8), scratch.data() + sizeof(u8),
                        static_cast<size_t>(cached_count) * sizeof(RemotePtr));
            auto nlist = std::make_shared<VamanaNeighborlist>(local_buffer, read_size, thread.get());
            return awaitable{local_buffer, read_size, &thread, std::move(nlist),
                             RemotePtr{}, nullptr, true};
        }
    }

    // Cache miss — perform the original two-step RDMA read
    byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(read_size);

    const QP& qp = thread->ctx->qps[node_rptr.memory_node()]->qp;
    track_neighbor_rdma_read(thread, read_size, 2);
    thread->track_post();
    qp->post_send(reinterpret_cast<u64>(local_buffer),
                  sizeof(u8),
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node_rptr.memory_node()),
                  node_rptr.byte_offset() + VamanaNode::offset_edge_count(),
                  0,
                  thread->create_wr_id());

    thread->track_post();
    qp->post_send(reinterpret_cast<u64>(local_buffer + sizeof(u8)),
                  VamanaNode::NEIGHBORS_SIZE,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node_rptr.memory_node()),
                  node_rptr.byte_offset() + VamanaNode::offset_neighbors(),
                  0,
                  thread->create_wr_id());

    return awaitable{local_buffer, read_size, &thread, nullptr, node_rptr, cache, false};
}

inline auto batch_read_vectors(const vec<RemotePtr>& node_rptrs,
                               const u_ptr<ComputeThread>& thread,
                               const vec<BatchReadDestination>* destinations = nullptr,
                               void* gpu_buffer = nullptr,
                               u32 gpu_lkey = 0) {
    const size_t vec_size = VamanaNode::vector_bytes();
    const bool using_destinations = destinations != nullptr && !destinations->empty();
    const bool direct_to_gpu = using_destinations || (gpu_buffer != nullptr && gpu_lkey != 0);
    vec<byte_t*> host_buffers;
    host_buffers.reserve(node_rptrs.size());

    for (size_t i = 0; i < node_rptrs.size(); ++i) {
        const auto& rptr = node_rptrs[i];
        u64 local_addr;
        u32 lkey;
        if (using_destinations) {
            const auto& dst = (*destinations)[i];
            local_addr = dst.local_addr;
            lkey = dst.gpu_destination ? dst.lkey : thread->ctx->get_lkey();
            if (dst.host_buffer) {
                host_buffers.push_back(dst.host_buffer);
            }
        } else if (direct_to_gpu) {
            local_addr = reinterpret_cast<u64>(gpu_buffer) + i * vec_size;
            lkey = gpu_lkey;
        } else {
            byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(vec_size);
            host_buffers.push_back(local_buffer);
            local_addr = reinterpret_cast<u64>(local_buffer);
            lkey = thread->ctx->get_lkey();
        }

        track_vector_rdma_read(thread, vec_size);
        thread->track_post();

        const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
        qp->post_send(local_addr,
                      vec_size,
                      lkey,
                      IBV_WR_RDMA_READ,
                      true,
                      false,
                      thread->ctx->get_remote_mrt(rptr.memory_node()),
                      rptr.byte_offset() + VamanaNode::offset_vector(),
                      0,
                      thread->create_wr_id());
    }

    struct awaitable {
        VectorBatchReadResult result;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        VectorBatchReadResult await_resume() { return std::move(result); }
    };

    return awaitable{VectorBatchReadResult{std::move(host_buffers), direct_to_gpu}};
}

inline auto batch_read_vectors(const vec<RemotePtr>& node_rptrs,
                               const u_ptr<ComputeThread>& thread,
                               void* gpu_buffer,
                               u32 gpu_lkey) {
    return batch_read_vectors(node_rptrs, thread, nullptr, gpu_buffer, gpu_lkey);
}

inline auto read_vamana_nodes(const span<RemotePtr> remote_ptrs, const u_ptr<ComputeThread>& thread) {
    vec<s_ptr<VamanaNode>> nodes;
    nodes.reserve(remote_ptrs.size());

    const size_t read_size = VamanaNode::size_until_vector_end();

    for (auto& rptr : remote_ptrs) {
        byte_t* node_ptr = thread->buffer_allocator.allocate_buffer(read_size);
        nodes.emplace_back(std::make_shared<VamanaNode>(node_ptr, read_size, rptr, thread.get()));

        track_vector_rdma_read(thread, read_size);
        thread->track_post();

        const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
        qp->post_send(reinterpret_cast<u64>(node_ptr),
                      read_size,
                      thread->ctx->get_lkey(),
                      IBV_WR_RDMA_READ,
                      true,
                      false,
                      thread->ctx->get_remote_mrt(rptr.memory_node()),
                      rptr.byte_offset(),
                      0,
                      thread->create_wr_id());
    }

    struct awaitable {
        vec<s_ptr<VamanaNode>> nodes;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        vec<s_ptr<VamanaNode>> await_resume() { return std::move(nodes); }
    };

    return awaitable{std::move(nodes)};
}

inline auto read_medoid_ptr(const u_ptr<ComputeThread>& thread) {
    track_total_rdma_read(thread, sizeof(u64));
    thread->track_post();

    const QP& qp = thread->ctx->qps[0]->qp;
    qp->post_send(reinterpret_cast<u64>(thread->coros_pointer_slot()),
                  sizeof(u64),
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(0),
                  8,
                  0,
                  thread->create_wr_id());

    struct awaitable {
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        RemotePtr await_resume() const { return RemotePtr{*thread->coros_pointer_slot()}; }
    };

    return awaitable{thread};
}

}  // namespace rdma::vamana
