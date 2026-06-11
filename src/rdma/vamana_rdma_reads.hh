#pragma once

/**
 * RDMA read operations for the Vamana index.
 */

#include <cstring>

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

    const QP& qp = thread->ctx->qps[rptr.memory_node()][0]->qp;
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

inline auto read_vamana_node_prefix(RemotePtr rptr, const u_ptr<ComputeThread>& thread) {
    const size_t read_size = VamanaNode::NODE_PREFIX_SIZE;
    byte_t* node_ptr = thread->buffer_allocator.allocate_buffer(read_size);

    track_total_rdma_read(thread, read_size);
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()][0]->qp;
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

    const QP& qp = thread->ctx->qps[rptr.memory_node()][0]->qp;
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

    const QP& qp = thread->ctx->qps[rptr.memory_node()][0]->qp;
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

struct NeighborReadAwaitable {
    byte_t* local_buffer{nullptr};
    size_t read_size{0};
    const u_ptr<ComputeThread>* thread_ptr{nullptr};

    NeighborReadAwaitable() = default;
    NeighborReadAwaitable(byte_t* buf, size_t size, const u_ptr<ComputeThread>* tp)
        : local_buffer(buf), read_size(size), thread_ptr(tp) {}

    // Explicit move: the compiler-generated move would copy the raw
    // local_buffer pointer without nulling the source, causing the
    // source destructor to free the buffer (double-free / use-after-free).
    NeighborReadAwaitable(NeighborReadAwaitable&& other) noexcept
        : local_buffer(other.local_buffer)
        , read_size(other.read_size)
        , thread_ptr(other.thread_ptr) {
        other.local_buffer = nullptr;
    }
    NeighborReadAwaitable& operator=(NeighborReadAwaitable&& other) noexcept {
        if (this != &other) {
            if (local_buffer && thread_ptr)
                (*thread_ptr)->buffer_allocator.free_buffer(local_buffer, read_size);
            local_buffer = other.local_buffer;
            read_size = other.read_size;
            thread_ptr = other.thread_ptr;
            other.local_buffer = nullptr;
        }
        return *this;
    }

    ~NeighborReadAwaitable() {
        // If the awaitable is destroyed without being awaited (e.g.,
        // overwritten by move-assignment or the coroutine frame is freed),
        // release the allocated RDMA buffer back to the freelist.
        // After a proper move, local_buffer is nullptr → no-op.
        if (local_buffer && thread_ptr) {
            (*thread_ptr)->buffer_allocator.free_buffer(local_buffer, read_size);
        }
    }

    bool valid() const { return local_buffer != nullptr; }
    bool await_ready() const { return ready_; }
    void await_suspend(std::coroutine_handle<>) {}
    s_ptr<VamanaNeighborlist> await_resume() {
        auto nlist = std::make_shared<VamanaNeighborlist>(local_buffer, read_size, thread_ptr->get());
        local_buffer = nullptr;
        ready_ = false;
        return nlist;
    }
    void mark_ready() { ready_ = true; }
private:
    bool ready_{false};
};

// NOTE: takes thread as a pointer (not reference) so the awaitable can be
// stored and awaited later.  The pointer must point into the coroutine frame
// (which is heap-allocated and stable across suspensions), NOT the caller's
// stack frame.
inline NeighborReadAwaitable read_vamana_neighbors(RemotePtr node_rptr, const u_ptr<ComputeThread>* thread_ptr) {
    auto& thread = *thread_ptr;
    const size_t read_size = VamanaNode::neighbor_read_size();
    byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(read_size);

    const QP& qp = thread->ctx->qps[node_rptr.memory_node()][0]->qp;
    track_neighbor_rdma_read(thread, read_size, 1);
    thread->track_post();
    qp->post_send(reinterpret_cast<u64>(local_buffer),
                  read_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node_rptr.memory_node()),
                  node_rptr.byte_offset() + VamanaNode::neighbor_read_offset(),
                  0,
                  thread->create_wr_id());

    return NeighborReadAwaitable(local_buffer, read_size, thread_ptr);
}

struct VectorBatchReadAwaitable {
    VectorBatchReadResult result;
    VectorBatchReadAwaitable() = default;
    explicit VectorBatchReadAwaitable(VectorBatchReadResult r)
        : result(std::move(r)) {}
    bool await_ready() const { return ready_; }
    void await_suspend(std::coroutine_handle<>) {}
    VectorBatchReadResult await_resume() { ready_ = false; return std::move(result); }
    void mark_ready() { ready_ = true; }
private:
    bool ready_{false};
};

inline VectorBatchReadAwaitable batch_read_vectors(const vec<RemotePtr>& node_rptrs,
                               const u_ptr<ComputeThread>& thread,
                               const vec<BatchReadDestination>* destinations = nullptr,
                               void* gpu_buffer = nullptr,
                               u32 gpu_lkey = 0,
                               size_t read_size_override = 0,
                               u64 node_offset_override = 0) {
    const size_t vec_size = read_size_override == 0
        ? VamanaNode::vector_bytes() : read_size_override;
    const bool using_destinations = destinations != nullptr && !destinations->empty();
    const bool direct_to_gpu = using_destinations || (gpu_buffer != nullptr && gpu_lkey != 0);
    vec<byte_t*> host_buffers;
    host_buffers.reserve(node_rptrs.size());

    // Group WRs per (memory_node, QP) pair and build linked lists.
    // Round-robin across the QP pool for each node.
    const u32 num_nodes = static_cast<u32>(thread->ctx->qps.size());
    vec<u32> qp_counters(num_nodes, 0);
    vec<vec<vec<ibv_send_wr>>> wr_lists(num_nodes);
    vec<vec<vec<ibv_sge>>> sge_lists(num_nodes);
    for (u32 n = 0; n < num_nodes; ++n) {
        u32 pool_sz = static_cast<u32>(thread->ctx->qps[n].size());
        wr_lists[n].resize(pool_sz);
        sge_lists[n].resize(pool_sz);
        size_t cap = node_rptrs.size() / num_nodes / pool_sz + 1;
        for (u32 p = 0; p < pool_sz; ++p) {
            wr_lists[n][p].reserve(cap);
            sge_lists[n][p].reserve(cap);
        }
    }

    for (size_t i = 0; i < node_rptrs.size(); ++i) {
        const auto& rptr = node_rptrs[i];
        u64 local_addr;
        u32 lkey;
        if (using_destinations) {
            const auto& dst = (*destinations)[i];
            local_addr = dst.local_addr;
            lkey = dst.gpu_destination ? dst.lkey : thread->ctx->get_lkey();
            if (dst.host_buffer) host_buffers.push_back(dst.host_buffer);
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

        const u32 node = rptr.memory_node();
        const u32 qp_idx = (qp_counters[node]++) % wr_lists[node].size();
        auto* token = thread->ctx->get_remote_mrt(node);
        auto& sges = sge_lists[node][qp_idx];
        auto& wrs = wr_lists[node][qp_idx];

        sges.push_back({});
        ibv_sge& sge = sges.back();
        sge.addr = local_addr;
        sge.length = static_cast<u32>(vec_size);
        sge.lkey = lkey;

        wrs.push_back({});
        ibv_send_wr& wr = wrs.back();
        wr.wr_id = thread->create_wr_id();
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = token->address + rptr.byte_offset()
                                 + (read_size_override == 0
                                        ? VamanaNode::offset_vector() : node_offset_override);
        wr.wr.rdma.rkey = token->rkey;
        wr.next = nullptr;
    }

    // Post batched WRs: one ibv_post_send per (node, qp_idx) pair.
    for (u32 node = 0; node < num_nodes; ++node) {
        for (u32 qp = 0; qp < wr_lists[node].size(); ++qp) {
            auto& wrs = wr_lists[node][qp];
            auto& sges = sge_lists[node][qp];
            if (wrs.empty()) continue;
            for (size_t j = 0; j < wrs.size(); ++j) {
                wrs[j].sg_list = &sges[j];
                if (j + 1 < wrs.size()) wrs[j].next = &wrs[j + 1];
            }
            struct ibv_send_wr* bad = nullptr;
            ibv_post_send(thread->ctx->qps[node][qp]->qp->get_ibv_qp(),
                          &wrs[0], &bad);
        }
    }

    return VectorBatchReadAwaitable(VectorBatchReadResult{std::move(host_buffers), direct_to_gpu});
}

// Read arbitrary bytes from each node at a custom offset (e.g. RaBitQ codes).
inline VectorBatchReadAwaitable batch_read_at_offset(
        const vec<RemotePtr>& node_rptrs,
        const u_ptr<ComputeThread>& thread,
        u64 node_offset, size_t read_size,
        void* gpu_buffer, u32 gpu_lkey) {
    return batch_read_vectors(node_rptrs, thread, nullptr, gpu_buffer, gpu_lkey,
                              read_size, node_offset);
}

// Host-buffer variant of batch_read_at_offset: reads to host buffers,
// then the caller is responsible for copying to GPU.
inline VectorBatchReadAwaitable batch_read_at_offset_to_host(
        const vec<RemotePtr>& node_rptrs,
        const u_ptr<ComputeThread>& thread,
        u64 node_offset, size_t read_size) {
    return batch_read_vectors(node_rptrs, thread, nullptr, nullptr, 0,
                              read_size, node_offset);
}

inline VectorBatchReadAwaitable batch_read_vectors(const vec<RemotePtr>& node_rptrs,
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

        const QP& qp = thread->ctx->qps[rptr.memory_node()][0]->qp;
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

    const QP& qp = thread->ctx->qps[0][0]->qp;
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
