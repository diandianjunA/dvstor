#pragma once

/**
 * RDMA read operations for the Vamana index.
 */

#include <chrono>
#include <cstring>
#include <limits>
#include <thread>

#include "compute_thread.hh"
#include "coroutine.hh"
#include "remote_pointer.hh"
#include "rdma/rdma_send_chain.hh"
#include "vamana/storage_layout_resolver.hh"
#include "vamana/vamana_neighborlist.hh"
#include "vamana/vamana_node.hh"
#include "rdma/vector_batch_planner.hh"

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

struct VectorBatchReadScratch {
    struct ChunkBuffers {
        vec<ibv_send_wr> wrs;
        vec<ibv_sge> sges;
    };

    vec<ChunkBuffers> chunks;
    VectorReadPlannerScratch planner_scratch;
    VectorReadBatchPlan batch_plan;
    vec<u32> request_nodes;
    vec<u32> qp_counts;
    vec<vec<u32>> outstanding_wrs;
    vec<u32> tie_breakers;
    vec<vec<bool>> actual_qps;
    vec<u64> local_addrs;
    vec<u32> local_lkeys;
    vec<u64> remote_addrs;
    vec<u32> remote_rkeys;

    void prepare(const VectorReadBatchPlan& plan) {
        if (chunks.size() < plan.chunks.size()) chunks.resize(plan.chunks.size());
        for (u32 i = 0; i < plan.chunks.size(); ++i) {
            auto& buffers = chunks[i];
            const size_t count = plan.chunks[i].request_count;
            buffers.wrs.clear();
            buffers.sges.clear();
            if (buffers.wrs.capacity() < count) buffers.wrs.reserve(count);
            if (buffers.sges.capacity() < count) buffers.sges.reserve(count);
        }
    }
};

inline thread_local VectorBatchReadScratch vector_batch_read_scratch;

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
    const size_t read_size = VamanaNode::offset_id() + sizeof(node_t);
    byte_t* buffer = thread->buffer_allocator.allocate_buffer(read_size);

    track_total_rdma_read(thread, read_size);
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()][0]->qp;
    qp->post_send(reinterpret_cast<u64>(buffer),
                  read_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(rptr.memory_node()),
                  ::vamana::StorageLayoutResolver::header(rptr).offset,
                  0,
                  thread->create_wr_id());

    struct awaitable {
        byte_t* buffer;
        size_t read_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        node_t await_resume() {
            node_t id = std::numeric_limits<node_t>::max();
            const u64 header = *reinterpret_cast<const u64*>(buffer);
            if ((header & VamanaNode::HEADER_DELETED) == 0) {
                id = *reinterpret_cast<const node_t*>(buffer + VamanaNode::offset_id());
            }
            thread->buffer_allocator.free_buffer(buffer, read_size);
            return id;
        }
    };

    return awaitable{buffer, read_size, thread};
}

struct NeighborReadAwaitable {
    RemotePtr node_rptr{};
    byte_t* local_buffer{nullptr};
    size_t read_size{0};
    const u_ptr<ComputeThread>* thread_ptr{nullptr};
    bool hot_graph_entry{false};

    NeighborReadAwaitable() = default;
    NeighborReadAwaitable(RemotePtr rptr, byte_t* buf, size_t size, const u_ptr<ComputeThread>* tp,
                          bool hot_graph = false)
        : node_rptr(rptr), local_buffer(buf), read_size(size), thread_ptr(tp), hot_graph_entry(hot_graph) {}

    // Explicit move: the compiler-generated move would copy the raw
    // local_buffer pointer without nulling the source, causing the
    // source destructor to free the buffer (double-free / use-after-free).
    NeighborReadAwaitable(NeighborReadAwaitable&& other) noexcept
        : node_rptr(other.node_rptr)
        , local_buffer(other.local_buffer)
        , read_size(other.read_size)
        , thread_ptr(other.thread_ptr)
        , hot_graph_entry(other.hot_graph_entry) {
        other.local_buffer = nullptr;
    }
    NeighborReadAwaitable& operator=(NeighborReadAwaitable&& other) noexcept {
        if (this != &other) {
            if (local_buffer && thread_ptr)
                (*thread_ptr)->buffer_allocator.free_buffer(local_buffer, read_size);
            node_rptr = other.node_rptr;
            local_buffer = other.local_buffer;
            read_size = other.read_size;
            thread_ptr = other.thread_ptr;
            hot_graph_entry = other.hot_graph_entry;
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
        byte_t* neighbor_buffer = local_buffer;
        size_t neighbor_size = read_size;
        if (hot_graph_entry) {
            neighbor_size = VamanaNode::neighbor_read_size();
            neighbor_buffer = (*thread_ptr)->buffer_allocator.allocate_buffer(neighbor_size);
            bool ok = false;
            constexpr u32 kMaxReadAttempts = 3;
            for (u32 attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
                ok = VamanaNode::decode_hot_graph_entry(local_buffer, neighbor_buffer);
                if (ok || attempt + 1 == kMaxReadAttempts) {
                    break;
                }
                auto& thread = *thread_ptr;
                const auto neighbor_read =
                    ::vamana::StorageLayoutResolver::neighbor_read(node_rptr);
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
                              neighbor_read.address.offset,
                              0,
                              thread->create_wr_id());
                const u32 coro_id = thread->current_coroutine_id();
                while (thread->post_balances[coro_id].load(std::memory_order_acquire) != 0) {
                    thread->poll_cq();
                    std::this_thread::yield();
                }
            }
            (*thread_ptr)->buffer_allocator.free_buffer(local_buffer, read_size);
            if (!ok) {
                std::memset(neighbor_buffer, 0, neighbor_size);
            }
        }
        auto nlist = std::make_shared<VamanaNeighborlist>(neighbor_buffer, neighbor_size, thread_ptr->get());
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
    const auto neighbor_read = ::vamana::StorageLayoutResolver::neighbor_read(node_rptr);
    const bool use_hot_graph = neighbor_read.compact;
    const size_t read_size = neighbor_read.address.size;
    const u64 remote_offset = neighbor_read.address.offset;
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
                  remote_offset,
                  0,
                  thread->create_wr_id());

    return NeighborReadAwaitable(node_rptr, local_buffer, read_size, thread_ptr, use_hot_graph);
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
    if (!direct_to_gpu || using_destinations) {
        host_buffers.reserve(node_rptrs.size());
    }
    if (node_rptrs.empty()) {
        return VectorBatchReadAwaitable(
            VectorBatchReadResult{std::move(host_buffers), direct_to_gpu});
    }

    ++thread->stats.vector_rdma_batch_calls;
    if (thread->is_query_worker()) ++thread->stats.query_vector_rdma_batch_calls;

    auto& scratch = vector_batch_read_scratch;
    const u32 num_nodes = static_cast<u32>(thread->ctx->qps.size());
    scratch.request_nodes.clear();
    scratch.request_nodes.reserve(node_rptrs.size());
    scratch.qp_counts.clear();
    scratch.qp_counts.reserve(num_nodes);
    for (u32 node = 0; node < num_nodes; ++node) {
        scratch.qp_counts.push_back(static_cast<u32>(thread->ctx->qps[node].size()));
    }
    for (const auto& rptr : node_rptrs) scratch.request_nodes.push_back(rptr.memory_node());

    u32 chain_size = thread->ctx->effective_chain_size();
    if (num_nodes > 0 && !thread->ctx->qps[0].empty()) {
        const u32 bulk_qp = thread->ctx->qps[0].size() > 1 ? 1 : 0;
        chain_size = std::min(chain_size, thread->ctx->qp_credit_limit(0, bulk_qp));
    }
    thread->ctx->qp_outstanding_snapshot(scratch.outstanding_wrs);
    thread->ctx->next_qp_tie_breakers(scratch.tie_breakers);
    plan_vector_read_batch(
        scratch.request_nodes, scratch.qp_counts, scratch.outstanding_wrs,
        scratch.tie_breakers, chain_size, thread->ctx->batch_options().adaptive,
        scratch.batch_plan, scratch.planner_scratch);
    const auto& batch_plan = scratch.batch_plan;
    thread->stats.vector_rdma_active_nodes += batch_plan.active_nodes;
    thread->stats.vector_rdma_max_chain_wrs = std::max<size_t>(
        thread->stats.vector_rdma_max_chain_wrs, batch_plan.max_chain_wrs);
    scratch.actual_qps.resize(num_nodes);
    for (u32 node = 0; node < num_nodes; ++node) {
        scratch.actual_qps[node].assign(thread->ctx->qps[node].size(), false);
    }

    scratch.local_addrs.resize(node_rptrs.size());
    scratch.local_lkeys.resize(node_rptrs.size());
    scratch.remote_addrs.resize(node_rptrs.size());
    scratch.remote_rkeys.resize(node_rptrs.size());

    for (size_t i = 0; i < node_rptrs.size(); ++i) {
        const auto& rptr = node_rptrs[i];
        if (using_destinations) {
            const auto& dst = (*destinations)[i];
            scratch.local_addrs[i] = dst.local_addr;
            scratch.local_lkeys[i] = dst.gpu_destination ? dst.lkey : thread->ctx->get_lkey();
            if (dst.host_buffer) host_buffers.push_back(dst.host_buffer);
        } else if (direct_to_gpu) {
            scratch.local_addrs[i] = reinterpret_cast<u64>(gpu_buffer) + i * vec_size;
            scratch.local_lkeys[i] = gpu_lkey;
        } else {
            byte_t* local_buffer = thread->buffer_allocator.allocate_buffer(vec_size);
            host_buffers.push_back(local_buffer);
            scratch.local_addrs[i] = reinterpret_cast<u64>(local_buffer);
            scratch.local_lkeys[i] = thread->ctx->get_lkey();
        }

        track_vector_rdma_read(thread, vec_size);
        const u32 node = rptr.memory_node();
        auto* token = thread->ctx->get_remote_mrt(node);
        const u64 resolved_offset = read_size_override == 0
            ? ::vamana::StorageLayoutResolver::vector(rptr).offset
            : rptr.byte_offset() + node_offset_override;
        scratch.remote_addrs[i] = token->address + resolved_offset;
        scratch.remote_rkeys[i] = token->rkey;
    }

    scratch.prepare(batch_plan);
    for (u32 chunk_index = 0; chunk_index < batch_plan.chunks.size(); ++chunk_index) {
        const auto& chunk = batch_plan.chunks[chunk_index];
        auto& buffers = scratch.chunks[chunk_index];
        for (u32 i = 0; i < chunk.request_count; ++i) {
            const u32 request_index =
                batch_plan.request_order[chunk.request_offset + i];
            buffers.sges.push_back({});
            auto& sge = buffers.sges.back();
            sge.addr = scratch.local_addrs[request_index];
            sge.length = static_cast<u32>(vec_size);
            sge.lkey = scratch.local_lkeys[request_index];

            buffers.wrs.push_back({});
            auto& wr = buffers.wrs.back();
            wr.wr_id = 0;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_READ;
            wr.send_flags = 0;
            wr.wr.rdma.remote_addr = scratch.remote_addrs[request_index];
            wr.wr.rdma.rkey = scratch.remote_rkeys[request_index];
            wr.next = nullptr;
        }
    }

    for (u32 chunk_index = 0; chunk_index < batch_plan.chunks.size(); ++chunk_index) {
        const auto& chunk = batch_plan.chunks[chunk_index];
        auto& buffers = scratch.chunks[chunk_index];
        auto& wrs = buffers.wrs;
        auto& sges = buffers.sges;
        const u32 wr_count = static_cast<u32>(wrs.size());
        lib_assert(wr_count > 0, "empty vector RDMA chunk");
        for (u32 i = 0; i < wr_count; ++i) {
            wrs[i].sg_list = &sges[i];
            wrs[i].next = i + 1 < wr_count ? &wrs[i + 1] : nullptr;
        }

        std::chrono::steady_clock::time_point wait_start{};
        bool waited_for_credit = false;
        u32 selected_qp = chunk.qp_index;
        while (!thread->ctx->try_reserve_bulk_qp_wrs(
            chunk.memory_node, chunk.qp_index, wr_count, selected_qp)) {
            if (!waited_for_credit) {
                wait_start = std::chrono::steady_clock::now();
            }
            waited_for_credit = true;
            thread->poll_cq();
            std::this_thread::yield();
        }
        scratch.actual_qps[chunk.memory_node][selected_qp] = true;
        thread->stats.vector_rdma_qp_high_water_wrs = std::max<size_t>(
            thread->stats.vector_rdma_qp_high_water_wrs,
            thread->ctx->qp_runtime[chunk.memory_node][selected_qp]
                ->high_water_wrs.load(std::memory_order_relaxed));
        if (waited_for_credit) {
            ++thread->stats.vector_rdma_credit_waits;
            thread->stats.vector_rdma_credit_wait_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_start).count();
        }

        u64 completion_id = 0;
        while (completion_id == 0) {
            completion_id = thread->ctx->try_create_batch_completion(
                thread->ctx_tid, thread->current_coroutine_id(),
                chunk.memory_node, selected_qp, wr_count);
            if (completion_id == 0) {
                ++thread->stats.vector_rdma_completion_token_waits;
                thread->poll_cq();
                std::this_thread::yield();
            }
        }

        wrs.back().wr_id = completion_id;
        wrs.back().send_flags = IBV_SEND_SIGNALED;
        thread->track_post();
        ++thread->stats.vector_rdma_chunks;
        thread->stats.vector_rdma_chain_wrs += wr_count;
        if (thread->is_query_worker()) ++thread->stats.query_vector_rdma_cqes;

        const auto post_result = ::rdma::post_send_chain_with_retry(
            wrs.data(),
            [&](ibv_send_wr* first, ibv_send_wr** bad) {
                return ibv_post_send(
                    thread->ctx->qps[chunk.memory_node][selected_qp]
                        ->qp->get_ibv_qp(),
                    first, bad);
            },
            [&] {
                thread->poll_cq();
                std::this_thread::yield();
            });
        thread->stats.vector_rdma_post_send_calls += post_result.post_calls;
        thread->stats.vector_rdma_post_send_retries += post_result.retries;
        if (!post_result.success) {
            ++thread->stats.vector_rdma_post_send_errors;
            lib_failure("Cannot post vector RDMA READ chain: rc=" +
                        std::to_string(post_result.error));
        }
    }

    for (const auto& node_qps : scratch.actual_qps) {
        thread->stats.vector_rdma_active_qps += static_cast<size_t>(
            std::count(node_qps.begin(), node_qps.end(), true));
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
