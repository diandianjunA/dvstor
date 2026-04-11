#pragma once

/**
 * RDMA atomic operations for the Vamana index.
 * Lock/unlock, node allocation, medoid pointer swap.
 */

#include "rdma/vamana_rdma_reads.hh"

namespace rdma::vamana {

/**
 * Try to lock a VamanaNode via CAS on the header.
 */
inline auto try_lock_vamana_node(const RemotePtr& rptr,
                                 u64 expected_header,
                                 const u_ptr<ComputeThread>& thread) {
    const u64 compare = expected_header & ~VamanaNode::HEADER_NODE_LOCK;
    const u64 swap = compare | VamanaNode::HEADER_NODE_LOCK;

    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_CAS(reinterpret_cast<u64>(thread->coros_pointer_slot()),
                 thread->ctx->get_lkey(),
                 thread->ctx->get_remote_mrt(rptr.memory_node()),
                 rptr.byte_offset(),
                 compare,
                 swap,
                 true,
                 thread->create_wr_id());

    struct awaitable {
        u64 compare;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}

        std::pair<bool, u64> await_resume() const {
            const u64 original_value = *thread->coros_pointer_slot();
            return {original_value == compare, original_value};
        }
    };

    return awaitable{compare, thread};
}

/**
 * Spinlock a VamanaNode: repeatedly CAS until lock is acquired.
 */
inline MinorCoroutine spinlock_vamana_node(const s_ptr<VamanaNode>& node,
                                           const u_ptr<ComputeThread>& thread) {
    bool success;
    u64 original_header;
    u64 attempts = 0;

    do {
        std::tie(success, original_header) =
            co_await try_lock_vamana_node(node->rptr, node->header(), thread);
        if (auto* sample = thread->current_breakdown_sample()) {
            ++sample->lock_attempts;
            if (!success) {
                ++sample->lock_retries;
                ++sample->cas_failures;
            }
        }
        node->header() = original_header;
        ++attempts;
        if (!success && attempts % 100000 == 0) {
            std::cerr << "[spinlock_vamana_node] thread=" << thread->get_id()
                      << " attempts=" << attempts
                      << " node_id=" << node->id()
                      << " rptr=" << node->rptr
                      << " header=0x" << std::hex << original_header << std::dec
                      << std::endl;
        }
    } while (!success);

    node->set_lock();
}

/**
 * Allocate a VamanaNode on a random memory node via FAA on free_ptr.
 */
inline auto allocate_vamana_node(const u_ptr<ComputeThread>& thread) {
    const u32 memory_node = thread->get_random_memory_node();
    size_t node_size = VamanaNode::total_size();

    // Ensure 8B alignment for CAS on header
    while (node_size % 8 != 0) {
        node_size += 4;
    }

    thread->stats.allocation_size += node_size;
    ++thread->stats.remote_allocations;
    thread->track_post();

    const QP& qp = thread->ctx->qps[memory_node]->qp;
    qp->post_FAA(reinterpret_cast<u64>(thread->coros_pointer_slot()),
                 thread->ctx->get_lkey(),
                 thread->ctx->get_remote_mrt(memory_node),
                 0,  // free_ptr at offset 0
                 node_size,
                 true,
                 thread->create_wr_id());

    struct awaitable {
        const u32 memory_node;
        const size_t node_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}

        RemotePtr await_resume() const {
            const RemotePtr rptr{memory_node, *thread->coros_pointer_slot()};
            lib_assert(rptr.byte_offset() + node_size <= MEMORY_NODE_MAX_MEMORY,
                       "memory node " << rptr.memory_node() << " out of memory");
            return rptr;
        }
    };

    return awaitable{memory_node, node_size, thread};
}

/**
 * Swap medoid pointer atomically via CAS.
 */
inline auto swap_medoid_ptr(const RemotePtr& old_ptr, const RemotePtr& new_ptr,
                            const u_ptr<ComputeThread>& thread) {
    thread->track_post();

    const QP& qp = thread->ctx->qps[0]->qp;
    qp->post_CAS(reinterpret_cast<u64>(thread->coros_pointer_slot()),
                 thread->ctx->get_lkey(),
                 thread->ctx->get_remote_mrt(0),
                 8,  // medoid_ptr at offset 8
                 old_ptr.raw_address,
                 new_ptr.raw_address,
                 true,
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
