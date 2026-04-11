#pragma once

/**
 * RDMA write operations for the Vamana index.
 * Follows the same patterns as rdma_writes.hh for HNSW.
 */

#include "compute_thread.hh"
#include "coroutine.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_neighborlist.hh"
#include "vamana/vamana_node.hh"

namespace rdma::vamana {

inline void track_total_rdma_write(const u_ptr<ComputeThread>& thread, size_t bytes) {
    thread->stats.rdma_writes_in_bytes += bytes;
    if (thread->is_query_worker()) {
        thread->stats.query_rdma_writes_in_bytes += bytes;
    } else if (thread->is_insert_worker()) {
        thread->stats.build_rdma_writes_in_bytes += bytes;
    }
}

/**
 * Unlock a VamanaNode by writing 0 to the lock byte.
 */
inline auto unlock_vamana_node(const s_ptr<VamanaNode>& node, const u_ptr<ComputeThread>& thread) {
    byte_t unlock = 0;

    track_total_rdma_write(thread, 1);
    thread->track_post();

    const QP& qp = thread->ctx->qps[node->rptr.memory_node()]->qp;
    qp->post_send_inlined(std::addressof(unlock),
                          1,
                          IBV_WR_RDMA_WRITE,
                          true,
                          thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                          node->rptr.byte_offset() + VamanaNode::HEADER_UNTIL_LOCK,
                          thread->create_wr_id());

    node->reset_lock();
    return std::suspend_always{};
}

/**
 * Write a complete new VamanaNode to remote memory.
 * Writes: header + id + edge_count + pad + vector + rabitq + neighbors.
 */
inline auto write_vamana_node(const RemotePtr& rptr,
                              node_t id,
                              const span<element_t> components,
                              const byte_t* rabitq_data,
                              const span<RemotePtr> neighbors,
                              u8 edge_count,
                              bool is_medoid,
                              bool node_lock,
                              const u_ptr<ComputeThread>& thread) {
    byte_t* local_buffer = thread->buffer_allocator.allocate_vamana_node(thread->get_id());
    const size_t total = VamanaNode::total_size();

    // Build node in local buffer
    u64 header = 0;
    if (is_medoid) header |= VamanaNode::HEADER_IS_MEDOID;
    if (node_lock) header |= VamanaNode::HEADER_NODE_LOCK;

    byte_t* ptr = local_buffer;

    // Header (8B)
    *reinterpret_cast<u64*>(ptr) = header;
    ptr += VamanaNode::HEADER_SIZE;

    // ID (4B)
    *reinterpret_cast<u32*>(ptr) = id;
    ptr += sizeof(u32);

    // Edge count (1B)
    *reinterpret_cast<u8*>(ptr) = edge_count;
    ptr += sizeof(u8);

    // Padding (3B)
    std::memset(ptr, 0, VamanaNode::PADDING_SIZE);
    ptr += VamanaNode::PADDING_SIZE;

    // Vector (dim * 4B)
    std::memcpy(ptr, components.data(), VamanaNode::DIM * sizeof(element_t));
    ptr += VamanaNode::DIM * sizeof(element_t);

    // RaBitQ data
    std::memcpy(ptr, rabitq_data, VamanaNode::RABITQ_SIZE);
    ptr += VamanaNode::RABITQ_SIZE;

    // Neighbors (R * 8B) — write active + zero the rest
    for (u8 i = 0; i < edge_count && i < neighbors.size(); ++i) {
        reinterpret_cast<u64*>(ptr)[i] = neighbors[i].raw_address;
    }
    for (u32 i = edge_count; i < VamanaNode::R; ++i) {
        reinterpret_cast<u64*>(ptr)[i] = 0;
    }

    track_total_rdma_write(thread, total);
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_send(reinterpret_cast<u64>(local_buffer),
                  total,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_WRITE,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(rptr.memory_node()),
                  rptr.byte_offset(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        byte_t* local_buffer;
        size_t total_size;
        const RemotePtr& rptr;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}

        s_ptr<VamanaNode> await_resume() {
            return std::make_shared<VamanaNode>(local_buffer, total_size, rptr, thread.get());
        }
    };

    return awaitable{local_buffer, total, rptr, thread};
}

/**
 * Write the neighbor list of a VamanaNode (edge_count + neighbors).
 *
 * The node layout has vector and rabitq data BETWEEN the edge_count and
 * the neighbor slots, so we must write them at their actual offsets:
 *   - edge_count + padding (4B) at offset_edge_count()
 *   - neighbor slots (R*8B) at offset_neighbors()
 */
inline auto write_vamana_neighbors(const s_ptr<VamanaNode>& node,
                                   const span<RemotePtr> neighbors,
                                   u8 edge_count,
                                   const u_ptr<ComputeThread>& thread) {
    // Buffer 1: edge_count(1B) + padding(3B)
    const size_t meta_size = VamanaNode::EDGE_COUNT_SIZE + VamanaNode::PADDING_SIZE;
    byte_t* meta_buffer = thread->buffer_allocator.allocate_buffer(meta_size);
    *reinterpret_cast<u8*>(meta_buffer) = edge_count;
    std::memset(meta_buffer + sizeof(u8), 0, VamanaNode::PADDING_SIZE);

    // Buffer 2: neighbors(R*8B)
    byte_t* nbr_buffer = thread->buffer_allocator.allocate_buffer(VamanaNode::NEIGHBORS_SIZE);
    for (u8 i = 0; i < edge_count && i < neighbors.size(); ++i) {
        reinterpret_cast<u64*>(nbr_buffer)[i] = neighbors[i].raw_address;
    }
    for (u32 i = edge_count; i < VamanaNode::R; ++i) {
        reinterpret_cast<u64*>(nbr_buffer)[i] = 0;
    }

    track_total_rdma_write(thread, meta_size + VamanaNode::NEIGHBORS_SIZE);

    const QP& qp = thread->ctx->qps[node->rptr.memory_node()]->qp;

    // Write 1: edge_count + padding at offset_edge_count()
    thread->track_post();
    qp->post_send(reinterpret_cast<u64>(meta_buffer),
                  meta_size,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_WRITE,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                  node->rptr.byte_offset() + VamanaNode::offset_edge_count(),
                  0,
                  thread->create_wr_id());

    // Write 2: neighbor slots at offset_neighbors()
    thread->track_post();
    qp->post_send(reinterpret_cast<u64>(nbr_buffer),
                  VamanaNode::NEIGHBORS_SIZE,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_WRITE,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(node->rptr.memory_node()),
                  node->rptr.byte_offset() + VamanaNode::offset_neighbors(),
                  0,
                  thread->create_wr_id());

    struct awaitable {
        byte_t* meta_buffer;
        byte_t* nbr_buffer;
        size_t meta_size;
        size_t nbr_size;
        const u_ptr<ComputeThread>& thread;

        static bool await_ready() { return false; }
        static void await_suspend(std::coroutine_handle<>) {}
        void await_resume() {
            thread->buffer_allocator.free_buffer(meta_buffer, meta_size);
            thread->buffer_allocator.free_buffer(nbr_buffer, nbr_size);
        }
    };

    return awaitable{meta_buffer, nbr_buffer, meta_size, VamanaNode::NEIGHBORS_SIZE, thread};
}

/**
 * Write the medoid pointer to memory node 0 (at offset 8).
 */
inline auto write_medoid_ptr(const RemotePtr& medoid_ptr, const u_ptr<ComputeThread>& thread) {
    track_total_rdma_write(thread, RemotePtr::SIZE);
    thread->track_post();

    const QP& qp = thread->ctx->qps[0]->qp;
    qp->post_send_inlined(std::addressof(medoid_ptr.raw_address),
                          RemotePtr::SIZE,
                          IBV_WR_RDMA_WRITE,
                          true,
                          thread->ctx->get_remote_mrt(0),
                          8,  // medoid_ptr at offset 8
                          thread->create_wr_id());

    return std::suspend_always{};
}

/**
 * Write the header of a VamanaNode.
 */
inline auto write_vamana_header(const RemotePtr& rptr,
                                bool is_medoid,
                                bool medoid_lock,
                                bool node_lock,
                                const u_ptr<ComputeThread>& thread) {
    u64 header = 0;
    if (is_medoid) header |= VamanaNode::HEADER_IS_MEDOID;
    if (medoid_lock) header |= VamanaNode::HEADER_MEDOID_LOCK;
    if (node_lock) header |= VamanaNode::HEADER_NODE_LOCK;

    track_total_rdma_write(thread, VamanaNode::HEADER_SIZE);
    thread->track_post();

    const QP& qp = thread->ctx->qps[rptr.memory_node()]->qp;
    qp->post_send_inlined(std::addressof(header),
                          VamanaNode::HEADER_SIZE,
                          IBV_WR_RDMA_WRITE,
                          true,
                          thread->ctx->get_remote_mrt(rptr.memory_node()),
                          rptr.byte_offset(),
                          thread->create_wr_id());

    return std::suspend_always{};
}

}  // namespace rdma::vamana
