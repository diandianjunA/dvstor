#ifndef RDMA_LIBRARY_DETACHED_QP_HH
#define RDMA_LIBRARY_DETACHED_QP_HH

#include <iostream>

#include "context.hh"
#include "queue_pair.hh"
#include "types.hh"
#include "utils.hh"

class DetachedQP {
public:
  // ctor with delegated completion queues
  DetachedQP(Context& context, ibv_cq* send_cq, ibv_cq* recv_cq)
      : send_cq_(send_cq), recv_cq_(recv_cq), owns_cqs_(false) {
    qp = std::make_unique<QueuePair>(&context, send_cq, recv_cq);
  }

  // ctor with own completion queues
  explicit DetachedQP(Context& context) : owns_cqs_(true) {
    try {
      send_cq_ = ibv_create_cq(context.get_raw_context(),
                               context.get_config().max_send_queue_wr,
                               nullptr, nullptr, 0);
      recv_cq_ = ibv_create_cq(context.get_raw_context(),
                               context.get_config().max_recv_queue_wr,
                               nullptr, nullptr, 0);
      lib_assert(send_cq_ && recv_cq_, "Cannot create completion queues");

      qp = std::make_unique<QueuePair>(&context, send_cq_, recv_cq_);
    } catch (...) {
      qp.reset();
      if (recv_cq_ != nullptr) (void)ibv_destroy_cq(recv_cq_);
      if (send_cq_ != nullptr) (void)ibv_destroy_cq(send_cq_);
      recv_cq_ = nullptr;
      send_cq_ = nullptr;
      throw;
    }
  }

  ~DetachedQP() {
    if (owns_cqs_) {
      qp.reset();  // destroy qp first
      lib_assert(ibv_destroy_cq(recv_cq_) == 0,
                 "Cannot destroy receive completion queue");
      lib_assert(ibv_destroy_cq(send_cq_) == 0,
                 "Cannot destroy send completion queue");
    }
  }

  DetachedQP(const DetachedQP&) = delete;
  DetachedQP& operator=(const DetachedQP&) = delete;

  i32 poll_send_cq(ibv_wc* work_completion, const i32 max_cqes) const {
    return Context::poll_send_cq(work_completion, max_cqes, send_cq_);
  }

  i32 poll_recv_cq(ibv_wc* work_completion,
                   i32 max_cqes,
                   ReceiveInfo* recv_info = nullptr) const {
    return Context::poll_recv_cq(
      work_completion, max_cqes, recv_cq_, recv_info);
  }

  // channel_context is the context we use for communication
  // (the context to which other_qp belongs to)
  void connect(Context& channel_context, u16 lid, QP& other_qp) const {
    QPInfo send_buffer{lid,
                       qp->get_qp_num(),
                       0,
                       qp->active_mtu(),
                       qp->max_qp_read_atomic(),
                       qp->max_qp_dest_read_atomic()},
      receive_buffer{};
    LocalMemoryRegion region{channel_context, &receive_buffer, sizeof(QPInfo)};

    // other_qp is the qp we use to exchange information
    other_qp->post_receive(region);
    other_qp->post_send_inlined(&send_buffer, sizeof(QPInfo), IBV_WR_SEND);

    channel_context.poll_send_cq_until_completion();
    const ReceiveInfo receive_info = channel_context.receive();
    lib_assert(receive_info.bytes_written == sizeof(QPInfo),
               "Detached QPInfo receive length mismatch");
    lib_assert(receive_buffer.wire_valid(),
               "Detached QP received invalid or incompatible QPInfo");

    std::cerr << "pairing: " << qp->get_qp_num() << " -- "
              << receive_buffer.qp_number << std::endl;

    qp->transition_to_rtr(receive_buffer);
    qp->transition_to_rts(receive_buffer);
  }

public:
  QP qp;

private:
  ibv_cq* send_cq_{nullptr};
  ibv_cq* recv_cq_{nullptr};
  bool owns_cqs_;
};

#endif  // RDMA_LIBRARY_DETACHED_QP_HH
