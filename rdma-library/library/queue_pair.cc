#include "queue_pair.hh"

#include <limits>

#include "utils.hh"

// delegating ctor
QueuePair::QueuePair(Context* context, bool use_shared_receive_cq)
    : QueuePair(context,
                context->get_send_cq(),
                context->get_receive_cq(),
                use_shared_receive_cq) {}

QueuePair::QueuePair(Context* context,
                     ibv_cq* send_cq,
                     ibv_cq* recv_cq,
                     bool use_shared_receive_cq)
    : context_(context),
      lid_(context->get_lid()),
      use_shared_receive_cq_(use_shared_receive_cq) {
  ibv_qp_init_attr init_attributes =
    get_qp_initial_attributes(send_cq, recv_cq);
  queue_pair_ =
    ibv_create_qp(context->get_protection_domain(), &init_attributes);
  try {
    lib_assert(queue_pair_, "Cannot create queue pair");
    max_send_wr_ = init_attributes.cap.max_send_wr;
    transition_to_init();
  } catch (...) {
    if (queue_pair_ != nullptr) {
      (void)ibv_destroy_qp(queue_pair_);
      queue_pair_ = nullptr;
    }
    throw;
  }
}

enum ibv_mtu QueuePair::active_mtu() const {
  return context_->get_active_mtu();
}

u8 QueuePair::max_qp_read_atomic() const {
  return static_cast<u8>(context_->max_qp_read_atomic());
}

u8 QueuePair::max_qp_dest_read_atomic() const {
  return static_cast<u8>(context_->max_qp_dest_read_atomic());
}

QueuePair::~QueuePair() {
  lib_assert(ibv_destroy_qp(queue_pair_) == 0, "Cannot destroy queue pair.");
}

ibv_qp_init_attr QueuePair::get_qp_initial_attributes(ibv_cq* send_cq,
                                                      ibv_cq* recv_cq) {
  ibv_qp_init_attr attributes{};
  const i32 max_sge_elements = 1;

  // FYI: if a shared rcq is used, no normal receive request RR can be posted
  if (use_shared_receive_cq_) {
    attributes.srq = context_->get_shared_receive_cq();
  }

  attributes.send_cq = send_cq;
  attributes.recv_cq = recv_cq;
  attributes.cap.max_send_wr = context_->get_config().max_send_queue_wr;
  attributes.cap.max_send_sge = max_sge_elements;
  attributes.cap.max_recv_wr = context_->get_config().max_recv_queue_wr;
  attributes.cap.max_recv_sge = max_sge_elements;
  attributes.cap.max_inline_data = INLINE_SIZE;
  attributes.qp_type = IBV_QPT_RC;
  // if 1, all WRs will generate CQEs, if 0, only flagged WRs generate CQEs
  attributes.sq_sig_all = 0;

  return attributes;
}

// transition state of queue pair from RESET to INIT:
// basic information set, ready for posting to receive queue.
void QueuePair::transition_to_init() {
  ibv_qp_attr attributes{};

  attributes.qp_state = IBV_QPS_INIT;
  attributes.pkey_index = 0;
  attributes.port_num = context_->get_config().device_port;
  attributes.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_ATOMIC;
  lib_assert(ibv_modify_qp(queue_pair_,
                           &attributes,
                           IBV_QP_STATE | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS |
                             IBV_QP_PKEY_INDEX) == 0,
             "Cannot change state of queue pair to INIT");
  lib_debug("Transitioned state to INIT successfully");
}

void QueuePair::transition_to_rtr(const QPInfo& remote_buffer) {
  lib_assert(remote_buffer.wire_valid(),
             "Invalid or incompatible remote QPInfo wire record");
  lib_assert(QPInfo::mtu_valid(static_cast<u8>(context_->get_active_mtu())),
             "Local RDMA port reports an invalid active MTU");
  ibv_qp_attr attributes{};

  attributes.qp_state = IBV_QPS_RTR;
  attributes.path_mtu = remote_buffer.negotiated_mtu(
    context_->get_active_mtu());
  attributes.dest_qp_num = remote_buffer.qp_number;
  attributes.rq_psn = 0;
  attributes.max_dest_rd_atomic = remote_buffer.negotiated_max_qp_rd_atom(
    max_qp_dest_read_atomic());
  attributes.min_rnr_timer = 12;
  attributes.ah_attr.is_global = 0;
  attributes.ah_attr.dlid = remote_buffer.lid;
  attributes.ah_attr.sl = 0;
  attributes.ah_attr.src_path_bits = 0;
  attributes.ah_attr.port_num = context_->get_config().device_port;

  lib_assert(
    ibv_modify_qp(queue_pair_,
                  &attributes,
                  IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                    IBV_QP_RQ_PSN | IBV_QP_MIN_RNR_TIMER |
                    IBV_QP_MAX_DEST_RD_ATOMIC) == 0,
    "Cannot change state of queue pair to RTR");
  lib_debug("Transitioned state to RTR successfully");
}

void QueuePair::transition_to_rts(const QPInfo& remote_buffer) {
  lib_assert(remote_buffer.wire_valid(),
             "Invalid or incompatible remote QPInfo wire record");
  ibv_qp_attr attributes{};

  attributes.qp_state = IBV_QPS_RTS;
  attributes.timeout = 14;
  attributes.retry_cnt = 7;
  attributes.rnr_retry = 7;
  attributes.sq_psn = 0;
  attributes.max_rd_atomic = remote_buffer.negotiated_max_qp_init_rd_atom(
    max_qp_read_atomic());

  lib_assert(ibv_modify_qp(queue_pair_,
                           &attributes,
                           IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                             IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                             IBV_QP_MAX_QP_RD_ATOMIC) == 0,
             "Cannot change state of queue pair to RTS");
  lib_debug("Transitioned state to RTS successfully");
}

void QueuePair::post_receive(MemoryRegion& region) {
  lib_assert(region.get_size_in_bytes() <=
               std::numeric_limits<u32>::max(),
             "Receive region exceeds the verbs 32-bit SGE length");
  post_receive(region, static_cast<u32>(region.get_size_in_bytes()));
}

void QueuePair::post_receive(MemoryRegion& region,
                             u32 size_in_bytes,
                             u64 wr_id,
                             u64 local_offset) {
  lib_assert(QueuePairRequestContract::local_range_valid(
               static_cast<u64>(region.get_size_in_bytes()),
               local_offset,
               size_in_bytes),
             "Receive request exceeds the local memory region or is empty");
  ibv_recv_wr work_request{};
  ibv_sge scatter_gather_entry{};

  // points to the RR that failed to be posted (if not successful)
  ibv_recv_wr* bad_work_request{nullptr};

  scatter_gather_entry.addr = region.get_address() + local_offset;
  scatter_gather_entry.length = size_in_bytes;
  scatter_gather_entry.lkey = region.get_lkey();

  work_request.wr_id = wr_id;
  work_request.next = nullptr;
  work_request.sg_list = &scatter_gather_entry;
  work_request.num_sge = 1;

  // post receive request to receive queue
  lib_assert(ibv_post_recv(queue_pair_, &work_request, &bad_work_request) == 0,
             "Cannot post receive request");
  lib_debug("Receive request successfully posted");
}

u32 QueuePair::receive_u32(Context& context) {
  u32 value;

  LocalMemoryRegion region{context, std::addressof(value), sizeof(u32)};
  post_receive(region);
  context.receive();

  return value;
}

void QueuePair::post_send_inlined(const void* address,
                                  u32 size_in_bytes,
                                  enum ibv_wr_opcode opcode,
                                  bool signaled,
                                  MemoryRegionToken* token,
                                  u64 remote_offset,
                                  u64 wr_id) {
  post_send(reinterpret_cast<u64>(address),
            size_in_bytes,
            0,
            opcode,
            signaled,
            true,
            token,
            remote_offset,
            0,
            wr_id);
}

void QueuePair::post_send_u32(u32& value, bool signaled) {
  post_send(reinterpret_cast<u64>(std::addressof(value)),
            sizeof(u32),
            0,
            IBV_WR_SEND,
            signaled,
            true,
            nullptr,
            0,
            0,
            0);
}

void QueuePair::post_send(MemoryRegion& region,
                          enum ibv_wr_opcode opcode,
                          bool signaled,
                          MemoryRegionToken* token,
                          u64 remote_offset,
                          u64 local_offset) {
  lib_assert(local_offset == 0,
             "Whole-region send cannot use a non-zero local offset");
  lib_assert(region.get_size_in_bytes() <= MESSAGE_SIZE,
             "Whole-region send exceeds the verbs message limit");
  post_send(region.get_address(),
            static_cast<u32>(region.get_size_in_bytes()),
            region.get_lkey(),
            opcode,
            signaled,
            false,
            token,
            remote_offset,
            local_offset,
            0);
}

void QueuePair::post_send(MemoryRegion& region,
                          u32 size_in_bytes,
                          enum ibv_wr_opcode opcode,
                          bool signaled,
                          MemoryRegionToken* token,
                          u64 remote_offset,
                          u64 local_offset) {
  lib_assert(QueuePairRequestContract::local_range_valid(
               static_cast<u64>(region.get_size_in_bytes()),
               local_offset,
               size_in_bytes),
             "Send request exceeds the local memory region or is empty");
  post_send(region.get_address(),
            size_in_bytes,
            region.get_lkey(),
            opcode,
            signaled,
            false,
            token,
            remote_offset,
            local_offset,
            0);
}

void QueuePair::post_send_with_id(MemoryRegion& region,
                                  u32 size_in_bytes,
                                  enum ibv_wr_opcode opcode,
                                  u64 wr_id,
                                  bool signaled,
                                  MemoryRegionToken* token,
                                  u64 remote_offset,
                                  u64 local_offset) {
  lib_assert(QueuePairRequestContract::local_range_valid(
               static_cast<u64>(region.get_size_in_bytes()),
               local_offset,
               size_in_bytes),
             "Send request exceeds the local memory region or is empty");
  post_send(region.get_address(),
            size_in_bytes,
            region.get_lkey(),
            opcode,
            signaled,
            false,
            token,
            remote_offset,
            local_offset,
            wr_id);
}

void QueuePair::post_send(u64 address,
                          u32 size,
                          u32 lkey,
                          enum ibv_wr_opcode opcode,
                          bool signaled,
                          bool inlined,
                          MemoryRegionToken* token,
                          u64 remote_offset,
                          u64 local_offset,
                          u64 wr_id) {
  lib_assert(QueuePairRequestContract::opcode_supported(opcode),
             "Unsupported QueuePair message opcode");
  lib_assert(QueuePairRequestContract::address_range_valid(
               address, local_offset, size),
             "Send request has an invalid or overflowing local address range");
  lib_assert(!inlined ||
               QueuePairRequestContract::inline_opcode_supported(opcode),
             "RDMA reads and unsupported opcodes cannot be inlined");
  lib_assert(!inlined || size <= INLINE_SIZE, "Request cannot be inlined");
  lib_assert(size <= MESSAGE_SIZE, "Message size too large");

  ibv_send_wr work_request{};
  ibv_sge scatter_gather_entry{};

  // points to the SR that failed to be posted (if not successful)
  struct ibv_send_wr* bad_work_request{nullptr};

  scatter_gather_entry.addr = address + local_offset;
  scatter_gather_entry.length = size;
  scatter_gather_entry.lkey = lkey;

  work_request.opcode = opcode;
  work_request.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  work_request.send_flags |= inlined ? IBV_SEND_INLINE : 0;
  work_request.wr_id = wr_id;
  work_request.next = nullptr;
  work_request.sg_list = &scatter_gather_entry;
  work_request.num_sge = 1;

  if (opcode != IBV_WR_SEND) {
    lib_assert(token, "MemoryRegionToken does not exist");
    lib_assert(token->contains(remote_offset, size),
               "RDMA request exceeds the remote memory region");
    work_request.wr.rdma.remote_addr = token->address + remote_offset;
    work_request.wr.rdma.rkey = token->rkey;
  }

  // post send request to send queue
  lib_assert(ibv_post_send(queue_pair_, &work_request, &bad_work_request) == 0,
             "Cannot post send request");

  switch (opcode) {
  case IBV_WR_SEND:
    lib_debug("SEND request successfully posted");
    break;
  case IBV_WR_RDMA_READ:
    lib_debug("RDMA_READ request successfully posted");
    break;
  case IBV_WR_RDMA_WRITE:
    lib_debug("RDMA_WRITE request successfully posted");
    break;
  default:
    lib_failure("Unknown request posted");
    break;
  }
}
void QueuePair::post_CAS(MemoryRegion& local_region,
                         MemoryRegionToken* remote_token,
                         u64 remote_offset,
                         u64 compare_to,
                         u64 swap_with,
                         bool signaled,
                         u64 wr_id) {
  lib_assert(local_region.get_size_in_bytes() >= sizeof(u64),
             "CAS local memory region is smaller than 8 bytes");
  post_CAS(local_region.get_address(),
           local_region.get_lkey(),
           remote_token,
           remote_offset,
           compare_to,
           swap_with,
           signaled,
           wr_id);
}

void QueuePair::post_CAS(u64 laddr,
                         u32 lkey,
                         MemoryRegionToken* remote_token,
                         u64 remote_offset,
                         u64 compare_to,
                         u64 swap_with,
                         bool signaled,
                         u64 wr_id) {
  lib_assert(QueuePairRequestContract::atomic_address_valid(laddr),
             "CAS local address must be non-zero and 8B aligned");
  lib_assert(remote_offset % 8 == 0, "CAS offset must be 8B aligned");
  lib_assert(remote_token != nullptr &&
               remote_token->contains(remote_offset, sizeof(u64)),
             "CAS exceeds the remote memory region");
  lib_assert((remote_token->address + remote_offset) % 8 == 0,
             "CAS remote address must be 8B aligned");

  ibv_send_wr work_request{};
  ibv_sge sge{};

  struct ibv_send_wr* bad_work_request{nullptr};

  sge.addr = laddr;
  sge.length = 8;
  sge.lkey = lkey;

  work_request.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  work_request.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  work_request.wr_id = wr_id;
  work_request.next = nullptr;
  work_request.sg_list = &sge;
  work_request.num_sge = 1;

  auto& atomic = work_request.wr.atomic;
  atomic.remote_addr = remote_token->address + remote_offset;
  atomic.rkey = remote_token->rkey;

  atomic.compare_add = compare_to;
  atomic.swap = swap_with;

  lib_assert(ibv_post_send(queue_pair_, &work_request, &bad_work_request) == 0,
             "Cannot post CAS request");
}

void QueuePair::post_FAA(u64 laddress,
                         u32 lkey,
                         MemoryRegionToken* remote_token,
                         u64 remote_offset,
                         u64 to_add,
                         bool signaled,
                         u64 wr_id) {
  lib_assert(QueuePairRequestContract::atomic_address_valid(laddress),
             "FAA local address must be non-zero and 8B aligned");
  lib_assert(remote_offset % 8 == 0, "FAA offset must be 8B aligned");
  lib_assert(remote_token != nullptr &&
               remote_token->contains(remote_offset, sizeof(u64)),
             "FAA exceeds the remote memory region");
  lib_assert((remote_token->address + remote_offset) % 8 == 0,
             "FAA remote address must be 8B aligned");
  ibv_send_wr work_request{};
  ibv_sge sge{};

  struct ibv_send_wr* bad_work_request{nullptr};

  sge.addr = laddress;
  sge.length = 8;
  sge.lkey = lkey;

  work_request.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  work_request.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  work_request.wr_id = wr_id;
  work_request.next = nullptr;
  work_request.sg_list = &sge;
  work_request.num_sge = 1;

  auto& atomic = work_request.wr.atomic;
  atomic.remote_addr = remote_token->address + remote_offset;
  atomic.rkey = remote_token->rkey;

  atomic.compare_add = to_add;

  lib_assert(ibv_post_send(queue_pair_, &work_request, &bad_work_request) == 0,
             "Cannot post FAA request");
}
