#include "context.hh"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <iostream>
#include <limits>
#include <sstream>
#include <thread>

#include "queue_pair.hh"
#include "utils.hh"

namespace {

str describe_wc_failure(const char* operation, const ibv_wc& wc) {
  std::ostringstream out;
  out << operation << " request failed: status=" << wc.status << " ("
      << ibv_wc_status_str(wc.status) << "), opcode=" << wc.opcode
      << ", vendor_err=" << wc.vendor_err << ", wr_id=" << wc.wr_id;
  return out.str();
}

bool send_exact(i32 socket_fd, const void* data, size_t bytes) {
  const auto* cursor = static_cast<const byte_t*>(data);
  size_t sent = 0;
  while (sent < bytes) {
    const ssize_t result = send(
      socket_fd, cursor + sent, bytes - sent, MSG_NOSIGNAL);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) return false;
    sent += static_cast<size_t>(result);
  }
  return true;
}

bool receive_exact(i32 socket_fd, void* data, size_t bytes) {
  auto* cursor = static_cast<byte_t*>(data);
  size_t received = 0;
  while (received < bytes) {
    const ssize_t result = recv(
      socket_fd, cursor + received, bytes - received, 0);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) return false;
    received += static_cast<size_t>(result);
  }
  return true;
}

void configure_qp_handshake_socket(i32 socket_fd, u32 timeout_ms) {
  const timeval timeout{
    static_cast<time_t>(timeout_ms / 1000),
    static_cast<suseconds_t>((timeout_ms % 1000) * 1000)};
  lib_assert(setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO,
                        &timeout, sizeof(timeout)) == 0,
             "Cannot configure QP handshake receive timeout");
  lib_assert(setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO,
                        &timeout, sizeof(timeout)) == 0,
             "Cannot configure QP handshake send timeout");
}

}  // namespace

Context::Context(Configuration& config,
                 const i32 device_idx,
                 bool create_shared_rcq)
    : config_(config) {
  lib_assert(config_.rdma_limits_valid(),
             "Invalid RDMA queue, polling, port, or client-count limits");
  i32 num_devices = 0;
  IBDeviceList device_list = ibv_get_device_list(&num_devices);
  try {
    lib_assert(num_devices > 0, "No InfiniBand devices found");
    lib_assert(device_list != nullptr, "Device list is null");
    if (!config_.ib_device.empty()) {
      for (i32 i = 0; i < num_devices; ++i) {
        if (config_.ib_device == ibv_get_device_name(device_list[i])) {
          device_ = device_list[i];
          break;
        }
      }
      lib_assert(device_ != nullptr,
                 "RDMA device " + config_.ib_device + " not found");
    } else {
      lib_assert(0 <= device_idx && device_idx < num_devices,
                 "Device " + std::to_string(device_idx) + " not found");
      device_ = device_list[device_idx];
    }

    std::cerr << num_devices << " device(s) found" << std::endl;
    std::cerr << "Selected device: " << ibv_get_device_name(device_)
              << std::endl;

    context_ = ibv_open_device(device_);
    lib_assert(device_ && context_, "Cannot open device");
    lib_assert(ibv_query_device(context_, &device_attributes_) == 0,
               "Cannot query RDMA device capabilities");
    lib_assert(
      device_attributes_.max_qp_init_rd_atom > 0 &&
        device_attributes_.max_qp_rd_atom > 0,
      "DVSTOR requires RDMA read/atomic initiator and responder resources");

    // allocate protection domain
    protection_domain_ = ibv_alloc_pd(context_);
    lib_assert(protection_domain_ != nullptr,
               "Cannot allocate RDMA protection domain");

    // query port
    lib_assert(
      ibv_query_port(context_, config_.device_port, &port_attributes_) == 0,
      "Cannot query port " + std::to_string(config_.device_port));
    lib_assert(port_attributes_.state == IBV_PORT_ACTIVE,
               "Selected RDMA port is not active");
    lib_assert(
      port_attributes_.lid != 0,
      "DVSTOR requires an InfiniBand LID; RoCE/GRH addressing is not supported");
    lib_assert(QPInfo::mtu_valid(static_cast<u8>(port_attributes_.active_mtu)),
               "RDMA port reports an invalid active MTU");
    std::cerr << "Selected port state: " << port_attributes_.state
              << ", lid: " << port_attributes_.lid
              << ", active_mtu: "
              << static_cast<u32>(port_attributes_.active_mtu) << std::endl;

    // create completion queues
    send_cq_ =
      ibv_create_cq(context_, config_.max_send_queue_wr, nullptr, nullptr, 0);
    receive_cq_ =
      ibv_create_cq(context_, config_.max_recv_queue_wr, nullptr, nullptr, 0);

    lib_assert(send_cq_ && receive_cq_, "Cannot create completion queues");

    if (create_shared_rcq) {
      ibv_srq_init_attr attributes{};
      attributes.srq_context = context_;
      attributes.attr.max_wr = config_.max_recv_queue_wr;
      attributes.attr.max_sge = 1;
      shared_receive_cq_ = ibv_create_srq(protection_domain_, &attributes);

      lib_assert(shared_receive_cq_,
                 "Cannot create shared receive completion queue");
    }

    ibv_free_device_list(device_list);
    device_list = nullptr;
  } catch (...) {
    if (shared_receive_cq_ != nullptr) {
      (void)ibv_destroy_srq(shared_receive_cq_);
    }
    if (receive_cq_ != nullptr) (void)ibv_destroy_cq(receive_cq_);
    if (send_cq_ != nullptr) (void)ibv_destroy_cq(send_cq_);
    if (protection_domain_ != nullptr) {
      (void)ibv_dealloc_pd(protection_domain_);
    }
    if (context_ != nullptr) (void)ibv_close_device(context_);
    if (device_list != nullptr) ibv_free_device_list(device_list);
    shared_receive_cq_ = nullptr;
    receive_cq_ = nullptr;
    send_cq_ = nullptr;
    protection_domain_ = nullptr;
    context_ = nullptr;
    throw;
  }
}

Context::~Context() {
  lib_assert(!shared_receive_cq_ || ibv_destroy_srq(shared_receive_cq_) == 0,
             "Cannot destroy shared receive completion queue");
  lib_assert(ibv_destroy_cq(receive_cq_) == 0,
             "Cannot destroy receive completion queue");
  lib_assert(ibv_destroy_cq(send_cq_) == 0,
             "Cannot destroy send completion queue");
  lib_assert(ibv_dealloc_pd(protection_domain_) == 0,
             "Cannot deallocate protection domain");
  lib_assert(ibv_close_device(context_) == 0, "Cannot close device.");

  close_server_socket();
}

void Context::bind_to_port(u32 tcp_port) {
  lib_assert(server_socket_ < 0, "Server socket is already bound");
  lib_assert(tcp_port > 0 && tcp_port <= 65535,
             "TCP listen port is out of range");
  server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  lib_assert(server_socket_ >= 0, "Cannot open socket.");

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(tcp_port);

  // activate reuse address option
  i32 option_val = 1;
  lib_assert(setsockopt(server_socket_,
                        SOL_SOCKET,
                        SO_REUSEADDR,
                        &option_val,
                        sizeof(option_val)) == 0,
             "Cannot set socket option to reuse address");

  lib_assert(
    bind(server_socket_, (sockaddr*)&address, sizeof(sockaddr_in)) == 0,
    "Cannot bind to port " + std::to_string(tcp_port));

  lib_assert(listen(server_socket_, 128) == 0, "Cannot listen on socket");
}

void Context::close_server_socket() {
  if (server_socket_ >= 0) {
    close(server_socket_);
    server_socket_ = -1;
  }
}

std::pair<QP, u32> Context::wait_for_connection() {
  QP queue_pair = std::make_unique<QueuePair>(this);

  QPInfo receive_buffer{},
    send_buffer{get_lid(),
                queue_pair->get_qp_num(),
                0,
                get_active_mtu(),
                static_cast<u8>(max_qp_read_atomic()),
                static_cast<u8>(max_qp_dest_read_atomic())};
  constexpr size_t qp_size = sizeof(QPInfo);

  i32 tcp_socket;
  do {
    tcp_socket = accept(server_socket_, nullptr, nullptr);
  } while (tcp_socket < 0 && errno == EINTR);
  lib_assert(tcp_socket >= 0, "Cannot accept TCP connection");
  configure_qp_handshake_socket(
    tcp_socket, config_.qp_handshake_timeout_ms);

  lib_debug("Exchange QP information with client");
  lib_assert(receive_exact(tcp_socket, &receive_buffer, qp_size),
             "Failed to receive complete QPInfo; peer may use an incompatible wire version");
  lib_assert(receive_buffer.wire_valid(),
             "Received invalid or incompatible QPInfo wire record");
  lib_assert(send_exact(tcp_socket, &send_buffer, qp_size),
             "Failed to transmit complete QPInfo wire record");

  std::cerr << "pairing: " << queue_pair->get_qp_num() << " -- "
            << receive_buffer.qp_number << std::endl;

  queue_pair->transition_to_rtr(receive_buffer);
  queue_pair->transition_to_rts(receive_buffer);

  // TODO: set remote user data

  close(tcp_socket);

  return {std::move(queue_pair), receive_buffer.node_id};
}

QP Context::connect_to_server(const str& address, u32 tcp_port, u32 node_id) {
  lib_assert(tcp_port > 0 && tcp_port <= 65535,
             "TCP connection port is out of range");
  QP queue_pair = std::make_unique<QueuePair>(this);

  QPInfo send_buffer{
    get_lid(),
    queue_pair->get_qp_num(),
    node_id,
    get_active_mtu(),
    static_cast<u8>(max_qp_read_atomic()),
    static_cast<u8>(max_qp_dest_read_atomic())},
    receive_buffer{};
  constexpr size_t qp_size = sizeof(QPInfo);

  sockaddr_in remote_address{};
  remote_address.sin_family = AF_INET;
  remote_address.sin_port = htons(tcp_port);
  lib_assert(inet_pton(AF_INET, address.c_str(),
                       &(remote_address.sin_addr)) == 1,
             "Invalid resolved IPv4 address: " + address);

  lib_debug("Connect to server with address " + address);
  i32 tcp_socket = -1;
  for (;;) {
    tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
    lib_assert(tcp_socket >= 0, "Cannot open socket.");
    if (connect(tcp_socket, reinterpret_cast<sockaddr*>(&remote_address),
                sizeof(remote_address)) == 0) {
      break;
    }
    const int connect_error = errno;
    close(tcp_socket);
    tcp_socket = -1;
    lib_assert(connect_error == EINTR || connect_error == ECONNREFUSED ||
                 connect_error == ETIMEDOUT || connect_error == EHOSTUNREACH ||
                 connect_error == ENETUNREACH || connect_error == EAGAIN,
               "Cannot connect to " + address + ": " +
                 std::strerror(connect_error));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  configure_qp_handshake_socket(
    tcp_socket, config_.qp_handshake_timeout_ms);
  lib_debug("Exchange QP information with server");
  lib_assert(send_exact(tcp_socket, &send_buffer, qp_size),
             "Failed to transmit complete QPInfo wire record");
  lib_assert(receive_exact(tcp_socket, &receive_buffer, qp_size),
             "Failed to receive complete QPInfo; peer may use an incompatible wire version");
  lib_assert(receive_buffer.wire_valid(),
             "Received invalid or incompatible QPInfo wire record");

  std::cerr << "pairing: " << queue_pair->get_qp_num() << " -- "
            << receive_buffer.qp_number << std::endl;

  queue_pair->transition_to_rtr(receive_buffer);
  queue_pair->transition_to_rts(receive_buffer);
  close(tcp_socket);

  return queue_pair;
}

void Context::post_shared_receive(MemoryRegion& region) {
  ibv_recv_wr work_request{};
  ibv_sge scatter_gather_entry{};
  ibv_recv_wr* bad_work_request{nullptr};

  lib_assert(shared_receive_cq_, "No shared receive CQ exists");
  lib_assert(region.get_size_in_bytes() > 0 &&
               region.get_size_in_bytes() <=
                 std::numeric_limits<u32>::max(),
             "Shared receive region exceeds the verbs 32-bit SGE length");

  scatter_gather_entry.addr = region.get_address();
  scatter_gather_entry.length =
    static_cast<u32>(region.get_size_in_bytes());
  scatter_gather_entry.lkey = region.get_lkey();

  work_request.wr_id = reinterpret_cast<u64>(&region);
  work_request.next = nullptr;
  work_request.sg_list = &scatter_gather_entry;
  work_request.num_sge = 1;

  lib_assert(ibv_post_srq_recv(
               get_shared_receive_cq(), &work_request, &bad_work_request) == 0,
             "Cannot post shared receive request");
  lib_debug("Shared receive request successfully posted");
}

// static function
i32 Context::poll_recv_cq(ibv_wc* work_completion,
                          const i32 max_cqes,
                          ibv_cq* recv_cq,
                          ReceiveInfo* recv_info) {
  lib_assert(max_cqes > 0, "CQ receive poll count must be positive");
  lib_assert(work_completion != nullptr,
             "CQ receive completion buffer must not be null");
  lib_assert(recv_cq != nullptr, "Receive completion queue must not be null");
  // work_completion and recv_info must be arrays of size max_cqes.
  i32 num_entries = ibv_poll_cq(recv_cq, max_cqes, work_completion);

  if (num_entries > 0) {
    // verify completion status
    for (i32 i = 0; i < num_entries; ++i) {
      lib_assert(work_completion[i].status == IBV_WC_SUCCESS,
                 describe_wc_failure("Receive", work_completion[i]));
      lib_debug("Receive request completed");

      if (recv_info && work_completion[i].opcode == IBV_WC_RECV) {
        recv_info[i].mr =
          reinterpret_cast<MemoryRegion*>(work_completion[i].wr_id);
        recv_info[i].bytes_written = work_completion[i].byte_len;
      }
    }

  } else if (num_entries < 0) {
    lib_failure("Cannot poll receive completion queue");
  }

  return num_entries;
}

i32 Context::poll_recv_cq(ibv_wc* work_completion,
                          const i32 max_cqes,
                          ReceiveInfo* recv_info) {
  lib_assert(max_cqes > 0 && max_cqes <= config_.max_recv_queue_wr,
             "receive CQ poll count is outside the configured queue depth");
  const i32 batch = CompletionPollContract::batch_size(
    max_cqes, config_.max_poll_cqes);
  return poll_recv_cq(work_completion, batch, receive_cq_, recv_info);
}

ReceiveInfo Context::receive() {
  ibv_wc work_completion{};
  ReceiveInfo recv_info{};
  i32 num_entries;

  do {
    num_entries = poll_recv_cq(&work_completion, 1, &recv_info);
  } while (num_entries == 0);

  return recv_info;
}

// receive exactly n completion events
void Context::receive(i32 n) {
  lib_assert(n >= 0, "receive completion count must not be negative");
  if (n == 0) return;

  const i32 capacity = CompletionPollContract::batch_size(
    n, config_.max_poll_cqes);
  vec<ibv_wc> work_completions(static_cast<size_t>(capacity));
  i32 num_entries = 0;
  while (num_entries < n) {
    const i32 batch = CompletionPollContract::batch_size(
      n - num_entries, capacity);
    num_entries += poll_recv_cq(work_completions.data(), batch);
  }
}

// static function
i32 Context::poll_send_cq(ibv_wc* work_completion,
                          const i32 max_cqes,
                          ibv_cq* send_cq,
                          const func<void(u64)>& id_handler) {
  lib_assert(max_cqes > 0, "CQ send poll count must be positive");
  lib_assert(work_completion != nullptr,
             "CQ send completion buffer must not be null");
  lib_assert(send_cq != nullptr, "Send completion queue must not be null");
  lib_assert(static_cast<bool>(id_handler),
             "Send completion handler must not be empty");
  // work_completion must be an array of size max_cqes.
  i32 num_entries = ibv_poll_cq(send_cq, max_cqes, work_completion);

  if (num_entries > 0) {
    // verify completion status
    for (i32 i = 0; i < num_entries; ++i) {
      lib_assert(work_completion[i].status == IBV_WC_SUCCESS,
                 describe_wc_failure("Send", work_completion[i]));

      id_handler(work_completion[i].wr_id);
    }
    lib_debug("Send request completed");

  } else if (num_entries < 0) {
    lib_failure("Cannot poll completion queue");
  }

  return num_entries;
}

// static function
i32 Context::poll_send_cq(ibv_wc* work_completion,
                          const i32 max_cqes,
                          ibv_cq* send_cq) {
  return poll_send_cq(work_completion, max_cqes, send_cq, [](u64) {});
}

i32 Context::poll_send_cq(ibv_wc* work_completion, const i32 max_cqes) {
  lib_assert(max_cqes > 0 && max_cqes <= config_.max_send_queue_wr,
             "send CQ poll count is outside the configured queue depth");
  const i32 batch = CompletionPollContract::batch_size(
    max_cqes, config_.max_poll_cqes);
  return poll_send_cq(work_completion, batch, send_cq_);
}

i32 Context::poll_send_cq_until_completion() {
  ibv_wc work_completion{};
  i32 num_entries;

  do {
    num_entries = poll_send_cq(&work_completion, 1);
  } while (num_entries == 0);

  return num_entries;
}

// poll completion until we get exactly n completion events
void Context::poll_send_cq_until_completion(i32 n) {
  lib_assert(n >= 0, "send completion count must not be negative");
  if (n == 0) return;

  const i32 capacity = CompletionPollContract::batch_size(
    n, config_.max_poll_cqes);
  vec<ibv_wc> work_completions(static_cast<size_t>(capacity));
  i32 num_entries = 0;
  while (num_entries < n) {
    const i32 batch = CompletionPollContract::batch_size(
      n - num_entries, capacity);
    num_entries += poll_send_cq(work_completions.data(), batch);
  }
}
