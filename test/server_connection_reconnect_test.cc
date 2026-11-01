#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <thread>

#include "library/connection_manager.hh"
#include "library/context.hh"

namespace {

u32 available_port() {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  assert(fd >= 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  assert(bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  socklen_t bytes = sizeof(address);
  assert(getsockname(fd, reinterpret_cast<sockaddr*>(&address), &bytes) == 0);
  const u32 port = ntohs(address.sin_port);
  close(fd);
  return port;
}

bool rdma_device_available() {
  int count = 0;
  ibv_device** devices = ibv_get_device_list(&count);
  const bool available = devices != nullptr && count > 0;
  if (devices != nullptr) ibv_free_device_list(devices);
  return available;
}

}  // namespace

int main() {
  if (!rdma_device_available()) {
    std::cout << "SKIP: no RDMA device available\n";
    return 0;
  }

  configuration::Configuration server_config;
  server_config.num_clients = 2;
  server_config.port = available_port();
  Context server_context(server_config);
  ServerConnectionManager server(server_context, server_config);

  std::array<configuration::Configuration, 2> client_configs;
  std::array<std::unique_ptr<Context>, 2> client_contexts;
  for (u32 client = 0; client < 2; ++client) {
    client_configs[client].port = server_config.port;
    client_contexts[client] =
      std::make_unique<Context>(client_configs[client]);
  }
  QPs client_qps(2);

  std::thread initial_accept([&] { server.connect_to_clients(); });
  for (u32 client = 0; client < 2; ++client) {
    client_qps[client] = client_contexts[client]->connect_to_server(
      "127.0.0.1", server_config.port, client);
  }
  initial_accept.join();
  assert(server.initiator_qp == server.client_qps.front().get());

  std::atomic<bool> shutdown{false};
  std::thread reconnect_accept([&] { server.accept_reconnection(shutdown); });
  for (u32 client = 0; client < 2; ++client) {
    client_qps[client] = client_contexts[client]->connect_to_server(
      "127.0.0.1", server_config.port, client);
  }
  reconnect_accept.join();
  assert(shutdown.load(std::memory_order_acquire));
  for (const QP& qp : server.pending_reconnected_qps) {
    assert(qp != nullptr);
  }

  server.reset_for_reconnect();
  assert(server.initiator_qp == nullptr);
  server.connect_to_clients();
  assert(server.initiator_qp == server.client_qps.front().get());
  for (u32 client = 0; client < 2; ++client) {
    assert(server.client_qps[client] != nullptr);
    assert(server.pending_reconnected_qps[client] == nullptr);
  }
  return 0;
}
