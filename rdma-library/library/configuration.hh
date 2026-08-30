#ifndef RDMA_LIBRARY_CONFIGURATION_HH
#define RDMA_LIBRARY_CONFIGURATION_HH

#include <boost/program_options.hpp>

#include "types.hh"

namespace configuration {

namespace po = boost::program_options;

class Configuration {
public:
  i32 max_send_queue_wr{1024};
  i32 max_recv_queue_wr{1024};
  i32 max_poll_cqes{16};
  u32 port{1234};
  u32 qp_handshake_timeout_ms{10000};
  str ib_device;
  u32 device_port{1};
  bool is_server{false};
  vec<str> server_nodes;
  vec<str> client_nodes;
  u32 num_clients{1};
  bool is_initiator{false};

protected:
  po::options_description desc{"Allowed options"};

public:
  Configuration();
  Configuration(int argc, char** argv);

  u32 num_server_nodes() const { return server_nodes.size(); }
  u32 num_client_nodes() const { return client_nodes.size(); }
  bool rdma_limits_valid() const {
    return num_clients > 0 && port > 0 && port <= 65535 &&
      qp_handshake_timeout_ms > 0 && qp_handshake_timeout_ms <= 300000 &&
      device_port > 0 && device_port <= 255 && max_poll_cqes > 0 &&
      max_send_queue_wr > 0 && max_recv_queue_wr > 0 &&
      max_poll_cqes <= max_send_queue_wr &&
      max_poll_cqes <= max_recv_queue_wr;
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const Configuration& config);

protected:
  static void exit_with_help_message(char** argv);
  void process_program_options(int argc, char** argv);

private:
  void create_rdma_options();
};

}  // namespace configuration

#endif  // RDMA_LIBRARY_CONFIGURATION_HH
