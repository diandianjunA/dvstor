#include <cassert>
#include <limits>

#include "library/configuration.hh"
#include "library/context.hh"
#include "library/memory_region.hh"
#include "library/queue_pair.hh"
#include "library/utils.hh"

namespace {

MemoryRegionToken valid_token() {
  return MemoryRegionToken{
    0x1000,
    0,
    0,
    16,
    MemoryRegionToken::kWireMagic,
    MemoryRegionToken::kWireVersion,
    MemoryRegionToken::kWireBytes};
}

void test_memory_region_token_wire_contract() {
  const MemoryRegionToken token = valid_token();
  assert(token.address_range_valid());
  assert(token.contains(0, 1));
  assert(token.contains(0, 16));
  assert(token.contains(15, 1));
  assert(!token.contains(0, 0));
  assert(!token.contains(16, 1));
  assert(!token.contains(std::numeric_limits<u64>::max(), 1));

  MemoryRegionToken boundary = token;
  boundary.address = std::numeric_limits<u64>::max() - 7;
  boundary.bytes = 8;
  assert(boundary.address_range_valid());
  assert(boundary.contains(7, 1));
  boundary.bytes = 9;
  assert(!boundary.address_range_valid());

  MemoryRegionToken malformed = token;
  malformed.wire_magic ^= 1;
  assert(!malformed.address_range_valid());
  malformed = token;
  ++malformed.wire_version;
  assert(!malformed.address_range_valid());
  malformed = token;
  --malformed.wire_bytes;
  assert(!malformed.address_range_valid());
  malformed = token;
  malformed.address = 0;
  assert(!malformed.address_range_valid());
  malformed = token;
  malformed.bytes = 0;
  assert(!malformed.address_range_valid());
}

void test_qp_info_wire_and_mtu_contract() {
  const QPInfo info{0x1234, 0x00abc123, 7, IBV_MTU_4096, 16, 8};
  assert(info.wire_valid());
  assert(info.negotiated_mtu(IBV_MTU_2048) == IBV_MTU_2048);
  assert(info.negotiated_max_qp_init_rd_atom(16) == 8);
  assert(info.negotiated_max_qp_rd_atom(4) == 4);

  QPInfo smaller_remote = info;
  smaller_remote.active_mtu = static_cast<u8>(IBV_MTU_1024);
  assert(smaller_remote.wire_valid());
  assert(smaller_remote.negotiated_mtu(IBV_MTU_4096) == IBV_MTU_1024);

  QPInfo malformed{};
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.wire_magic ^= 1;
  assert(!malformed.wire_valid());
  malformed = info;
  ++malformed.wire_version;
  assert(!malformed.wire_valid());
  malformed = info;
  --malformed.wire_bytes;
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.qp_number = 0;
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.qp_number = 0x01000000;
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.lid = 0;
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.active_mtu = 0;
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.active_mtu = static_cast<u8>(IBV_MTU_4096) + 1;
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.max_qp_init_rd_atom = 0;
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.max_qp_rd_atom = 0;
  assert(!malformed.wire_valid());
  malformed = info;
  malformed.reserved1 = 1;
  assert(!malformed.wire_valid());
}

void test_queue_pair_request_contract() {
  assert(QueuePairRequestContract::local_range_valid(16, 0, 16));
  assert(QueuePairRequestContract::local_range_valid(16, 15, 1));
  assert(!QueuePairRequestContract::local_range_valid(16, 0, 0));
  assert(!QueuePairRequestContract::local_range_valid(16, 16, 1));
  assert(!QueuePairRequestContract::local_range_valid(16, 17, 1));

  const u64 maximum = std::numeric_limits<u64>::max();
  assert(QueuePairRequestContract::address_range_valid(0x1000, 8, 8));
  assert(QueuePairRequestContract::address_range_valid(maximum, 0, 1));
  assert(QueuePairRequestContract::address_range_valid(maximum - 7, 0, 8));
  assert(!QueuePairRequestContract::address_range_valid(0, 0, 1));
  assert(!QueuePairRequestContract::address_range_valid(1, 0, 0));
  assert(!QueuePairRequestContract::address_range_valid(maximum, 1, 1));
  assert(!QueuePairRequestContract::address_range_valid(maximum - 7, 0, 9));

  assert(QueuePairRequestContract::opcode_supported(IBV_WR_SEND));
  assert(QueuePairRequestContract::opcode_supported(IBV_WR_RDMA_READ));
  assert(QueuePairRequestContract::opcode_supported(IBV_WR_RDMA_WRITE));
  assert(!QueuePairRequestContract::opcode_supported(IBV_WR_ATOMIC_CMP_AND_SWP));
  assert(QueuePairRequestContract::inline_opcode_supported(IBV_WR_SEND));
  assert(QueuePairRequestContract::inline_opcode_supported(IBV_WR_RDMA_WRITE));
  assert(!QueuePairRequestContract::inline_opcode_supported(IBV_WR_RDMA_READ));
  assert(!QueuePairRequestContract::atomic_address_valid(0));
  assert(QueuePairRequestContract::atomic_address_valid(8));
  assert(!QueuePairRequestContract::atomic_address_valid(10));
}

void test_completion_poll_contract() {
  assert(CompletionPollContract::batch_size(100, 16) == 16);
  assert(CompletionPollContract::batch_size(8, 16) == 8);
  assert(CompletionPollContract::batch_size(0, 16) == 0);
  assert(CompletionPollContract::batch_size(-1, 16) == 0);
  assert(CompletionPollContract::batch_size(8, 0) == 0);
  assert(CompletionPollContract::batch_size(8, -1) == 0);

  i32 remaining = 37;
  const i32 partial_completions[]{5, 0, 11, 16, 5};
  for (const i32 completed : partial_completions) {
    const i32 requested = CompletionPollContract::batch_size(remaining, 16);
    assert(requested > 0 && requested <= remaining);
    assert(completed >= 0 && completed <= requested);
    remaining -= completed;
  }
  assert(remaining == 0);
}

void test_configuration_contract() {
  configuration::Configuration defaults;
  assert(defaults.rdma_limits_valid());

  configuration::Configuration send_poll;
  send_poll.max_poll_cqes = send_poll.max_send_queue_wr + 1;
  assert(!send_poll.rdma_limits_valid());

  configuration::Configuration receive_poll;
  receive_poll.max_poll_cqes = receive_poll.max_recv_queue_wr + 1;
  assert(!receive_poll.rdma_limits_valid());

  configuration::Configuration zero_timeout;
  zero_timeout.qp_handshake_timeout_ms = 0;
  assert(!zero_timeout.rdma_limits_valid());

  configuration::Configuration large_timeout;
  large_timeout.qp_handshake_timeout_ms = 300001;
  assert(!large_timeout.rdma_limits_valid());

  configuration::Configuration bad_port;
  bad_port.port = 0;
  assert(!bad_port.rdma_limits_valid());

  configuration::Configuration bad_device_port;
  bad_device_port.device_port = 256;
  assert(!bad_device_port.rdma_limits_valid());
}

void test_endpoint_contract() {
  assert(is_ipv4_literal("127.0.0.1"));
  assert(!is_ipv4_literal("server.example.com"));
  assert(!is_ipv4_literal("1.2.3.999"));

  const Endpoint literal = parse_endpoint("127.0.0.1:65535", 1234);
  assert(literal.host == "127.0.0.1");
  assert(literal.address == "127.0.0.1");
  assert(literal.port == 65535);

}

}  // namespace

int main() {
  test_memory_region_token_wire_contract();
  test_qp_info_wire_and_mtu_contract();
  test_queue_pair_request_contract();
  test_completion_poll_contract();
  test_configuration_contract();
  test_endpoint_contract();
  return 0;
}
