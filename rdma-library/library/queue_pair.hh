#ifndef RDMA_LIBRARY_QUEUE_PAIR_HH
#define RDMA_LIBRARY_QUEUE_PAIR_HH

#include <infiniband/verbs.h>

#include <bit>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "types.hh"
#include "context.hh"
#include "memory_region.hh"

// forward declarations
class Context;
class MemoryRegion;
struct MemoryRegionToken;

constexpr u32 INLINE_SIZE = 256;
constexpr u32 MESSAGE_SIZE = 1073741824;  // = 1GB (is max on our machines)

struct QueuePairRequestContract {
  static constexpr bool local_range_valid(u64 capacity,
                                          u64 offset,
                                          u64 length) {
    return length != 0 && offset <= capacity && length <= capacity - offset;
  }

  static constexpr bool address_range_valid(u64 address,
                                            u64 offset,
                                            u64 length) {
    return address != 0 && length != 0 &&
      offset <= std::numeric_limits<u64>::max() - address &&
      address + offset <= std::numeric_limits<u64>::max() - (length - 1);
  }

  static constexpr bool opcode_supported(enum ibv_wr_opcode opcode) {
    return opcode == IBV_WR_SEND || opcode == IBV_WR_RDMA_READ ||
      opcode == IBV_WR_RDMA_WRITE;
  }

  static constexpr bool inline_opcode_supported(enum ibv_wr_opcode opcode) {
    return opcode == IBV_WR_SEND || opcode == IBV_WR_RDMA_WRITE;
  }

  static constexpr bool atomic_address_valid(u64 address) {
    return address != 0 && address % sizeof(u64) == 0;
  }
};

struct QPInfo {
  static constexpr u32 kWireMagic = 0x44565150u;  // "DVQP"
  static constexpr u16 kWireVersion = 1;
  static constexpr u16 kWireBytes = 24;

  u32 wire_magic{};
  u16 wire_version{};
  u16 wire_bytes{};
  u32 qp_number{};
  u32 node_id{};
  u16 lid{};
  u8 active_mtu{};
  u8 max_qp_init_rd_atom{};
  u8 max_qp_rd_atom{};
  u8 reserved0{};
  u16 reserved1{};

  constexpr QPInfo() = default;
  constexpr QPInfo(u16 local_lid,
                   u32 local_qp_number,
                   u32 local_node_id,
                   enum ibv_mtu local_active_mtu,
                   u8 local_max_qp_init_rd_atom,
                   u8 local_max_qp_rd_atom)
      : wire_magic(kWireMagic),
        wire_version(kWireVersion),
        wire_bytes(kWireBytes),
        qp_number(local_qp_number),
        node_id(local_node_id),
        lid(local_lid),
        active_mtu(static_cast<u8>(local_active_mtu)),
        max_qp_init_rd_atom(local_max_qp_init_rd_atom),
        max_qp_rd_atom(local_max_qp_rd_atom) {}

  static constexpr bool mtu_valid(u8 value) {
    return value >= static_cast<u8>(IBV_MTU_256) &&
      value <= static_cast<u8>(IBV_MTU_4096);
  }

  constexpr bool wire_valid() const {
    return wire_magic == kWireMagic && wire_version == kWireVersion &&
      wire_bytes == kWireBytes && qp_number > 0 && qp_number <= 0x00ffffffu &&
      lid != 0 && mtu_valid(active_mtu) && max_qp_init_rd_atom > 0 &&
      max_qp_rd_atom > 0 && reserved0 == 0 && reserved1 == 0;
  }

  constexpr enum ibv_mtu negotiated_mtu(enum ibv_mtu local_active_mtu) const {
    const u8 local = static_cast<u8>(local_active_mtu);
    return static_cast<enum ibv_mtu>(
      local < active_mtu ? local : active_mtu);
  }

  constexpr u8 negotiated_max_qp_init_rd_atom(u8 local_limit) const {
    return local_limit < max_qp_rd_atom ? local_limit : max_qp_rd_atom;
  }

  constexpr u8 negotiated_max_qp_rd_atom(u8 local_limit) const {
    return local_limit < max_qp_init_rd_atom
      ? local_limit
      : max_qp_init_rd_atom;
  }
};
static_assert(sizeof(QPInfo) == QPInfo::kWireBytes);
static_assert(std::endian::native == std::endian::little,
              "QPInfo wire protocol requires little-endian hosts");
static_assert(std::is_standard_layout_v<QPInfo>);
static_assert(std::is_trivially_copyable_v<QPInfo>);
static_assert(offsetof(QPInfo, wire_magic) == 0);
static_assert(offsetof(QPInfo, qp_number) == 8);
static_assert(offsetof(QPInfo, node_id) == 12);
static_assert(offsetof(QPInfo, lid) == 16);
static_assert(offsetof(QPInfo, active_mtu) == 18);
static_assert(offsetof(QPInfo, max_qp_init_rd_atom) == 19);
static_assert(offsetof(QPInfo, max_qp_rd_atom) == 20);
static_assert(offsetof(QPInfo, reserved1) == 22);

class QueuePair {
public:
  explicit QueuePair(Context* context, bool use_shared_receive_cq = false);
  QueuePair(Context* context,
            ibv_cq* send_cq,
            ibv_cq* recv_cq,
            bool use_shared_receive_cq = false);

  QueuePair(const QueuePair&) = delete;
  QueuePair& operator=(const QueuePair&) = delete;
  ~QueuePair();

  u32 get_qp_num() { return queue_pair_->qp_num; }
  ibv_qp* get_ibv_qp() { return queue_pair_; }
  u32 max_send_wr() const { return max_send_wr_; }
  enum ibv_mtu active_mtu() const;
  u8 max_qp_read_atomic() const;
  u8 max_qp_dest_read_atomic() const;

  void transition_to_init();
  void transition_to_rtr(const QPInfo& remote_buffer);
  void transition_to_rts(const QPInfo& remote_buffer);

  void post_receive(MemoryRegion& region);
  void post_receive(MemoryRegion& region,
                    u32 size_in_bytes,
                    u64 wr_id = 0,
                    u64 local_offset = 0);
  u32 receive_u32(Context& context);

  void post_send_inlined(const void* address,
                         u32 size_in_bytes,
                         enum ibv_wr_opcode opcode,
                         bool signaled = true,
                         MemoryRegionToken* token = nullptr,
                         u64 remote_offset = 0,
                         u64 wr_id = 0);
  void post_send_u32(u32& value, bool signaled);
  void post_send(MemoryRegion& region,
                 enum ibv_wr_opcode opcode,
                 bool signaled = true,
                 MemoryRegionToken* token = nullptr,
                 u64 remote_offset = 0,
                 u64 local_offset = 0);
  void post_send(MemoryRegion& region,
                 u32 size_in_bytes,
                 enum ibv_wr_opcode opcode,
                 bool signaled = true,
                 MemoryRegionToken* token = nullptr,
                 u64 remote_offset = 0,
                 u64 local_offset = 0);
  void post_send_with_id(MemoryRegion& region,
                         u32 size_in_bytes,
                         enum ibv_wr_opcode opcode,
                         u64 wr_id,
                         bool signaled = true,
                         MemoryRegionToken* token = nullptr,
                         u64 remote_offset = 0,
                         u64 local_offset = 0);
  ibv_qp_init_attr get_qp_initial_attributes(ibv_cq* send_cq, ibv_cq* recv_cq);
  void post_send(u64 address,
                 u32 size,
                 u32 lkey,
                 enum ibv_wr_opcode opcode,
                 bool signaled,
                 bool inlined,
                 MemoryRegionToken* token,
                 u64 remote_offset,
                 u64 local_offset,
                 u64 wr_id);

  void post_CAS(MemoryRegion& local_region,
                MemoryRegionToken* remote_token,
                u64 remote_offset,
                u64 compare_to,
                u64 swap_with,
                bool signaled = true,
                u64 wr_id = 0);

  void post_CAS(u64 laddr,
                u32 lkey,
                MemoryRegionToken* remote_token,
                u64 remote_offset,
                u64 compare_to,
                u64 swap_with,
                bool signaled = true,
                u64 wr_id = 0);

  void post_FAA(u64 laddr,
                u32 lkey,
                MemoryRegionToken* remote_token,
                u64 remote_offset,
                u64 to_add,
                bool signaled = true,
                u64 wr_id = 0);

private:
  Context* context_;
  const u16 lid_;
  const bool use_shared_receive_cq_;

  ibv_qp* queue_pair_{nullptr};
  u32 max_send_wr_{};
};

using QP = u_ptr<QueuePair>;
using QPs = vec<QP>;

#endif  // RDMA_LIBRARY_QUEUE_PAIR_HH
