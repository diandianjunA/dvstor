#pragma once

#include <array>
#include <cstring>
#include <utility>

#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace memory_node_storage_owner_index_detail {

struct LockedNodeIdentity {
  u64 header{};
  node_t id{};
  u32 generation{};
  u32 slot_incarnation{};
};

inline constexpr size_t kLockedNodeIdentityBytes =
  VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;

inline LockedNodeIdentity decode_locked_node_identity(
    const byte_t* encoded) {
  LockedNodeIdentity identity;
  std::memcpy(&identity.header, encoded, sizeof(identity.header));
  std::memcpy(&identity.id, encoded + VamanaNode::offset_id(),
              sizeof(identity.id));
  std::memcpy(&identity.generation,
              encoded + VamanaNode::offset_generation(),
              sizeof(identity.generation));
  std::memcpy(&identity.slot_incarnation,
              encoded + VamanaNode::offset_slot_incarnation(),
              sizeof(identity.slot_incarnation));
  return identity;
}

inline bool locked_node_identity_matches(
    RemotePtr pointer, const LockedNodeIdentity& identity) {
  return !pointer.is_null() && pointer.is_well_formed() &&
    (identity.header & VamanaNode::HEADER_NODE_LOCK) != 0 &&
    VamanaNode::header_incarnation(identity.header) ==
      pointer.incarnation() &&
    identity.slot_incarnation == pointer.incarnation();
}

// Reader must copy exactly kLockedNodeIdentityBytes into destination and
// return true. Both local memory and a one-sided RDMA read use this protocol,
// which keeps identity validation identical across physical homes.
template <typename Reader>
bool read_and_validate_locked_node_identity(
    RemotePtr pointer, Reader&& reader, LockedNodeIdentity& identity) {
  std::array<byte_t, kLockedNodeIdentityBytes> encoded{};
  if (!std::forward<Reader>(reader)(encoded.data(), encoded.size())) {
    identity = {};
    return false;
  }
  identity = decode_locked_node_identity(encoded.data());
  return locked_node_identity_matches(pointer, identity);
}

inline bool make_locked_header_publication(
    RemotePtr pointer, u64 observed_header, u64 set_flags, u64 clear_flags,
    u64& desired_header) {
  desired_header = 0;
  const u64 changed_flags = set_flags | clear_flags;
  if (pointer.is_null() || !pointer.is_well_formed() ||
      (observed_header & VamanaNode::HEADER_NODE_LOCK) == 0 ||
      VamanaNode::header_incarnation(observed_header) !=
        pointer.incarnation() ||
      (changed_flags & ~VamanaNode::HEADER_FLAG_MASK) != 0 ||
      (changed_flags & VamanaNode::HEADER_NODE_LOCK) != 0 ||
      (set_flags & clear_flags) != 0) {
    return false;
  }
  desired_header =
    ((observed_header | set_flags) & ~clear_flags) &
    ~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK);
  return VamanaNode::header_incarnation(desired_header) ==
    pointer.incarnation();
}

// CompareExchange returns the header value that occupied the slot when the
// CAS was attempted. A failed CAS never performs a fallback unlock: doing so
// could clear NODE_LOCK belonging to an ABA-reused incarnation.
template <typename CompareExchange>
bool publish_locked_node_header_transition(
    RemotePtr pointer, u64 observed_header, u64 set_flags, u64 clear_flags,
    CompareExchange&& compare_exchange) {
  u64 desired_header = 0;
  if (!make_locked_header_publication(
        pointer, observed_header, set_flags, clear_flags, desired_header)) {
    return false;
  }
  const u64 original = std::forward<CompareExchange>(compare_exchange)(
    observed_header, desired_header);
  return original == observed_header;
}

}  // namespace memory_node_storage_owner_index_detail
