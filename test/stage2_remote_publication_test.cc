#include <array>
#include <cassert>
#include <cstring>

#include "memory_node/storage_owner_index/locked_node_publication.hh"

namespace {

using memory_node_storage_owner_index_detail::LockedNodeIdentity;
using memory_node_storage_owner_index_detail::kLockedNodeIdentityBytes;
using memory_node_storage_owner_index_detail::
  publish_locked_node_header_transition;
using memory_node_storage_owner_index_detail::
  read_and_validate_locked_node_identity;

class InjectedRemoteRecord {
public:
  void store(u64 header, node_t id, u32 generation, u32 slot_incarnation) {
    std::memcpy(bytes_.data(), &header, sizeof(header));
    std::memcpy(bytes_.data() + VamanaNode::offset_id(), &id, sizeof(id));
    std::memcpy(bytes_.data() + VamanaNode::offset_generation(),
                &generation, sizeof(generation));
    std::memcpy(bytes_.data() + VamanaNode::offset_slot_incarnation(),
                &slot_incarnation, sizeof(slot_incarnation));
  }

  bool rdma_read(byte_t* destination, size_t bytes) const {
    assert(bytes == bytes_.size());
    std::memcpy(destination, bytes_.data(), bytes);
    return true;
  }

  u64 compare_and_swap(u64 expected, u64 desired) {
    ++cas_calls_;
    const u64 original = header();
    if (original == expected) {
      std::memcpy(bytes_.data(), &desired, sizeof(desired));
    }
    return original;
  }

  u64 header() const {
    u64 value = 0;
    std::memcpy(&value, bytes_.data(), sizeof(value));
    return value;
  }

  u32 cas_calls() const { return cas_calls_; }

private:
  std::array<byte_t, kLockedNodeIdentityBytes> bytes_{};
  u32 cas_calls_{};
};

u64 acquire_lock(InjectedRemoteRecord& record) {
  const u64 observed = record.header();
  const u64 original = record.compare_and_swap(
    observed, observed | static_cast<u64>(VamanaNode::HEADER_NODE_LOCK));
  assert(original == observed);
  return record.header();
}

bool read_locked(RemotePtr pointer, const InjectedRemoteRecord& record,
                 LockedNodeIdentity& identity) {
  return read_and_validate_locked_node_identity(
    pointer,
    [&](byte_t* destination, size_t bytes) {
      return record.rdma_read(destination, bytes);
    },
    identity);
}

bool publish(RemotePtr pointer, InjectedRemoteRecord& record,
             u64 observed, u64 set_flags, u64 clear_flags) {
  return publish_locked_node_header_transition(
    pointer, observed, set_flags, clear_flags,
    [&](u64 expected, u64 desired) {
      return record.compare_and_swap(expected, desired);
    });
}

void test_cross_shard_final_target_freeze_and_rebase_publication() {
  constexpr node_t id = 42;
  constexpr u32 generation = 6;
  constexpr u32 source_incarnation = 3;
  constexpr u32 destination_incarnation = 9;
  const RemotePtr stage1_home{0, 0x1000, source_incarnation};
  const RemotePtr final_target{1, 0x2000, destination_incarnation};
  assert(stage1_home.memory_node() != final_target.memory_node());

  InjectedRemoteRecord source;
  const u64 source_header = VamanaNode::make_header(
    source_incarnation,
    VamanaNode::HEADER_PROVISIONAL |
      VamanaNode::HEADER_STAGE2_FROZEN);
  source.store(source_header, id, generation, source_incarnation);

  InjectedRemoteRecord destination;
  destination.store(
    VamanaNode::make_header(destination_incarnation), id, generation,
    destination_incarnation);

  // The Stage1 owner acquires and validates the record that physically lives
  // on another shard. Publication freezes and unlocks that exact incarnation
  // in one CAS; it cannot accidentally touch the local Stage1 source.
  const u64 first_locked = acquire_lock(destination);
  LockedNodeIdentity first_identity;
  assert(read_locked(final_target, destination, first_identity));
  assert(first_identity.header == first_locked);
  assert(first_identity.id == id);
  assert(first_identity.generation == generation);
  assert(publish(final_target, destination, first_identity.header,
                 VamanaNode::HEADER_STAGE2_FROZEN, 0));
  const u64 frozen = destination.header();
  assert((frozen & VamanaNode::HEADER_NODE_LOCK) == 0);
  assert((frozen & VamanaNode::HEADER_STAGE2_FROZEN) != 0);
  assert(source.header() == source_header);

  // A concurrent reverse mutation may acquire the remote lock after the
  // freeze, but the shared admission predicate rejects it. Its unlock leaves
  // FROZEN intact, so retry cannot create an ACK-to-publication gap.
  acquire_lock(destination);
  LockedNodeIdentity mutation_identity;
  assert(read_locked(final_target, destination, mutation_identity));
  assert(!VamanaNode::stable_graph_mutation_allowed(
    mutation_identity.header));
  assert(publish(final_target, destination, mutation_identity.header, 0, 0));
  assert(destination.header() == frozen);

  // Stage2 then reacquires the same remote incarnation after its adjacency
  // rebase and atomically clears FROZEN while unlocking it.
  acquire_lock(destination);
  LockedNodeIdentity rebase_identity;
  assert(read_locked(final_target, destination, rebase_identity));
  assert((rebase_identity.header &
          VamanaNode::HEADER_STAGE2_FROZEN) != 0);
  assert(publish(final_target, destination, rebase_identity.header, 0,
                 VamanaNode::HEADER_STAGE2_FROZEN));
  const u64 published = destination.header();
  assert((published & (VamanaNode::HEADER_NODE_LOCK |
                       VamanaNode::HEADER_STAGE2_FROZEN)) == 0);
  assert(VamanaNode::header_incarnation(published) ==
         destination_incarnation);
  assert(VamanaNode::stable_graph_mutation_allowed(published));
  assert(source.header() == source_header);
}

void test_remote_identity_mismatch_never_reaches_publication() {
  constexpr u32 pointer_incarnation = 12;
  const RemotePtr final_target{2, 0x4000, pointer_incarnation};
  InjectedRemoteRecord destination;
  destination.store(
    VamanaNode::make_header(
      pointer_incarnation, VamanaNode::HEADER_NODE_LOCK),
    77, 5, pointer_incarnation + 1);

  LockedNodeIdentity identity;
  assert(!read_locked(final_target, destination, identity));
  const u32 cas_calls_before = destination.cas_calls();
  // This models the worker's failure path: a failed identity read is never
  // passed to the lifecycle publication CAS, and the lock that this worker
  // successfully acquired is released without changing lifecycle flags.
  assert((destination.header() & VamanaNode::HEADER_NODE_LOCK) != 0);
  assert(destination.cas_calls() == cas_calls_before);
  const u64 locked = destination.header();
  assert(destination.compare_and_swap(
           locked,
           locked & ~static_cast<u64>(VamanaNode::HEADER_NODE_LOCK)) ==
         locked);
  assert((destination.header() & VamanaNode::HEADER_NODE_LOCK) == 0);
}

void test_remote_publication_cas_is_aba_fenced_and_does_not_unlock_reuse() {
  constexpr u32 old_incarnation = 21;
  constexpr u32 new_incarnation = 22;
  const RemotePtr stale_target{3, 0x6000, old_incarnation};
  InjectedRemoteRecord destination;
  destination.store(
    VamanaNode::make_header(old_incarnation,
                            VamanaNode::HEADER_NODE_LOCK),
    88, 10, old_incarnation);

  LockedNodeIdentity old_identity;
  assert(read_locked(stale_target, destination, old_identity));

  // Inject slot reuse between the locked identity read and publication. Real
  // reclamation must not do this while locked, but the CAS fence is the final
  // defense against stale/buggy ownership and must never clear the new lock.
  const u64 replacement = VamanaNode::make_header(
    new_incarnation,
    VamanaNode::HEADER_NODE_LOCK | VamanaNode::HEADER_CENTROID_ACCOUNTED);
  destination.store(replacement, 99, 11, new_incarnation);
  assert(!publish(stale_target, destination, old_identity.header,
                  VamanaNode::HEADER_STAGE2_FROZEN, 0));
  assert(destination.header() == replacement);
  assert((destination.header() & VamanaNode::HEADER_NODE_LOCK) != 0);
  assert(VamanaNode::header_incarnation(destination.header()) ==
         new_incarnation);
}

void test_publication_rejects_incarnation_or_lock_flag_mutation() {
  constexpr u32 incarnation = 31;
  const RemotePtr final_target{4, 0x8000, incarnation};
  InjectedRemoteRecord destination;
  destination.store(
    VamanaNode::make_header(incarnation, VamanaNode::HEADER_NODE_LOCK),
    100, 12, incarnation);
  const u64 observed = destination.header();
  const u32 cas_calls_before = destination.cas_calls();

  assert(!publish(final_target, destination, observed,
                  u64{1} << VamanaNode::HEADER_INCARNATION_SHIFT, 0));
  assert(!publish(final_target, destination, observed,
                  VamanaNode::HEADER_NODE_LOCK, 0));
  assert(!publish(final_target, destination, observed,
                  VamanaNode::HEADER_STAGE2_FROZEN,
                  VamanaNode::HEADER_STAGE2_FROZEN));
  assert(destination.cas_calls() == cas_calls_before);
  assert(destination.header() == observed);
}

}  // namespace

int main() {
  test_cross_shard_final_target_freeze_and_rebase_publication();
  test_remote_identity_mismatch_never_reaches_publication();
  test_remote_publication_cas_is_aba_fenced_and_does_not_unlock_reuse();
  test_publication_rejects_incarnation_or_lock_flag_mutation();
  return 0;
}
