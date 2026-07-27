#include <algorithm>
#include <array>
#include <cassert>

#include "memory_node/storage_owner_index/stage1_reachability_policy.hh"

namespace {

RemotePtr pointer(u32 shard, u64 offset, u32 incarnation = 0) {
  return RemotePtr{shard, offset, incarnation};
}

void test_identical_hot_neighbor_sets_rotate_across_parents() {
  constexpr size_t record_bytes = 256;
  vec<RemotePtr> targets;
  for (u32 index = 0; index < 8; ++index) {
    targets.push_back(pointer(0, 0x1000 + index * 0x100));
  }
  std::array<u32, 8> loads{};
  for (u32 insertion = 0; insertion < 8; ++insertion) {
    const RemotePtr candidate = pointer(
      0, 0x10000 + static_cast<u64>(insertion) * record_bytes, 1);
    const vec<RemotePtr> accepted =
      memory_node_storage_owner_index_detail::
        select_stage1_reachability_bridges(
          candidate, span<const RemotePtr>{targets}, record_bytes,
          [&](RemotePtr target) {
            const auto position = std::find(
              targets.begin(), targets.end(), target);
            assert(position != targets.end());
            ++loads[static_cast<size_t>(position - targets.begin())];
            return true;
          });
    assert(accepted.size() == 2);
  }

  // Eight consecutive insertions create sixteen certificates. A fixed-first
  // policy would put all eight in targets[0]/targets[1]; cyclic placement
  // spreads them exactly across the identical candidate set.
  for (const u32 load : loads) assert(load == 2);
}

void test_hot_cluster_uses_aggregate_protected_capacity() {
  constexpr size_t record_bytes = 256;
  constexpr u32 neighbor_count = 96;
  constexpr u32 protected_slots = 6;
  constexpr u32 expected_insertions =
    neighbor_count * protected_slots / 2;
  vec<RemotePtr> targets;
  targets.reserve(neighbor_count);
  for (u32 index = 0; index < neighbor_count; ++index) {
    targets.push_back(pointer(0, 0x1000 + index * 0x100));
  }
  std::array<u32, neighbor_count> loads{};

  for (u32 insertion = 0; insertion < expected_insertions; ++insertion) {
    const RemotePtr candidate = pointer(
      0, 0x100000 + static_cast<u64>(insertion) * record_bytes, 1);
    const vec<RemotePtr> accepted =
      memory_node_storage_owner_index_detail::
        select_stage1_reachability_bridges(
          candidate, span<const RemotePtr>{targets}, record_bytes,
          [&](RemotePtr target) {
            const size_t index = static_cast<size_t>(
              std::find(targets.begin(), targets.end(), target) -
              targets.begin());
            assert(index < targets.size());
            if (loads[index] == protected_slots) return false;
            ++loads[index];
            return true;
          });
    assert(accepted.size() == 2);
  }
  for (const u32 load : loads) assert(load == protected_slots);
}

void test_only_successful_ack_targets_are_recorded() {
  constexpr size_t record_bytes = 256;
  const vec<RemotePtr> targets{
    pointer(0, 0x1000), pointer(0, 0x2000), pointer(0, 0x3000),
    pointer(0, 0x4000)};
  const RemotePtr candidate = pointer(0, record_bytes * 12, 1);
  const size_t start =
    memory_node_storage_owner_index_detail::stage1_bridge_rotation(
      candidate, targets.size(), record_bytes);
  const RemotePtr rejected = targets[start];
  vec<RemotePtr> callbacks;
  const vec<RemotePtr> accepted =
    memory_node_storage_owner_index_detail::
      select_stage1_reachability_bridges(
        candidate, span<const RemotePtr>{targets}, record_bytes,
        [&](RemotePtr target) {
          callbacks.push_back(target);
          return target != rejected;
        });

  assert(callbacks.size() == 3);
  assert(accepted.size() == 2);
  assert(std::find(accepted.begin(), accepted.end(), rejected) ==
         accepted.end());
  assert(accepted[0] == targets[(start + 1) % targets.size()]);
  assert(accepted[1] == targets[(start + 2) % targets.size()]);
}

void test_one_bridge_is_a_valid_bounded_certificate() {
  constexpr size_t record_bytes = 256;
  const vec<RemotePtr> targets{
    pointer(0, 0x1000), pointer(0, 0x2000), pointer(0, 0x3000)};
  const RemotePtr only_available = targets[1];
  const vec<RemotePtr> accepted =
    memory_node_storage_owner_index_detail::
      select_stage1_reachability_bridges(
        pointer(0, 0x8000, 3), span<const RemotePtr>{targets}, record_bytes,
        [&](RemotePtr target) { return target == only_available; });
  assert((accepted == vec<RemotePtr>{only_available}));
}

void test_duplicates_do_not_consume_attempt_or_certificate_capacity() {
  constexpr size_t record_bytes = 256;
  const RemotePtr first = pointer(0, 0x1000);
  const RemotePtr second = pointer(0, 0x2000);
  const vec<RemotePtr> targets{first, first, second, first};
  u32 callbacks = 0;
  const vec<RemotePtr> accepted =
    memory_node_storage_owner_index_detail::
      select_stage1_reachability_bridges(
        pointer(0, 0x9000, 2), span<const RemotePtr>{targets}, record_bytes,
        [&](RemotePtr) {
          ++callbacks;
          return true;
        });
  assert(callbacks == 2);
  assert(accepted.size() == 2);
  assert(accepted[0] != accepted[1]);
}

}  // namespace

int main() {
  test_identical_hot_neighbor_sets_rotate_across_parents();
  test_hot_cluster_uses_aggregate_protected_capacity();
  test_only_successful_ack_targets_are_recorded();
  test_one_bridge_is_a_valid_bounded_certificate();
  test_duplicates_do_not_consume_attempt_or_certificate_capacity();
  return 0;
}
