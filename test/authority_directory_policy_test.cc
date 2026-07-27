#include <cassert>

#include "memory_node/storage_owner_index/authority_directory_policy.hh"

namespace detail = memory_node_storage_owner_index_detail;

namespace {

using service::storage_owner::MutationKind;

RemotePtr pointer(u32 shard, u64 offset) {
  assert(offset != 0);
  return RemotePtr{shard, offset};
}

detail::AuthorityDirectoryState live_state(RemotePtr current,
                                           u32 generation,
                                           u64 placement_version) {
  return {
    .exists = true,
    .entry = {
      .current = current,
      .generation = generation,
      .deleted = false,
      .placement_version = placement_version,
      .last_committed_operation = {},
      .last_committed_kind = MutationKind::insert,
      .last_committed_stage1_home = 0,
      .last_committed_result = {},
      .last_relocation_operation = {},
      .last_relocation_generation = 0,
      .last_relocation_expected = {},
      .last_relocation_desired = {},
      .last_relocation_expected_version = 0,
    },
    .lease = std::nullopt,
  };
}

void test_begin_replay_busy_and_token_fenced_commit() {
  detail::AuthorityDirectoryState state;
  const detail::AuthorityOperationToken operation{7, 3, 101};
  // Same source and batch, but a different item is a different operation.
  const detail::AuthorityOperationToken other{7, 4, 101};
  const RemotePtr stage1_result = pointer(3, 4096);
  constexpr u64 maintenance_sequence = 501;

  const detail::AuthorityBeginResult begin =
    detail::begin_authority_mutation(
      state, MutationKind::insert, operation, 3);
  assert(begin.state == detail::AuthorityBeginState::prepared);
  assert(begin.generation == 1);
  assert(begin.previous.current.is_null());
  assert(state.lease.has_value());

  const detail::AuthorityBeginResult replay =
    detail::begin_authority_mutation(
      state, MutationKind::insert, operation, 3);
  assert(replay.state == detail::AuthorityBeginState::replay);
  assert(replay.generation == begin.generation);

  const detail::AuthorityBeginResult conflicting_replay =
    detail::begin_authority_mutation(
      state, MutationKind::insert, operation, 4);
  assert(conflicting_replay.state == detail::AuthorityBeginState::conflict);

  const detail::AuthorityBeginResult competing_begin =
    detail::begin_authority_mutation(
      state, MutationKind::insert, other, 3);
  assert(competing_begin.state == detail::AuthorityBeginState::busy);

  assert(detail::commit_authority_mutation(
           state, other, stage1_result, 1, false,
           maintenance_sequence) ==
         detail::AuthorityCommitState::stale);
  assert(state.lease.has_value());
  assert(detail::commit_authority_mutation(
           state, operation, stage1_result, 2, false,
           maintenance_sequence) ==
         detail::AuthorityCommitState::stale);
  assert(state.lease.has_value());

  assert(detail::commit_authority_mutation(
           state, operation, stage1_result, 1, false,
           maintenance_sequence) ==
         detail::AuthorityCommitState::committed);
  assert(!state.lease.has_value());
  assert(state.exists);
  assert(state.entry.current == stage1_result);
  assert(state.entry.generation == 1);
  assert(state.entry.placement_version == 1);

  assert(detail::commit_authority_mutation(
           state, operation, stage1_result, 1, false,
           maintenance_sequence) ==
         detail::AuthorityCommitState::replay);
  assert(detail::commit_authority_mutation(
           state, operation, stage1_result, 1, false,
           maintenance_sequence + 1) ==
         detail::AuthorityCommitState::conflict);

  const auto committed_replay = detail::begin_authority_mutation(
    state, MutationKind::insert, operation, 3);
  assert(committed_replay.state ==
         detail::AuthorityBeginState::committed_replay);
  assert(!committed_replay.acquired());
  assert(committed_replay.replay_result.new_pointer == stage1_result);
  assert(committed_replay.replay_result.old_pointer.is_null());
  assert(committed_replay.replay_result.generation == 1);
  assert(committed_replay.replay_result.maintenance_sequence ==
         maintenance_sequence);
  assert(detail::begin_authority_mutation(
           state, MutationKind::insert, operation, 4).state ==
         detail::AuthorityBeginState::conflict);

  // A late public retry must reproduce the original Stage1 ACK even after
  // Stage2 has changed the physical directory pointer.
  const RemotePtr relocated = pointer(5, 8192);
  const detail::AuthorityOperationToken relocation_operation{99, 0, 7001};
  assert(detail::relocate_authority_if_current(
           state, relocation_operation, 1, stage1_result,
           relocated, 1) == detail::AuthorityRelocateState::committed);
  const auto replay_after_relocation = detail::begin_authority_mutation(
    state, MutationKind::insert, operation, 3);
  assert(replay_after_relocation.state ==
         detail::AuthorityBeginState::committed_replay);
  assert(replay_after_relocation.replay_result.new_pointer == stage1_result);
  assert(replay_after_relocation.replay_result.maintenance_sequence ==
         maintenance_sequence);
  assert(state.entry.current == relocated);
  assert(detail::commit_authority_mutation(
           state, operation, stage1_result, 1, false,
           maintenance_sequence) == detail::AuthorityCommitState::replay);
  assert(detail::abort_authority_mutation(state, operation) ==
         detail::AuthorityAbortState::already_committed);
  assert(detail::begin_authority_mutation(
           state, MutationKind::insert, other, 3).state ==
         detail::AuthorityBeginState::already_exists);
}

void test_abort_and_mutation_semantics() {
  detail::AuthorityDirectoryState state;
  const detail::AuthorityOperationToken operation{1, 0, 11};
  const detail::AuthorityOperationToken other{1, 0, 12};

  assert(detail::begin_authority_mutation(
           state, MutationKind::erase, operation, 0).state ==
         detail::AuthorityBeginState::not_found);

  const auto begin = detail::begin_authority_mutation(
    state, MutationKind::upsert, operation, 2);
  assert(begin.state == detail::AuthorityBeginState::prepared);
  assert(detail::abort_authority_mutation(state, other) ==
         detail::AuthorityAbortState::wrong_operation);
  assert(state.lease.has_value());
  assert(detail::abort_authority_mutation(state, operation) ==
         detail::AuthorityAbortState::aborted);
  assert(!state.lease.has_value());
  assert(detail::abort_authority_mutation(state, operation) ==
         detail::AuthorityAbortState::not_active);

  state = live_state(pointer(2, 8192), 4, 9);
  const auto erase_begin = detail::begin_authority_mutation(
    state, MutationKind::erase, operation, 2);
  assert(erase_begin.state == detail::AuthorityBeginState::prepared);
  assert(erase_begin.generation == 5);
  assert(detail::commit_authority_mutation(
           state, operation, erase_begin.previous.current,
           erase_begin.generation, true, 777) ==
         detail::AuthorityCommitState::committed);
  assert(state.entry.deleted);
  assert(state.entry.generation == 5);
  assert(state.entry.placement_version == 10);
  assert(detail::begin_authority_mutation(
           state, MutationKind::erase, other, 2).state ==
         detail::AuthorityBeginState::already_deleted);
}

void test_gen1_stage2_loses_to_gen2_begin() {
  const RemotePtr generation1 = pointer(1, 4096);
  const RemotePtr stale_relocation = pointer(4, 8192);
  const RemotePtr generation2 = pointer(2, 12288);
  detail::AuthorityDirectoryState state = live_state(generation1, 1, 7);
  const detail::AuthorityOperationToken stage2_operation{5, 0, 1001};
  const detail::AuthorityOperationToken successor_operation{6, 0, 2001};

  const auto successor = detail::begin_authority_mutation(
    state, MutationKind::upsert, successor_operation, 2);
  assert(successor.state == detail::AuthorityBeginState::prepared);
  assert(successor.generation == 2);
  assert(successor.previous.current == generation1);
  assert(successor.previous.placement_version == 7);

  assert(detail::check_authority_current(
           state, stage2_operation, 1, generation1, 7) ==
         detail::AuthorityCheckState::busy);
  assert(detail::check_authority_current(
           state, successor_operation, 2, generation1, 7) ==
         detail::AuthorityCheckState::pending);
  assert(detail::relocate_authority_if_current(
           state, stage2_operation, 1, generation1,
           stale_relocation, 7) ==
         detail::AuthorityRelocateState::busy);
  assert(state.entry.current == generation1);
  assert(state.entry.placement_version == 7);

  assert(detail::commit_authority_mutation(
           state, successor_operation, generation2, 2, false, 900) ==
         detail::AuthorityCommitState::committed);
  assert(state.entry.current == generation2);
  assert(state.entry.placement_version == 8);
  assert(detail::relocate_authority_if_current(
           state, stage2_operation, 1, generation1,
           stale_relocation, 7) ==
         detail::AuthorityRelocateState::stale);
}

void test_gen1_stage2_wins_before_gen2_begin() {
  const RemotePtr generation1 = pointer(1, 4096);
  const RemotePtr relocated = pointer(4, 8192);
  detail::AuthorityDirectoryState state = live_state(generation1, 1, 7);
  const detail::AuthorityOperationToken stage2_operation{5, 0, 1002};
  const detail::AuthorityOperationToken successor_operation{6, 0, 2002};

  assert(detail::relocate_authority_if_current(
           state, stage2_operation, 1, generation1, relocated, 7) ==
         detail::AuthorityRelocateState::committed);
  assert(state.entry.current == relocated);
  assert(state.entry.placement_version == 8);

  const auto successor = detail::begin_authority_mutation(
    state, MutationKind::upsert, successor_operation, 2);
  assert(successor.state == detail::AuthorityBeginState::prepared);
  assert(successor.generation == 2);
  assert(successor.previous.current == relocated);
  assert(successor.previous.placement_version == 8);
}

void test_abort_releases_old_stage2_without_corrupting_directory() {
  const RemotePtr generation1 = pointer(1, 4096);
  const RemotePtr relocated = pointer(4, 8192);
  detail::AuthorityDirectoryState state = live_state(generation1, 1, 7);
  const detail::AuthorityOperationToken stage2_operation{5, 0, 1003};
  const detail::AuthorityOperationToken successor_operation{6, 0, 2003};

  assert(detail::begin_authority_mutation(
           state, MutationKind::upsert, successor_operation, 2).state ==
         detail::AuthorityBeginState::prepared);
  assert(detail::relocate_authority_if_current(
           state, stage2_operation, 1, generation1, relocated, 7) ==
         detail::AuthorityRelocateState::busy);
  assert(detail::abort_authority_mutation(state, successor_operation) ==
         detail::AuthorityAbortState::aborted);
  assert(detail::relocate_authority_if_current(
           state, stage2_operation, 1, generation1, relocated, 7) ==
         detail::AuthorityRelocateState::committed);
  assert(state.entry.current == relocated);
  assert(state.entry.generation == 1);
  assert(state.entry.placement_version == 8);
}

void test_relocation_is_versioned_and_idempotent() {
  const RemotePtr original = pointer(1, 4096);
  const RemotePtr relocated = pointer(3, 8192);
  detail::AuthorityDirectoryState state = live_state(original, 5, 10);
  const detail::AuthorityOperationToken operation{9, 0, 3001};
  const detail::AuthorityOperationToken other{9, 0, 3002};

  assert(detail::check_authority_current(
           state, operation, 5, original, 10) ==
         detail::AuthorityCheckState::current);
  assert(detail::relocate_authority_if_current(
           state, operation, 5, original, relocated, 10) ==
         detail::AuthorityRelocateState::committed);
  assert(state.entry.current == relocated);
  assert(state.entry.placement_version == 11);

  assert(detail::relocate_authority_if_current(
           state, operation, 5, original, relocated, 10) ==
         detail::AuthorityRelocateState::replay);
  assert(detail::relocate_authority_if_current(
           state, operation, 5, original, pointer(4, 12288), 10) ==
         detail::AuthorityRelocateState::conflict);
  assert(detail::relocate_authority_if_current(
           state, other, 5, original, relocated, 10) ==
         detail::AuthorityRelocateState::stale);
  assert(detail::check_authority_current(
           state, operation, 5, relocated, 11) ==
         detail::AuthorityCheckState::current);

  assert(detail::relocate_authority_if_current(
           state, other, 5, relocated, relocated, 11) ==
         detail::AuthorityRelocateState::committed);
  assert(state.entry.placement_version == 11);
  assert(detail::relocate_authority_if_current(
           state, other, 5, relocated, relocated, 11) ==
         detail::AuthorityRelocateState::committed);
  // A validation barrier does not consume the token's later relocation.
  assert(detail::relocate_authority_if_current(
           state, other, 5, relocated, pointer(4, 16384), 11) ==
         detail::AuthorityRelocateState::committed);
  assert(state.entry.current == pointer(4, 16384));
  assert(state.entry.placement_version == 12);
}

void test_cleanup_retirement_barrier_waits_for_linearization() {
  const RemotePtr base_generation = pointer(2, 4096);
  const RemotePtr successor_generation = pointer(4, 8192);
  const detail::AuthorityOperationToken operation{11, 3, 4001};
  detail::AuthorityDirectoryState state = live_state(base_generation, 0, 0);

  const auto begin = detail::begin_authority_mutation(
    state, MutationKind::upsert, operation, 4);
  assert(begin.state == detail::AuthorityBeginState::prepared);
  assert(begin.generation == 1);
  // desired == null is the cleanup-only, read-only retirement barrier. The
  // old physical generation remains authoritative while the lease is active.
  assert(detail::relocate_authority_if_current(
           state, operation, 0, base_generation, RemotePtr{}, 0) ==
         detail::AuthorityRelocateState::busy);
  assert(state.entry.current == base_generation);

  assert(detail::abort_authority_mutation(state, operation) ==
         detail::AuthorityAbortState::aborted);
  assert(detail::relocate_authority_if_current(
           state, operation, 0, base_generation, RemotePtr{}, 0) ==
         detail::AuthorityRelocateState::stale);
  assert(state.entry.current == base_generation);

  const auto replay_begin = detail::begin_authority_mutation(
    state, MutationKind::upsert, operation, 4);
  assert(replay_begin.state == detail::AuthorityBeginState::prepared);
  assert(detail::commit_authority_mutation(
           state, operation, successor_generation, 1, false, 9001) ==
         detail::AuthorityCommitState::committed);
  assert(detail::relocate_authority_if_current(
           state, operation, 0, base_generation, RemotePtr{}, 0) ==
         detail::AuthorityRelocateState::committed);

  // The proof is generation based and remains bounded: no per-operation
  // history is needed after an even newer mutation starts.
  const detail::AuthorityOperationToken later{12, 0, 4002};
  assert(detail::begin_authority_mutation(
           state, MutationKind::upsert, later, 1).state ==
         detail::AuthorityBeginState::prepared);
  assert(detail::relocate_authority_if_current(
           state, operation, 0, base_generation, RemotePtr{}, 0) ==
         detail::AuthorityRelocateState::committed);
}

}  // namespace

int main() {
  test_begin_replay_busy_and_token_fenced_commit();
  test_abort_and_mutation_semantics();
  test_gen1_stage2_loses_to_gen2_begin();
  test_gen1_stage2_wins_before_gen2_begin();
  test_abort_releases_old_stage2_without_corrupting_directory();
  test_relocation_is_versioned_and_idempotent();
  test_cleanup_retirement_barrier_waits_for_linearization();
  return 0;
}
