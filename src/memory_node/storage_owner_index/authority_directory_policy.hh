#pragma once

#include <limits>
#include <optional>
#include <utility>

#include "common/types.hh"
#include "remote_pointer.hh"
#include "service/storage_owner_protocol.hh"

namespace memory_node_storage_owner_index_detail {

// A semantic mutation identity is independent of a transport request ID.  The
// transport may retry a request, but every retry of one logical update must
// carry the same token.
using AuthorityOperationToken =
  service::storage_owner::AuthorityOperationToken;

inline bool valid_authority_operation(
    const AuthorityOperationToken& operation) {
  return operation.client_batch_id != 0;
}

inline bool same_authority_operation(
    const AuthorityOperationToken& lhs,
    const AuthorityOperationToken& rhs) {
  return lhs.source_client == rhs.source_client &&
    lhs.item_index == rhs.item_index &&
    lhs.client_batch_id == rhs.client_batch_id;
}

// One bounded public-response record is retained per logical ID.  In
// particular, new_pointer is the pointer returned by the original Stage1
// commit and must not be reconstructed from directory.current after Stage2
// relocates that generation.
struct AuthorityCommittedMutationResult {
  RemotePtr new_pointer;
  RemotePtr old_pointer;
  u32 generation{};
  u64 maintenance_sequence{};
};

// The stable authority owns this logical directory entry even when current
// points into a different physical shard.  generation changes for a logical
// mutation; placement_version changes for a same-generation relocation.
struct AuthorityDirectoryEntry {
  RemotePtr current;
  u32 generation{};
  bool deleted{};
  u64 placement_version{};
  AuthorityOperationToken last_committed_operation;
  service::storage_owner::MutationKind last_committed_kind{
    service::storage_owner::MutationKind::insert};
  u32 last_committed_stage1_home{};
  AuthorityCommittedMutationResult last_committed_result;
  AuthorityOperationToken last_relocation_operation;
  u32 last_relocation_generation{};
  RemotePtr last_relocation_expected;
  RemotePtr last_relocation_desired;
  u64 last_relocation_expected_version{};
};

struct AuthorityMutationLease {
  AuthorityOperationToken operation;
  service::storage_owner::MutationKind kind{
    service::storage_owner::MutationKind::insert};
  u32 stage1_home{};
  u32 generation{};
  bool previous_exists{};
  AuthorityDirectoryEntry previous;
};

// A compact, independently testable view of one ID.  The storage-backed
// implementation materializes it from the immutable base map plus the dynamic
// overlay while holding the corresponding freshness-shard mutex.
struct AuthorityDirectoryState {
  bool exists{};
  AuthorityDirectoryEntry entry;
  std::optional<AuthorityMutationLease> lease;
};

enum class AuthorityBeginState : u8 {
  prepared,
  replay,
  committed_replay,
  busy,
  conflict,
  already_exists,
  not_found,
  already_deleted,
};

struct AuthorityBeginResult {
  AuthorityBeginState state{AuthorityBeginState::conflict};
  AuthorityDirectoryEntry previous;
  u32 generation{};
  AuthorityCommittedMutationResult replay_result;

  bool acquired() const {
    return state == AuthorityBeginState::prepared ||
      state == AuthorityBeginState::replay;
  }
};

enum class AuthorityCommitState : u8 {
  committed,
  replay,
  stale,
  conflict,
};

enum class AuthorityAbortState : u8 {
  aborted,
  not_active,
  wrong_operation,
  already_committed,
};

enum class AuthorityCheckState : u8 {
  pending,
  busy,
  current,
  stale,
};

enum class AuthorityRelocateState : u8 {
  committed,
  replay,
  busy,
  stale,
  conflict,
};

// Cleanup is armed while the successor mutation still owns its authority
// lease.  The physical owner must therefore distinguish "the successor has
// not linearized yet" from "that lease was aborted" before quiescing the old
// generation.  A generation strictly newer than retired_generation is a
// durable retirement proof even when another, later lease is already active.
// This keeps cleanup retryable without retaining an unbounded operation log.
inline AuthorityRelocateState check_authority_retirement(
    const AuthorityDirectoryState& state,
    AuthorityOperationToken operation,
    u32 retired_generation,
    RemotePtr retired) {
  if (!valid_authority_operation(operation) || retired.is_null()) {
    return AuthorityRelocateState::conflict;
  }

  if (state.exists && state.entry.generation > retired_generation) {
    return AuthorityRelocateState::committed;
  }

  if (state.lease.has_value()) {
    // This may be the activating successor or a later attempt after the
    // activating lease aborted.  In both cases the old generation is still
    // authoritative, so cleanup must wait rather than guess the outcome.
    return AuthorityRelocateState::busy;
  }

  // With no lease and no newer generation, the activating operation aborted
  // (or the request is stale/malformed).  In particular, never tombstone the
  // still-current physical record merely because cleanup was queued first.
  return AuthorityRelocateState::stale;
}

inline bool same_authority_entry_version(
    const AuthorityDirectoryEntry& lhs,
    const AuthorityDirectoryEntry& rhs) {
  return lhs.current == rhs.current && lhs.generation == rhs.generation &&
    lhs.deleted == rhs.deleted &&
    lhs.placement_version == rhs.placement_version;
}

inline AuthorityDirectoryEntry visible_previous_entry(
    AuthorityDirectoryEntry entry) {
  if (entry.deleted) entry.current.reset();
  return entry;
}

inline AuthorityBeginResult begin_authority_mutation(
    AuthorityDirectoryState& state,
    service::storage_owner::MutationKind kind,
    AuthorityOperationToken operation,
    u32 stage1_home) {
  if (!valid_authority_operation(operation)) {
    return {
      .state = AuthorityBeginState::conflict,
      .previous = {},
      .generation = 0,
      .replay_result = {},
    };
  }
  if (state.exists &&
      same_authority_operation(
        state.entry.last_committed_operation, operation)) {
    if (state.entry.last_committed_kind != kind ||
        state.entry.last_committed_stage1_home != stage1_home) {
      return {
        .state = AuthorityBeginState::conflict,
        .previous = {},
        .generation = 0,
        .replay_result = {},
      };
    }
    return {
      .state = AuthorityBeginState::committed_replay,
      .previous = {},
      .generation = state.entry.last_committed_result.generation,
      .replay_result = state.entry.last_committed_result,
    };
  }
  if (state.lease.has_value()) {
    const AuthorityMutationLease& active = *state.lease;
    if (!same_authority_operation(active.operation, operation)) {
      return {
        .state = AuthorityBeginState::busy,
        .previous = {},
        .generation = 0,
        .replay_result = {},
      };
    }
    if (active.kind != kind || active.stage1_home != stage1_home) {
      return {
        .state = AuthorityBeginState::conflict,
        .previous = {},
        .generation = 0,
        .replay_result = {},
      };
    }
    return {
      .state = AuthorityBeginState::replay,
      .previous = visible_previous_entry(active.previous),
      .generation = active.generation,
      .replay_result = {},
    };
  }

  const bool live = state.exists && !state.entry.deleted;
  switch (kind) {
    case service::storage_owner::MutationKind::insert:
      if (live) {
        return {
          .state = AuthorityBeginState::already_exists,
          .previous = visible_previous_entry(state.entry),
          .generation = state.entry.generation,
          .replay_result = {},
        };
      }
      break;
    case service::storage_owner::MutationKind::upsert:
      break;
    case service::storage_owner::MutationKind::erase:
      if (!state.exists) {
        return {
          .state = AuthorityBeginState::not_found,
          .previous = {},
          .generation = 0,
          .replay_result = {},
        };
      }
      if (!live) {
        return {
          .state = AuthorityBeginState::already_deleted,
          .previous = visible_previous_entry(state.entry),
          .generation = state.entry.generation,
          .replay_result = {},
        };
      }
      break;
    default:
      return {
        .state = AuthorityBeginState::conflict,
        .previous = {},
        .generation = 0,
        .replay_result = {},
      };
  }

  const u32 previous_generation = state.exists ? state.entry.generation : 0;
  if (previous_generation == std::numeric_limits<u32>::max()) {
    return {
      .state = AuthorityBeginState::conflict,
      .previous = {},
      .generation = 0,
      .replay_result = {},
    };
  }
  AuthorityMutationLease lease{
    .operation = operation,
    .kind = kind,
    .stage1_home = stage1_home,
    .generation = previous_generation + 1,
    .previous_exists = state.exists,
    .previous = state.entry,
  };
  const AuthorityBeginResult result{
    .state = AuthorityBeginState::prepared,
    .previous = visible_previous_entry(lease.previous),
    .generation = lease.generation,
    .replay_result = {},
  };
  state.lease = std::move(lease);
  return result;
}

inline AuthorityCommitState commit_authority_mutation(
    AuthorityDirectoryState& state,
    AuthorityOperationToken operation,
    RemotePtr desired,
    u32 generation,
    bool deleted,
    u64 maintenance_sequence) {
  if (!valid_authority_operation(operation) ||
      (!deleted && desired.is_null())) {
    return AuthorityCommitState::conflict;
  }

  if (!state.lease.has_value()) {
    if (state.exists &&
        same_authority_operation(
          state.entry.last_committed_operation, operation)) {
      const RemotePtr public_new_pointer = deleted ? RemotePtr{} : desired;
      const AuthorityCommittedMutationResult& replay =
        state.entry.last_committed_result;
      if (replay.new_pointer == public_new_pointer &&
          replay.generation == generation &&
          replay.maintenance_sequence == maintenance_sequence &&
          state.entry.generation == generation &&
          state.entry.deleted == deleted) {
        return AuthorityCommitState::replay;
      }
      return AuthorityCommitState::conflict;
    }
    return AuthorityCommitState::stale;
  }

  const AuthorityMutationLease& active = *state.lease;
  if (!same_authority_operation(active.operation, operation) ||
      active.generation != generation) {
    return AuthorityCommitState::stale;
  }
  if (state.exists != active.previous_exists ||
      (state.exists &&
       !same_authority_entry_version(state.entry, active.previous))) {
    return AuthorityCommitState::stale;
  }
  if (active.previous.placement_version ==
      std::numeric_limits<u64>::max()) {
    return AuthorityCommitState::conflict;
  }

  state.exists = true;
  state.entry = AuthorityDirectoryEntry{
    .current = desired,
    .generation = generation,
    .deleted = deleted,
    .placement_version = active.previous.placement_version + 1,
    .last_committed_operation = operation,
    .last_committed_kind = active.kind,
    .last_committed_stage1_home = active.stage1_home,
    .last_committed_result = {
      .new_pointer = deleted ? RemotePtr{} : desired,
      .old_pointer = visible_previous_entry(active.previous).current,
      .generation = generation,
      .maintenance_sequence = maintenance_sequence,
    },
    .last_relocation_operation = {},
    .last_relocation_generation = 0,
    .last_relocation_expected = {},
    .last_relocation_desired = {},
    .last_relocation_expected_version = 0,
  };
  state.lease.reset();
  return AuthorityCommitState::committed;
}

inline AuthorityAbortState abort_authority_mutation(
    AuthorityDirectoryState& state,
    AuthorityOperationToken operation) {
  if (!valid_authority_operation(operation)) {
    return AuthorityAbortState::wrong_operation;
  }
  if (!state.lease.has_value()) {
    if (state.exists &&
        same_authority_operation(
          state.entry.last_committed_operation, operation)) {
      return AuthorityAbortState::already_committed;
    }
    return AuthorityAbortState::not_active;
  }
  if (!same_authority_operation(state.lease->operation, operation)) {
    return AuthorityAbortState::wrong_operation;
  }
  state.lease.reset();
  return AuthorityAbortState::aborted;
}

inline AuthorityCheckState check_authority_current(
    const AuthorityDirectoryState& state,
    AuthorityOperationToken operation,
    u32 generation,
    RemotePtr expected,
    u64 expected_placement_version) {
  if (state.lease.has_value()) {
    const AuthorityMutationLease& active = *state.lease;
    return same_authority_operation(active.operation, operation) &&
        active.generation == generation
      ? AuthorityCheckState::pending
      : AuthorityCheckState::busy;
  }
  if (!state.exists || state.entry.deleted ||
      state.entry.generation != generation ||
      state.entry.current != expected ||
      state.entry.placement_version != expected_placement_version) {
    return AuthorityCheckState::stale;
  }
  return AuthorityCheckState::current;
}

inline AuthorityRelocateState relocate_authority_if_current(
    AuthorityDirectoryState& state,
    AuthorityOperationToken operation,
    u32 generation,
    RemotePtr expected,
    RemotePtr desired,
    u64 expected_placement_version) {
  if (desired.is_null()) {
    return check_authority_retirement(
      state, operation, generation, expected);
  }
  if (!valid_authority_operation(operation) || expected.is_null() ||
      (expected != desired &&
       expected_placement_version == std::numeric_limits<u64>::max())) {
    return AuthorityRelocateState::conflict;
  }
  // Begin and relocate share one authority-shard mutex.  Consequently this is
  // the decisive race check: a successor that captured expected must settle
  // before an older Stage2 may change that physical pointer.
  if (state.lease.has_value()) return AuthorityRelocateState::busy;
  if (!state.exists || state.entry.deleted ||
      state.entry.generation != generation) {
    return AuthorityRelocateState::stale;
  }

  // expected == desired is a read-only authority barrier used by an armed
  // Stage2 task. It proves that the foreground mutation has committed and
  // that no successor lease is active, but must not consume (or conflict
  // with) the operation's relocation replay receipt.
  if (expected == desired) {
    return state.entry.current == expected &&
        state.entry.placement_version == expected_placement_version
      ? AuthorityRelocateState::committed
      : AuthorityRelocateState::stale;
  }

  if (same_authority_operation(
        state.entry.last_relocation_operation, operation)) {
    if (state.entry.last_relocation_generation != generation ||
        state.entry.last_relocation_expected != expected ||
        state.entry.last_relocation_desired != desired ||
        state.entry.last_relocation_expected_version !=
          expected_placement_version) {
      return AuthorityRelocateState::conflict;
    }
    const u64 resulting_version = expected_placement_version +
      static_cast<u64>(expected != desired);
    return state.entry.current == desired &&
        state.entry.placement_version == resulting_version
      ? AuthorityRelocateState::replay
      : AuthorityRelocateState::stale;
  }
  if (state.entry.current != expected ||
      state.entry.placement_version != expected_placement_version) {
    return AuthorityRelocateState::stale;
  }

  state.entry.current = desired;
  ++state.entry.placement_version;
  state.entry.last_relocation_operation = operation;
  state.entry.last_relocation_generation = generation;
  state.entry.last_relocation_expected = expected;
  state.entry.last_relocation_desired = desired;
  state.entry.last_relocation_expected_version =
    expected_placement_version;
  return AuthorityRelocateState::committed;
}

}  // namespace memory_node_storage_owner_index_detail
