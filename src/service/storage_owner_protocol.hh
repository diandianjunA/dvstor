#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "common/types.hh"
#include "vamana/vamana_node.hh"

namespace service::storage_owner {

constexpr u32 kInsertMagic = 0x53494e54;  // "SINT"
constexpr u32 kMutationMagic = 0x4d555444;  // D T U M / "DUTM"
constexpr u32 kMutationCompletionMagic = 0x4d434d50;  // "MCMP"
// Version 4 carries a stable operation id for every logical mutation.  Batch
// ids and array ordinals are transport metadata and are no longer authority
// replay identities.
constexpr u32 kMutationProtocolVersion = 4;
constexpr u32 kPeerRpcMagic = 0x53505250;  // "SPRP"
constexpr u32 kPeerRpcVersion = 14;

enum class InsertStatus : u32 {
  ok = 0,
  failed = 1,
  overloaded = 2,
};

enum class MutationKind : u32 {
  insert = 1,
  upsert = 2,
  erase = 3,
};

enum class MutationStatus : u32 {
  ok = 0,
  not_found = 1,
  already_exists = 2,
  already_deleted = 3,
  failed = 4,
  // Internal Stage1 backpressure/duplicate-in-progress signal. It is never a
  // public mutation result: the authority must replay the same semantic token
  // until it observes either ok or a definitive pre-arm failure.
  retry = 5,
};

enum class PeerRpcType : u32 {
  reverse_update_request = 1,
  reverse_update_response = 2,
  cleanup_deleted_request = 3,
  cleanup_deleted_response = 6,
  reconcile_reverse_request = 7,
  reconcile_reverse_response = 8,
  centroid_membership_request = 9,
  centroid_membership_response = 10,
  stage1_execute_request = 11,
  stage1_execute_response = 12,
  stage1_arm_request = 13,
  stage1_arm_response = 14,
  cleanup_activate_request = 15,
  cleanup_activate_response = 16,
  authority_placement_request = 17,
  authority_placement_response = 18,
  dynamic_node_control_request = 19,
  dynamic_node_control_response = 20,
  // Stage2 keeps beam ownership at the insertion home.  A request expands
  // exactly the pointer selected by that beam and scores only neighbors whose
  // vectors are resident at the expansion pointer's physical home.  It is a
  // transport fusion, not a nested/local search and therefore cannot change
  // the natural convergence order.
  stage2_expand_score_request = 21,
  stage2_expand_score_response = 22,
};

enum class Stage2HomeDisposition : u32 {
  retryable = 0,
  stable = 1,
  terminal = 2,
  // The neighbor is valid but belongs to another physical home.  The beam
  // owner retains its ordinary score request for that pointer.
  unscored = 3,
};

enum class Stage2HomeOperation : u32 {
  expand_score = 0,
  score_only = 1,
};

struct Stage2ExpandScoreItem {
  u64 pointer_raw{};
  u64 generation{};
  u32 search_index{};
  u32 operation{static_cast<u32>(Stage2HomeOperation::expand_score)};
};

struct Stage2ExpandScoreResult {
  u64 pointer_raw{};
  u64 generation{};
  u32 search_index{};
  u32 neighbor_count{};
  u32 disposition{static_cast<u32>(Stage2HomeDisposition::retryable)};
  // Offset in the compact neighbor array that follows all result records.
  // Version 13 used a fixed graph-capacity stride here, which made a batch of
  // 32 responses consume more than 50 KiB even when nodes had few live edges.
  u32 neighbor_offset{};
  distance_t distance{};
  u32 operation{static_cast<u32>(Stage2HomeOperation::expand_score)};
};

struct Stage2ExpandScoreNeighbor {
  u64 pointer_raw{};
  distance_t distance{};
  u32 disposition{static_cast<u32>(Stage2HomeDisposition::unscored)};
};

static_assert(sizeof(Stage2ExpandScoreItem) == 24);
static_assert(sizeof(Stage2ExpandScoreResult) == 40);
static_assert(sizeof(Stage2ExpandScoreNeighbor) == 16);

struct InsertBatchRequestHeader {
  u32 magic{kInsertMagic};
  u32 dim{};
  u32 owner_storage{};
  u32 source_client{};
  u32 item_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 protocol_version{kMutationProtocolVersion};
  u64 batch_id{};
};

struct MutationBatchRequestHeader {
  u32 magic{kMutationMagic};
  u32 dim{};
  u32 owner_storage{};
  u32 source_client{};
  u32 item_count{};
  u32 vector_dtype{};
  u32 vector_bytes{};
  u32 protocol_version{kMutationProtocolVersion};
  u64 batch_id{};
};

static_assert(sizeof(InsertBatchRequestHeader) == 40);
static_assert(sizeof(MutationBatchRequestHeader) == 40);
static_assert(offsetof(InsertBatchRequestHeader, protocol_version) == 28);
static_assert(offsetof(MutationBatchRequestHeader, protocol_version) == 28);
static_assert(offsetof(InsertBatchRequestHeader, batch_id) == 32);
static_assert(offsetof(MutationBatchRequestHeader, batch_id) == 32);

struct InsertBatchResponseHeader {
  u32 magic{kInsertMagic};
  u32 owner_storage{};
  u32 item_count{};
  u32 reserved{};
  u64 batch_id{};
};

enum class MutationBatchAckStatus : u32 {
  accepted = 0,
  busy = 1,
  malformed = 2,
};

// Transport acceptance is deliberately separate from logical completion.
// Once accepted, every item owns completion credit and may finish out of
// order through MutationCompletionV2.
struct MutationBatchAckV2 {
  u32 magic{};
  u32 owner_storage{};
  u32 item_count{};
  u32 status{static_cast<u32>(MutationBatchAckStatus::malformed)};
  u32 protocol_version{kMutationProtocolVersion};
  u32 reserved{};
  u64 batch_id{};
};

struct MutationCompletionV2 {
  u32 magic{kMutationCompletionMagic};
  u32 protocol_version{kMutationProtocolVersion};
  u32 owner_storage{};
  u32 source_client{};
  u64 operation_id{};
  u64 new_rptr_raw{};
  u64 old_rptr_raw{};
  u64 maintenance_sequence{};
  u32 generation{};
  u32 status{static_cast<u32>(MutationStatus::failed)};
  u64 reserved{};
};

static_assert(sizeof(MutationBatchAckV2) == 32);
static_assert(sizeof(MutationCompletionV2) == 64);

// ACKs and completions share one RC QP. RC receive WQEs are consumed in queue
// order rather than selected by WR-ID, so every posted receive on that QP must
// accept the largest envelope.
inline constexpr size_t mutation_receive_slot_bytes() {
  return sizeof(MutationCompletionV2);
}
static_assert(sizeof(MutationBatchAckV2) <= mutation_receive_slot_bytes());

struct MutationResult {
  u64 new_rptr_raw{};
  u64 old_rptr_raw{};
  u32 generation{};
  u32 reserved{};
  u64 maintenance_sequence{};
};

struct InsertBreakdownCounters {
  u64 storage_owner_queue_wait_ns{};
  u64 storage_owner_stage1_execute_wait_ns{};
  u64 storage_owner_search_ns{};
  u64 storage_owner_prune_ns{};
  u64 storage_owner_write_node_ns{};
  u64 storage_owner_local_reverse_ns{};
  u64 storage_owner_remote_reverse_ns{};
  u64 storage_owner_peer_reverse_apply_ns{};
  u64 storage_owner_response_send_ns{};
  u64 storage_owner_prepare_mutation_ns{};
  u64 storage_owner_allocate_node_ns{};
  u64 storage_owner_publish_mutation_ns{};
  u64 storage_owner_schedule_maintenance_ns{};
  u64 storage_owner_response_build_ns{};

  u64 storage_owner_search_select_ns{};
  u64 storage_owner_search_neighbor_read_ns{};
  u64 storage_owner_search_snapshot_read_ns{};
  u64 storage_owner_search_distance_ns{};
  u64 storage_owner_search_beam_update_ns{};
  u64 storage_owner_search_result_sort_ns{};
  u64 storage_owner_prune_snapshot_read_ns{};
  u64 storage_owner_prune_distance_ns{};
  u64 storage_owner_prune_sort_ns{};
  u64 storage_owner_prune_pair_distance_ns{};

  u64 storage_owner_stage1_arm_wait_ns{};
  u64 storage_owner_stage1_release_wait_ns{};
  u64 storage_owner_cleanup_control_wait_ns{};
  u64 reserved_word{};

  u64 total() const {
    return storage_owner_queue_wait_ns +
           storage_owner_stage1_execute_wait_ns +
           storage_owner_search_ns +
           storage_owner_prune_ns +
           storage_owner_write_node_ns +
           storage_owner_local_reverse_ns +
           storage_owner_remote_reverse_ns +
           storage_owner_peer_reverse_apply_ns +
           storage_owner_response_send_ns +
           storage_owner_prepare_mutation_ns +
           storage_owner_allocate_node_ns +
           storage_owner_publish_mutation_ns +
           storage_owner_schedule_maintenance_ns +
           storage_owner_response_build_ns +
           storage_owner_stage1_arm_wait_ns +
           storage_owner_stage1_release_wait_ns +
           storage_owner_cleanup_control_wait_ns;
  }
};

static_assert(sizeof(InsertBreakdownCounters) == 224);
static_assert(offsetof(InsertBreakdownCounters,
                       storage_owner_stage1_arm_wait_ns) == 192);

struct PeerRpcHeader {
  u32 magic{kPeerRpcMagic};
  u32 version{kPeerRpcVersion};
  u32 type{};
  u32 source_shard{};
  u32 item_count{};
  u64 request_id{};
  u32 status{static_cast<u32>(InsertStatus::failed)};
  u32 reserved{};
};

static_assert(sizeof(PeerRpcHeader) == 40);
static_assert(offsetof(PeerRpcHeader, request_id) == 24);

// Stable identity of one public mutation across transport retries. The tuple
// is owned by the logical authority and must never be regenerated by a
// physical home.
struct AuthorityOperationToken {
  u32 source_client{};
  u32 item_index{};
  u64 client_batch_id{};
};

static_assert(sizeof(AuthorityOperationToken) == 16);
static_assert(offsetof(AuthorityOperationToken, source_client) == 0);
static_assert(offsetof(AuthorityOperationToken, item_index) == 4);
static_assert(offsetof(AuthorityOperationToken, client_batch_id) == 8);

struct ReverseUpdateOp {
  u64 target_raw{};
  u64 candidate_raw{};
  node_t target_id{};
  u32 target_generation{};
  node_t candidate_id{};
  u32 candidate_generation{};
};

static_assert(sizeof(ReverseUpdateOp) == 32);
static_assert(offsetof(ReverseUpdateOp, target_raw) == 0);
static_assert(offsetof(ReverseUpdateOp, candidate_raw) == 8);
static_assert(offsetof(ReverseUpdateOp, target_id) == 16);
static_assert(offsetof(ReverseUpdateOp, target_generation) == 20);
static_assert(offsetof(ReverseUpdateOp, candidate_id) == 24);
static_assert(offsetof(ReverseUpdateOp, candidate_generation) == 28);

// Idempotent handoff of one logical candidate between physical placements.
enum class ReconcileReverseOpKind : u32 {
  replace_or_add = 1,
  remove_if_present = 2,
  add = 3,
  // Transfers/retains an acknowledged Stage1 protected-backlink slot, or
  // reserves an actually empty protected slot for deletion-safe reparenting
  // when old_candidate_raw is zero. It never evicts stable/protected work.
  ensure_reachable = 4,
  // Stage2's single bounded reachability certificate. If old_candidate_raw
  // names a Stage1 protected edge, atomically consumes that edge while
  // publishing new_candidate_raw in the R-bounded stable plane. When ordinary
  // RobustPrune rejects the candidate, only this operation may replace its
  // lowest-priority survivor. The promoted edge is ordinary stable state and
  // may be pruned by later graph maintenance.
  promote_stable_bridge = 5,
};

struct ReconcileReverseOp {
  u64 target_raw{};
  u64 old_candidate_raw{};
  u64 new_candidate_raw{};
  u64 placement_sequence{};
  node_t id{};
  u32 generation{};
  u32 kind{};
  u32 reserved{};
};

// These fields describe postconditions, except replaced which reports that
// this invocation consumed an old-pointer slot.  In particular, removed is
// also true for an idempotent retry that observes old_candidate_raw absent,
// and accepted means new_candidate_raw is present after reconciliation.
struct ReconcileReverseResult {
  u64 placement_sequence{};
  u8 accepted{};
  u8 replaced{};
  u8 removed{};
  u8 stale{};
  u32 reserved{};
};

static_assert(sizeof(ReconcileReverseOp) == 48);
static_assert(sizeof(ReconcileReverseResult) == 16);

enum class CentroidMembershipKind : u32 {
  add = 1,
  remove = 2,
};

struct CentroidMembershipOp {
  u64 node_raw{};
  u64 maintenance_sequence{};
  node_t id{};
  u32 generation{};
  u32 kind{};
  u32 reserved{};
};

static_assert(sizeof(CentroidMembershipOp) == 32);

// Stable authority sends this bounded request to the centroid-selected
// physical home. The vector bytes follow the item array in storage dtype.
// (source_client, client_batch_id, item_index) is the public-operation token;
// retries must preserve it.
struct Stage1ExecuteItem {
  u64 client_batch_id{};
  u64 old_raw{};
  // A non-zero version asks the physical home to atomically turn a successful
  // prepare into a runnable Stage2 task before replying. Inserts have no old
  // generation cleanup dependency and therefore use this fused path; upserts
  // keep it zero until cleanup activation has completed.
  u64 initial_placement_version{};
  u32 source_client{};
  u32 item_index{};
  node_t id{};
  u32 generation{};
  u32 kind{};
  u32 authority_shard{};
};

struct Stage1ExecuteResult {
  u64 client_batch_id{};
  u64 target_raw{};
  // Non-zero only after the fused prepare+arm path owns a runnable bounded
  // maintenance descriptor. The authority must never commit without it.
  u64 maintenance_sequence{};
  u32 source_client{};
  u32 item_index{};
  u32 status{static_cast<u32>(MutationStatus::failed)};
  u32 reserved{};
};

enum class Stage1ArmAction : u32 {
  arm = 1,
  abort = 2,
  // Explicitly releases the semantic replay receipt after the authority has
  // observed arm/abort and reached its corresponding commit/abort boundary.
  // The physical home ACKs a missing receipt as an idempotent replay.
  release = 3,
};

struct Stage1ArmItem {
  AuthorityOperationToken token{};
  u64 target_raw{};
  u64 initial_placement_version{};
  node_t id{};
  u32 generation{};
  u32 action{static_cast<u32>(Stage1ArmAction::arm)};
  u32 reserved{};
};

// A standalone arm allocates a Stage2 maintenance sequence for upserts after
// old-generation cleanup is active. Pure inserts normally allocate the same
// fence in fused Stage1 execute. Either response is durable proof that the
// sequence already belongs to a runnable bounded task.
struct Stage1ArmResult {
  AuthorityOperationToken token{};
  u64 target_raw{};
  u64 maintenance_sequence{};
  u32 status{static_cast<u32>(MutationStatus::failed)};
  u32 reserved{};
};

static_assert(sizeof(Stage1ExecuteItem) == 48);
static_assert(offsetof(Stage1ExecuteItem, client_batch_id) == 0);
static_assert(offsetof(Stage1ExecuteItem, old_raw) == 8);
static_assert(offsetof(Stage1ExecuteItem, initial_placement_version) == 16);
static_assert(offsetof(Stage1ExecuteItem, source_client) == 24);
static_assert(offsetof(Stage1ExecuteItem, id) == 32);
static_assert(offsetof(Stage1ExecuteItem, authority_shard) == 44);
static_assert(sizeof(Stage1ExecuteResult) == 40);
static_assert(offsetof(Stage1ExecuteResult, client_batch_id) == 0);
static_assert(offsetof(Stage1ExecuteResult, target_raw) == 8);
static_assert(offsetof(Stage1ExecuteResult, maintenance_sequence) == 16);
static_assert(offsetof(Stage1ExecuteResult, source_client) == 24);
static_assert(offsetof(Stage1ExecuteResult, status) == 32);
static_assert(sizeof(Stage1ArmItem) == 48);
static_assert(offsetof(Stage1ArmItem, token) == 0);
static_assert(offsetof(Stage1ArmItem, target_raw) == 16);
static_assert(offsetof(Stage1ArmItem, initial_placement_version) == 24);
static_assert(offsetof(Stage1ArmItem, id) == 32);
static_assert(offsetof(Stage1ArmItem, generation) == 36);
static_assert(offsetof(Stage1ArmItem, action) == 40);
static_assert(sizeof(Stage1ArmResult) == 40);
static_assert(offsetof(Stage1ArmResult, token) == 0);
static_assert(offsetof(Stage1ArmResult, target_raw) == 16);
static_assert(offsetof(Stage1ArmResult, maintenance_sequence) == 24);
static_assert(offsetof(Stage1ArmResult, status) == 32);

enum class CleanupActivateAction : u32 {
  activate = 1,
  // Releases the activation replay receipt after the authority has consumed
  // its result and committed or aborted the successor lease.
  release = 2,
};

// While holding the next-generation authority lease, the authority asks the
// previous physical home to activate cleanup for the old generation. The new
// Stage1 record is already query-reachable, and the authority commits only
// after this idempotent activation has a maintenance fence. `release` uses the
// same identity and is ordered after the final activate retry on the RC control
// QP, so the receiver can reclaim the replay record without a time window.
struct CleanupActivateItem {
  AuthorityOperationToken token{};
  u64 old_raw{};
  node_t id{};
  u32 old_generation{};
  u32 authority_shard{};
  u32 action{static_cast<u32>(CleanupActivateAction::activate)};
};

struct CleanupActivateResult {
  u64 target_raw{};
  u64 maintenance_sequence{};
  AuthorityOperationToken token{};
  u32 status{static_cast<u32>(MutationStatus::failed)};
  u32 reserved{};
};

static_assert(sizeof(CleanupActivateItem) == 40);
static_assert(offsetof(CleanupActivateItem, token) == 0);
static_assert(offsetof(CleanupActivateItem, old_raw) == 16);
static_assert(offsetof(CleanupActivateItem, id) == 24);
static_assert(offsetof(CleanupActivateItem, old_generation) == 28);
static_assert(offsetof(CleanupActivateItem, authority_shard) == 32);
static_assert(offsetof(CleanupActivateItem, action) == 36);
static_assert(sizeof(CleanupActivateResult) == 40);
static_assert(offsetof(CleanupActivateResult, target_raw) == 0);
static_assert(offsetof(CleanupActivateResult, maintenance_sequence) == 8);
static_assert(offsetof(CleanupActivateResult, token) == 16);
static_assert(offsetof(CleanupActivateResult, status) == 32);

enum class AuthorityPlacementStatus : u32 {
  committed = 0,
  replay = 1,
  busy = 2,
  stale = 3,
  conflict = 4,
};

// A physical Stage2 home sends this compare-and-swap style request to the
// stable logical authority. expected_raw/version fence the directory state;
// a non-zero desired_raw is installed without changing the logical
// generation. Cleanup uses desired_raw == 0 as a read-only retirement barrier:
// committed means a strictly newer logical generation has retired
// expected_raw, busy means an authority lease is unresolved, and stale means
// the activating lease aborted while expected_raw remained authoritative.
struct AuthorityPlacementItem {
  AuthorityOperationToken token{};
  node_t id{};
  u32 generation{};
  u64 expected_raw{};
  u64 desired_raw{};
  u64 expected_placement_version{};
};

struct AuthorityPlacementResult {
  u64 resulting_placement_version{};
  u32 status{static_cast<u32>(AuthorityPlacementStatus::conflict)};
  u32 reserved{};
};

static_assert(sizeof(AuthorityPlacementItem) == 48);
static_assert(offsetof(AuthorityPlacementItem, token) == 0);
static_assert(offsetof(AuthorityPlacementItem, id) == 16);
static_assert(offsetof(AuthorityPlacementItem, generation) == 20);
static_assert(offsetof(AuthorityPlacementItem, expected_raw) == 24);
static_assert(offsetof(AuthorityPlacementItem, desired_raw) == 32);
static_assert(offsetof(AuthorityPlacementItem,
                       expected_placement_version) == 40);
static_assert(sizeof(AuthorityPlacementResult) == 16);
static_assert(offsetof(AuthorityPlacementResult,
                       resulting_placement_version) == 0);
static_assert(offsetof(AuthorityPlacementResult, status) == 8);

enum class DynamicNodeControlAction : u32 {
  allocate = 1,
  retire = 2,
  settle_allocation = 3,
};

enum class DynamicNodeControlStatus : u32 {
  ok = 0,
  stale = 1,
  failed = 2,
};

// Allocation is performed by the physical owner so both local Stage1 and a
// remote Stage2 migration consume the same RCU-safe reclaimed-slot pool.
// For allocate, node_raw is the original Stage1 record and makes retry-token
// conflicts detectable. For retire, node_raw is the exact physical record to
// tombstone and reclaim. settle_allocation releases the bounded owner receipt
// only after tagged physical identity proves source and destination handoff
// postconditions; it never relies on a replay timeout.
struct DynamicNodeControlItem {
  AuthorityOperationToken token{};
  // allocate: the Stage1 source whose live incarnation authorizes exactly one
  // reservation. retire: the physical record to retire. settle_allocation:
  // the original Stage1 source, paired with allocated_raw below.
  u64 node_raw{};
  // Only settle_allocation uses this field. It identifies the exact reserved
  // destination incarnation whose receipt may be released.
  u64 allocated_raw{};
  node_t id{};
  u32 generation{};
  u32 authority_shard{};
  u32 action{};
};

struct DynamicNodeControlResult {
  u64 node_raw{};
  u64 maintenance_sequence{};
  u32 status{static_cast<u32>(DynamicNodeControlStatus::failed)};
  u32 reserved{};
};

static_assert(sizeof(DynamicNodeControlItem) == 48);
static_assert(offsetof(DynamicNodeControlItem, token) == 0);
static_assert(offsetof(DynamicNodeControlItem, node_raw) == 16);
static_assert(offsetof(DynamicNodeControlItem, allocated_raw) == 24);
static_assert(offsetof(DynamicNodeControlItem, id) == 32);
static_assert(offsetof(DynamicNodeControlItem, generation) == 36);
static_assert(offsetof(DynamicNodeControlItem, authority_shard) == 40);
static_assert(offsetof(DynamicNodeControlItem, action) == 44);
static_assert(sizeof(DynamicNodeControlResult) == 24);
static_assert(offsetof(DynamicNodeControlResult, node_raw) == 0);
static_assert(offsetof(DynamicNodeControlResult,
                       maintenance_sequence) == 8);
static_assert(offsetof(DynamicNodeControlResult, status) == 16);

constexpr size_t wire_saturating_add(size_t lhs, size_t rhs) {
  return rhs > std::numeric_limits<size_t>::max() - lhs
    ? std::numeric_limits<size_t>::max() : lhs + rhs;
}

constexpr size_t wire_saturating_multiply(size_t lhs, size_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs
    ? std::numeric_limits<size_t>::max() : lhs * rhs;
}

constexpr size_t align_wire_u64(size_t value) {
  constexpr size_t mask = alignof(u64) - 1;
  if (value > std::numeric_limits<size_t>::max() - mask) {
    return std::numeric_limits<size_t>::max();
  }
  return (value + mask) & ~mask;
}

static_assert(align_wire_u64(1) == 8);
static_assert(align_wire_u64(8) == 8);

inline size_t insert_batch_request_bytes(u32 item_count) {
  size_t bytes = sizeof(InsertBatchRequestHeader);
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(node_t)));
  if ((bytes & (alignof(u64) - 1)) != 0) {
    bytes = wire_saturating_add(bytes, sizeof(u32));
  }
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(u64)));
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(u32)));
  return wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, VamanaNode::vector_bytes()));
}

inline size_t mutation_batch_request_bytes(u32 item_count) {
  size_t bytes = sizeof(MutationBatchRequestHeader);
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(u32)));
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(node_t)));
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(u64)));
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(u32)));
  return wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, VamanaNode::vector_bytes()));
}

inline size_t insert_batch_response_bytes(u32 item_count) {
  size_t bytes = sizeof(InsertBatchResponseHeader);
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(u32)));
  bytes = wire_saturating_add(bytes, wire_saturating_multiply(
    item_count, sizeof(MutationResult)));
  bytes = wire_saturating_add(bytes, sizeof(InsertBreakdownCounters));
  bytes = wire_saturating_add(bytes, sizeof(u32));
  const size_t invalidations = wire_saturating_multiply(
    item_count, VamanaNode::R);
  return wire_saturating_add(bytes, wire_saturating_multiply(
    invalidations, sizeof(u64)));
}

inline node_t* request_ids(void* payload) {
  return reinterpret_cast<node_t*>(reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchRequestHeader));
}

inline const node_t* request_ids(const void* payload) {
  return reinterpret_cast<const node_t*>(reinterpret_cast<const byte_t*>(payload) + sizeof(InsertBatchRequestHeader));
}

inline u64* request_operation_ids(void* payload, u32 item_count) {
  byte_t* address = reinterpret_cast<byte_t*>(request_ids(payload) +
                                              item_count);
  const uintptr_t raw = reinterpret_cast<uintptr_t>(address);
  const uintptr_t aligned = (raw + alignof(u64) - 1) &
                            ~(static_cast<uintptr_t>(alignof(u64) - 1));
  return reinterpret_cast<u64*>(aligned);
}

inline const u64* request_operation_ids(const void* payload,
                                        u32 item_count) {
  const byte_t* address = reinterpret_cast<const byte_t*>(
    request_ids(payload) + item_count);
  const uintptr_t raw = reinterpret_cast<uintptr_t>(address);
  const uintptr_t aligned = (raw + alignof(u64) - 1) &
                            ~(static_cast<uintptr_t>(alignof(u64) - 1));
  return reinterpret_cast<const u64*>(aligned);
}

inline u32* request_stage1_homes(void* payload, u32 item_count) {
  return reinterpret_cast<u32*>(request_operation_ids(payload, item_count) +
                                item_count);
}

inline const u32* request_stage1_homes(const void* payload,
                                       u32 item_count) {
  return reinterpret_cast<const u32*>(
    request_operation_ids(payload, item_count) + item_count);
}

inline u32* mutation_request_kinds(void* payload) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(payload) + sizeof(MutationBatchRequestHeader));
}

inline const u32* mutation_request_kinds(const void* payload) {
  return reinterpret_cast<const u32*>(reinterpret_cast<const byte_t*>(payload) + sizeof(MutationBatchRequestHeader));
}

inline node_t* mutation_request_ids(void* payload) {
  return reinterpret_cast<node_t*>(mutation_request_kinds(payload) +
                                   reinterpret_cast<MutationBatchRequestHeader*>(payload)->item_count);
}

inline const node_t* mutation_request_ids(const void* payload) {
  return reinterpret_cast<const node_t*>(mutation_request_kinds(payload) +
                                         reinterpret_cast<const MutationBatchRequestHeader*>(payload)->item_count);
}

inline u64* mutation_request_operation_ids(void* payload, u32 item_count) {
  return reinterpret_cast<u64*>(mutation_request_ids(payload) + item_count);
}

inline const u64* mutation_request_operation_ids(const void* payload,
                                                  u32 item_count) {
  return reinterpret_cast<const u64*>(
    mutation_request_ids(payload) + item_count);
}

inline u32* mutation_request_stage1_homes(void* payload, u32 item_count) {
  return reinterpret_cast<u32*>(
    mutation_request_operation_ids(payload, item_count) + item_count);
}

inline const u32* mutation_request_stage1_homes(const void* payload,
                                                 u32 item_count) {
  return reinterpret_cast<const u32*>(
    mutation_request_operation_ids(payload, item_count) + item_count);
}

inline byte_t* mutation_request_vectors(void* payload, u32 item_count) {
  return reinterpret_cast<byte_t*>(
    mutation_request_stage1_homes(payload, item_count) + item_count);
}

inline const byte_t* mutation_request_vectors(const void* payload, u32 item_count) {
  return reinterpret_cast<const byte_t*>(
    mutation_request_stage1_homes(payload, item_count) + item_count);
}

inline byte_t* request_vectors(void* payload, u32 item_count) {
  return reinterpret_cast<byte_t*>(
    request_stage1_homes(payload, item_count) + item_count);
}

inline const byte_t* request_vectors(const void* payload, u32 item_count) {
  return reinterpret_cast<const byte_t*>(
    request_stage1_homes(payload, item_count) + item_count);
}

inline byte_t* request_vector(void* payload, u32 item_count, u32 index) {
  return request_vectors(payload, item_count) + static_cast<size_t>(index) * VamanaNode::vector_bytes();
}

inline const byte_t* request_vector(const void* payload, u32 item_count, u32 index) {
  return request_vectors(payload, item_count) + static_cast<size_t>(index) * VamanaNode::vector_bytes();
}

inline u32* response_statuses(void* payload) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline const u32* response_statuses(const void* payload) {
  return reinterpret_cast<const u32*>(reinterpret_cast<const byte_t*>(payload) + sizeof(InsertBatchResponseHeader));
}

inline MutationResult* response_mutation_results(void* payload, u32 item_count) {
  return reinterpret_cast<MutationResult*>(response_statuses(payload) + item_count);
}

inline const MutationResult* response_mutation_results(const void* payload, u32 item_count) {
  return reinterpret_cast<const MutationResult*>(response_statuses(payload) + item_count);
}

inline InsertBreakdownCounters* response_breakdown(void* payload, u32 item_count) {
  return reinterpret_cast<InsertBreakdownCounters*>(
    reinterpret_cast<byte_t*>(response_mutation_results(payload, item_count) + item_count));
}

inline const InsertBreakdownCounters* response_breakdown(const void* payload, u32 item_count) {
  return reinterpret_cast<const InsertBreakdownCounters*>(
    reinterpret_cast<const byte_t*>(response_mutation_results(payload, item_count) + item_count));
}

inline u32* response_invalidation_count(void* payload, u32 item_count) {
  return reinterpret_cast<u32*>(reinterpret_cast<byte_t*>(response_breakdown(payload, item_count) + 1));
}

inline const u32* response_invalidation_count(const void* payload, u32 item_count) {
  return reinterpret_cast<const u32*>(reinterpret_cast<const byte_t*>(response_breakdown(payload, item_count) + 1));
}

inline u64* response_invalidated_raws(void* payload, u32 item_count) {
  return reinterpret_cast<u64*>(response_invalidation_count(payload, item_count) + 1);
}

inline const u64* response_invalidated_raws(const void* payload, u32 item_count) {
  return reinterpret_cast<const u64*>(response_invalidation_count(payload, item_count) + 1);
}

inline u32 response_invalidation_capacity(u32 item_count) {
  return item_count * VamanaNode::R;
}

inline size_t reverse_update_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) * sizeof(ReverseUpdateOp);
}

inline size_t reverse_update_response_bytes() {
  return sizeof(PeerRpcHeader);
}

inline size_t reconcile_reverse_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) +
         static_cast<size_t>(item_count) * sizeof(ReconcileReverseOp);
}

inline size_t reconcile_reverse_response_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(ReconcileReverseResult);
}

inline ReconcileReverseResult* reconcile_reverse_results(void* payload) {
  return reinterpret_cast<ReconcileReverseResult*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const ReconcileReverseResult* reconcile_reverse_results(
    const void* payload) {
  return reinterpret_cast<const ReconcileReverseResult*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline size_t centroid_membership_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(CentroidMembershipOp);
}

inline size_t centroid_membership_response_bytes() {
  return sizeof(PeerRpcHeader);
}

inline size_t stage1_execute_vectors_offset(u32 item_count) {
  return align_wire_u64(
    wire_saturating_add(sizeof(PeerRpcHeader), wire_saturating_multiply(
      item_count, sizeof(Stage1ExecuteItem))));
}

inline size_t stage1_execute_request_bytes(u32 item_count) {
  return wire_saturating_add(
    stage1_execute_vectors_offset(item_count), wire_saturating_multiply(
      item_count, VamanaNode::vector_bytes()));
}

inline size_t stage1_execute_response_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(Stage1ExecuteResult);
}

inline Stage1ExecuteItem* stage1_execute_items(void* payload) {
  return reinterpret_cast<Stage1ExecuteItem*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const Stage1ExecuteItem* stage1_execute_items(
    const void* payload) {
  return reinterpret_cast<const Stage1ExecuteItem*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline byte_t* stage1_execute_vectors(void* payload, u32 item_count) {
  return reinterpret_cast<byte_t*>(payload) +
         stage1_execute_vectors_offset(item_count);
}

inline const byte_t* stage1_execute_vectors(
    const void* payload, u32 item_count) {
  return reinterpret_cast<const byte_t*>(payload) +
         stage1_execute_vectors_offset(item_count);
}

inline Stage1ExecuteResult* stage1_execute_results(void* payload) {
  return reinterpret_cast<Stage1ExecuteResult*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const Stage1ExecuteResult* stage1_execute_results(
    const void* payload) {
  return reinterpret_cast<const Stage1ExecuteResult*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline size_t stage1_arm_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(Stage1ArmItem);
}

inline size_t stage1_arm_response_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(Stage1ArmResult);
}

inline Stage1ArmItem* stage1_arm_items(void* payload) {
  return reinterpret_cast<Stage1ArmItem*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const Stage1ArmItem* stage1_arm_items(const void* payload) {
  return reinterpret_cast<const Stage1ArmItem*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline Stage1ArmResult* stage1_arm_results(void* payload) {
  return reinterpret_cast<Stage1ArmResult*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const Stage1ArmResult* stage1_arm_results(const void* payload) {
  return reinterpret_cast<const Stage1ArmResult*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline size_t cleanup_activate_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(CleanupActivateItem);
}

inline size_t cleanup_activate_response_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(CleanupActivateResult);
}

inline CleanupActivateItem* cleanup_activate_items(void* payload) {
  return reinterpret_cast<CleanupActivateItem*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const CleanupActivateItem* cleanup_activate_items(
    const void* payload) {
  return reinterpret_cast<const CleanupActivateItem*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline CleanupActivateResult* cleanup_activate_results(void* payload) {
  return reinterpret_cast<CleanupActivateResult*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const CleanupActivateResult* cleanup_activate_results(
    const void* payload) {
  return reinterpret_cast<const CleanupActivateResult*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline size_t authority_placement_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(AuthorityPlacementItem);
}

inline size_t authority_placement_response_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(AuthorityPlacementResult);
}

inline size_t dynamic_node_control_request_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(DynamicNodeControlItem);
}

inline size_t dynamic_node_control_response_bytes(u32 item_count) {
  return sizeof(PeerRpcHeader) + static_cast<size_t>(item_count) *
    sizeof(DynamicNodeControlResult);
}

inline AuthorityPlacementItem* authority_placement_items(void* payload) {
  return reinterpret_cast<AuthorityPlacementItem*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const AuthorityPlacementItem* authority_placement_items(
    const void* payload) {
  return reinterpret_cast<const AuthorityPlacementItem*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline AuthorityPlacementResult* authority_placement_results(void* payload) {
  return reinterpret_cast<AuthorityPlacementResult*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const AuthorityPlacementResult* authority_placement_results(
    const void* payload) {
  return reinterpret_cast<const AuthorityPlacementResult*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline DynamicNodeControlItem* dynamic_node_control_items(void* payload) {
  return reinterpret_cast<DynamicNodeControlItem*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const DynamicNodeControlItem* dynamic_node_control_items(
    const void* payload) {
  return reinterpret_cast<const DynamicNodeControlItem*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline DynamicNodeControlResult* dynamic_node_control_results(void* payload) {
  return reinterpret_cast<DynamicNodeControlResult*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const DynamicNodeControlResult* dynamic_node_control_results(
    const void* payload) {
  return reinterpret_cast<const DynamicNodeControlResult*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline CentroidMembershipOp* centroid_membership_ops(void* payload) {
  return reinterpret_cast<CentroidMembershipOp*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const CentroidMembershipOp* centroid_membership_ops(
    const void* payload) {
  return reinterpret_cast<const CentroidMembershipOp*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline ReverseUpdateOp* reverse_update_ops(void* payload) {
  return reinterpret_cast<ReverseUpdateOp*>(reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const ReverseUpdateOp* reverse_update_ops(const void* payload) {
  return reinterpret_cast<const ReverseUpdateOp*>(reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline ReconcileReverseOp* reconcile_reverse_ops(void* payload) {
  return reinterpret_cast<ReconcileReverseOp*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const ReconcileReverseOp* reconcile_reverse_ops(
    const void* payload) {
  return reinterpret_cast<const ReconcileReverseOp*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline size_t stage2_expand_score_queries_offset(u32 item_count) {
  return align_wire_u64(
    wire_saturating_add(sizeof(PeerRpcHeader), wire_saturating_multiply(
      item_count, sizeof(Stage2ExpandScoreItem))));
}

inline size_t stage2_expand_score_request_bytes(u32 item_count) {
  return wire_saturating_add(
    stage2_expand_score_queries_offset(item_count), wire_saturating_multiply(
      item_count, VamanaNode::vector_bytes()));
}

inline size_t stage2_expand_score_response_bytes(
    u32 item_count, u32 neighbor_count) {
  const size_t result_bytes = wire_saturating_multiply(
    item_count, sizeof(Stage2ExpandScoreResult));
  return wire_saturating_add(
    wire_saturating_add(sizeof(PeerRpcHeader), result_bytes),
    wire_saturating_multiply(
      neighbor_count, sizeof(Stage2ExpandScoreNeighbor)));
}

inline size_t stage2_expand_score_response_bytes(u32 item_count) {
  const size_t neighbor_count = wire_saturating_multiply(
    item_count, VamanaNode::graph_entry_capacity());
  if (neighbor_count > std::numeric_limits<u32>::max()) {
    return std::numeric_limits<size_t>::max();
  }
  return stage2_expand_score_response_bytes(
    item_count, static_cast<u32>(neighbor_count));
}

inline Stage2ExpandScoreItem* stage2_expand_score_items(void* payload) {
  return reinterpret_cast<Stage2ExpandScoreItem*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const Stage2ExpandScoreItem* stage2_expand_score_items(
    const void* payload) {
  return reinterpret_cast<const Stage2ExpandScoreItem*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline byte_t* stage2_expand_score_queries(void* payload, u32 item_count) {
  return reinterpret_cast<byte_t*>(payload) +
         stage2_expand_score_queries_offset(item_count);
}

inline const byte_t* stage2_expand_score_queries(
    const void* payload, u32 item_count) {
  return reinterpret_cast<const byte_t*>(payload) +
         stage2_expand_score_queries_offset(item_count);
}

inline Stage2ExpandScoreResult* stage2_expand_score_results(void* payload) {
  return reinterpret_cast<Stage2ExpandScoreResult*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline const Stage2ExpandScoreResult* stage2_expand_score_results(
    const void* payload) {
  return reinterpret_cast<const Stage2ExpandScoreResult*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader));
}

inline Stage2ExpandScoreNeighbor* stage2_expand_score_neighbors(
    void* payload, u32 item_count) {
  return reinterpret_cast<Stage2ExpandScoreNeighbor*>(
    reinterpret_cast<byte_t*>(payload) + sizeof(PeerRpcHeader) +
    static_cast<size_t>(item_count) * sizeof(Stage2ExpandScoreResult));
}

inline const Stage2ExpandScoreNeighbor* stage2_expand_score_neighbors(
    const void* payload, u32 item_count) {
  return reinterpret_cast<const Stage2ExpandScoreNeighbor*>(
    reinterpret_cast<const byte_t*>(payload) + sizeof(PeerRpcHeader) +
    static_cast<size_t>(item_count) * sizeof(Stage2ExpandScoreResult));
}

}  // namespace service::storage_owner
