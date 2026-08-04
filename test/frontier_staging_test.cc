#include <cassert>
#include <cstdint>
#include <limits>

#include "gpu_search/persistent_kernel/frontier_staging.cuh"

namespace {

namespace detail = gpu_search::persistent_kernel_detail;

using gpu_search::FrontierRequestState;
using gpu_search::FrontierRobEntry;
using gpu_search::kPersistentMaxMergeCandidates;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kGraphScratchBit = 0x80000000u;
constexpr u64 kSelectedHandle = 0x123456789abcdef0ull;

FrontierRobEntry committed_entry(
    u32 issue_epoch = 42, std::uint8_t scratch_slot = 7) {
  FrontierRobEntry entry{};
  entry.node_handle = kSelectedHandle;
  entry.issue_epoch = issue_epoch;
  entry.scratch_slot = scratch_slot;
  entry.state = static_cast<std::uint8_t>(
    FrontierRequestState::committed);
  return entry;
}

bool reusable(
    u32 mask, u32 position, u32 token, const FrontierRobEntry& entry,
    u32 graph_record_slot =
      kGraphScratchBit | static_cast<u32>(7)) {
  return detail::frontier_staging_payload_reusable(
    mask, position, token, kSelectedHandle, entry,
    graph_record_slot, kGraphScratchBit);
}

void test_capacity_boundaries() {
  static_assert(kPersistentMaxMergeCandidates == 2048);

  assert(!detail::frontier_staging_capacity_valid(1, 0));
  assert(detail::frontier_staging_capacity_valid(0, 128));
  assert(detail::frontier_staging_capacity_valid(16, 128));
  assert(!detail::frontier_staging_capacity_valid(17, 128));

  assert(detail::frontier_staging_capacity_valid(3, 128, 511));
  assert(!detail::frontier_staging_capacity_valid(4, 128, 511));
  assert(!detail::frontier_staging_capacity_valid(
    1, std::numeric_limits<u32>::max()));
  assert(!detail::frontier_staging_capacity_valid(
    std::numeric_limits<u32>::max(), 2));
}

void test_fixed_stride_source_index() {
  assert(detail::frontier_staging_source_index(0, 0, 128) == 0);
  assert(detail::frontier_staging_source_index(0, 127, 128) == 127);
  assert(detail::frontier_staging_source_index(1, 0, 128) == 128);
  assert(detail::frontier_staging_source_index(3, 7, 128) == 391);

  // Every valid neighbor of one parent precedes the next parent's region.
  for (u32 parent = 0; parent != 16; ++parent) {
    const u32 first =
      detail::frontier_staging_source_index(parent, 0, 128);
    const u32 last =
      detail::frontier_staging_source_index(parent, 127, 128);
    assert(last - first == 127);
    if (parent != 15) {
      const u32 next =
        detail::frontier_staging_source_index(parent + 1, 0, 128);
      assert(last + 1 == next);
    }
  }
}

void test_token_boundaries_and_round_trip() {
  assert(detail::frontier_staging_token_encodable(0, 0));
  assert(detail::frontier_staging_token_encodable(
    detail::kFrontierStagingMaxIssueEpoch,
    detail::kFrontierStagingScratchMask));
  assert(!detail::frontier_staging_token_encodable(
    detail::kFrontierStagingMaxIssueEpoch + 1, 0));
  assert(!detail::frontier_staging_token_encodable(
    0, detail::kFrontierStagingScratchMask + 1));

  assert(detail::make_frontier_staging_token(0, 0) == 0);
  const u32 maximum = detail::make_frontier_staging_token(
    detail::kFrontierStagingMaxIssueEpoch,
    detail::kFrontierStagingScratchMask);
  assert(maximum == std::numeric_limits<u32>::max());

  constexpr u32 epochs[]{
    0, 1, 42, detail::kFrontierStagingMaxIssueEpoch};
  constexpr u32 slots[]{
    0, 1, 7, detail::kFrontierStagingScratchMask};
  for (const u32 epoch : epochs) {
    for (const u32 slot : slots) {
      const u32 token =
        detail::make_frontier_staging_token(epoch, slot);
      assert(detail::frontier_staging_token_epoch(token) == epoch);
      assert(detail::frontier_staging_token_scratch_slot(token) == slot);
    }
  }
}

void test_reusable_positive_mask_positions() {
  const FrontierRobEntry entry = committed_entry();
  const u32 token =
    detail::make_frontier_staging_token(
      entry.issue_epoch, entry.scratch_slot);

  assert(reusable(1u << 0, 0, token, entry));
  assert(reusable(1u << 7, 7, token, entry));
  assert(reusable(1u << 31, 31, token, entry));
}

void test_reusable_rejects_mask_and_position() {
  const FrontierRobEntry entry = committed_entry();
  const u32 token =
    detail::make_frontier_staging_token(
      entry.issue_epoch, entry.scratch_slot);

  assert(!reusable(0, 0, token, entry));
  assert(!reusable(1u << 6, 7, token, entry));
  assert(!reusable(std::numeric_limits<u32>::max(), 32, token, entry));
  assert(!reusable(std::numeric_limits<u32>::max(),
                   std::numeric_limits<u32>::max(), token, entry));
}

void test_reusable_rejects_epoch_and_scratch_token() {
  FrontierRobEntry entry = committed_entry();
  const u32 good_token =
    detail::make_frontier_staging_token(
      entry.issue_epoch, entry.scratch_slot);

  const u32 wrong_epoch =
    detail::make_frontier_staging_token(
      entry.issue_epoch + 1, entry.scratch_slot);
  assert(!reusable(1, 0, wrong_epoch, entry));

  const u32 wrong_scratch =
    detail::make_frontier_staging_token(
      entry.issue_epoch, entry.scratch_slot + 1);
  assert(!reusable(1, 0, wrong_scratch, entry));

  entry.issue_epoch = detail::kFrontierStagingMaxIssueEpoch + 1;
  assert(!reusable(1, 0, good_token, entry));
}

void test_reusable_rejects_handle_and_state() {
  FrontierRobEntry entry = committed_entry();
  const u32 token =
    detail::make_frontier_staging_token(
      entry.issue_epoch, entry.scratch_slot);

  entry.node_handle ^= 1;
  assert(!reusable(1, 0, token, entry));
  entry.node_handle = kSelectedHandle;

  constexpr FrontierRequestState noncommitted_states[]{
    FrontierRequestState::init,
    FrontierRequestState::issued,
    FrontierRequestState::inflight,
    FrontierRequestState::arrived,
    FrontierRequestState::validated,
    FrontierRequestState::stale,
  };
  for (const FrontierRequestState state : noncommitted_states) {
    entry.state = static_cast<std::uint8_t>(state);
    assert(!reusable(1, 0, token, entry));
  }
}

void test_reusable_rejects_graph_record_mapping() {
  const FrontierRobEntry entry = committed_entry();
  const u32 token =
    detail::make_frontier_staging_token(
      entry.issue_epoch, entry.scratch_slot);

  assert(!reusable(1, 0, token, entry, entry.scratch_slot));
  assert(!reusable(
    1, 0, token, entry,
    kGraphScratchBit | static_cast<u32>(entry.scratch_slot + 1)));
  assert(!reusable(1, 0, token, entry, kGraphScratchBit));

  // The largest representable scratch slot still composes without touching
  // the graph-bank selector.
  const FrontierRobEntry last =
    committed_entry(13, static_cast<std::uint8_t>(
      detail::kFrontierStagingScratchMask));
  const u32 last_token =
    detail::make_frontier_staging_token(
      last.issue_epoch, last.scratch_slot);
  assert(detail::frontier_staging_payload_reusable(
    1, 0, last_token, kSelectedHandle, last,
    kGraphScratchBit | detail::kFrontierStagingScratchMask,
    kGraphScratchBit));
}

}  // namespace

int main() {
  test_capacity_boundaries();
  test_fixed_stride_source_index();
  test_token_boundaries_and_round_trip();
  test_reusable_positive_mask_positions();
  test_reusable_rejects_mask_and_position();
  test_reusable_rejects_epoch_and_scratch_token();
  test_reusable_rejects_handle_and_state();
  test_reusable_rejects_graph_record_mapping();
  return 0;
}
