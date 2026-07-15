#include <array>
#include <cassert>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "memory_node/storage_owner_index/two_stage_insert_oracle.hh"

namespace detail = memory_node_storage_owner_index_detail;

namespace {

struct Point {
  distance_t x{};
  distance_t y{};
};

constexpr u32 kPartitionCount = 3;
constexpr u32 kBeamWidth = 3;
constexpr u32 kResultLimit = 2;
constexpr f64 kAlpha = 1.2;

RemotePtr node(const u32 partition, const u32 index) {
  return RemotePtr{partition, static_cast<u64>(index) + 1};
}

u32 index_of(const RemotePtr pointer) {
  assert(pointer.byte_offset() > 0);
  return static_cast<u32>(pointer.byte_offset() - 1);
}

distance_t squared_distance(const Point& lhs, const Point& rhs) {
  const distance_t dx = lhs.x - rhs.x;
  const distance_t dy = lhs.y - rhs.y;
  return dx * dx + dy * dy;
}

void test_direct_and_two_stage_have_identical_final_neighbors() {
  // Owner candidates A and B are diverse, so the temporary stage-1 edges are
  // {A, B}.  Remote C is closer and nearly duplicates A; the authoritative
  // global RobustPrune therefore yields {C, B}.  This makes the test prove
  // both equality after stage 2 and the intentionally weaker stage-1 view.
  const std::array<std::array<Point, kBeamWidth>, kPartitionCount> points{{
    {{{1.0F, 0.0F}, {0.0F, 2.0F}, {3.0F, 0.0F}}},
    {{{0.8F, 0.0F}, {2.0F, 0.0F}, {4.0F, 0.0F}}},
    {{{5.0F, 0.0F}, {6.0F, 0.0F}, {7.0F, 0.0F}}},
  }};
  const Point source{0.0F, 0.0F};

  vec<vec<RemotePtr>> entry_points(kPartitionCount);
  for (u32 partition = 0; partition < kPartitionCount; ++partition) {
    for (u32 index = 0; index < kBeamWidth; ++index) {
      entry_points[partition].push_back(node(partition, index));
    }
  }

  const auto point_of = [&](const RemotePtr pointer) -> const Point& {
    assert(pointer.memory_node() < kPartitionCount);
    assert(index_of(pointer) < kBeamWidth);
    return points[pointer.memory_node()][index_of(pointer)];
  };
  const auto score = [&](const RemotePtr pointer)
      -> std::optional<distance_t> {
    return squared_distance(source, point_of(pointer));
  };
  const auto expand = [](RemotePtr, auto&&) {};
  const auto pair_distance = [&](const RemotePtr lhs, const RemotePtr rhs) {
    return squared_distance(point_of(lhs), point_of(rhs));
  };

  const detail::PartitionedInsertOracleResult direct =
    detail::partitioned_direct_insert_reference(
      span<const vec<RemotePtr>>{entry_points},
      kBeamWidth,
      kResultLimit,
      kAlpha,
      score,
      expand,
      pair_distance);
  const detail::PartitionedInsertStage1 stage1 =
    detail::partitioned_two_stage_insert_begin(
      span<const vec<RemotePtr>>{entry_points},
      0,
      kBeamWidth,
      kResultLimit,
      kAlpha,
      score,
      expand,
      pair_distance);
  const detail::PartitionedInsertOracleResult staged =
    detail::partitioned_two_stage_insert_finalize(
      stage1,
      span<const vec<RemotePtr>>{entry_points},
      score,
      expand,
      pair_distance);

  const size_t expected_capacity = kPartitionCount * kBeamWidth;
  assert(direct.candidate_capacity == expected_capacity);
  assert(stage1.candidate_capacity == expected_capacity);
  assert(staged.candidate_capacity == expected_capacity);
  assert(stage1.owner_beam.size() == kBeamWidth);
  assert(direct.merged_candidates.size() == expected_capacity);
  assert(staged.merged_candidates.size() == expected_capacity);

  assert((stage1.temporary_neighbors ==
          vec<RemotePtr>{node(0, 0), node(0, 1)}));
  assert((direct.final_neighbors ==
          vec<RemotePtr>{node(1, 0), node(0, 1)}));
  assert(stage1.temporary_neighbors != direct.final_neighbors);
  assert(staged.final_neighbors == direct.final_neighbors);

  // Sorting and merging are also part of the reference semantics, so the
  // complete candidate stream must be byte-for-byte deterministic here.
  assert(staged.merged_candidates.size() == direct.merged_candidates.size());
  for (size_t index = 0; index < direct.merged_candidates.size(); ++index) {
    assert(staged.merged_candidates[index].rptr ==
           direct.merged_candidates[index].rptr);
    assert(staged.merged_candidates[index].distance ==
           direct.merged_candidates[index].distance);
  }
}

void test_candidate_capacity_and_stage_boundary_are_enforced() {
  assert(detail::partitioned_insert_candidate_capacity(5, 128) == 640);

  bool zero_partitions_rejected = false;
  try {
    (void)detail::partitioned_insert_candidate_capacity(0, 128);
  } catch (const std::invalid_argument&) {
    zero_partitions_rejected = true;
  }
  assert(zero_partitions_rejected);

  bool zero_width_rejected = false;
  try {
    (void)detail::partitioned_insert_candidate_capacity(5, 0);
  } catch (const std::invalid_argument&) {
    zero_width_rejected = true;
  }
  assert(zero_width_rejected);

  detail::PartitionedInsertStage1 stage1;
  stage1.owner_partition = 0;
  stage1.partition_count = 2;
  stage1.beam_width = 1;
  stage1.result_limit = 1;
  stage1.alpha = kAlpha;
  stage1.candidate_capacity = 2;
  stage1.owner_beam.push_back({node(0, 0), 1.0F, true});
  vec<vec<RemotePtr>> changed_partition_set(3);

  bool changed_set_rejected = false;
  try {
    (void)detail::partitioned_two_stage_insert_finalize(
      stage1,
      span<const vec<RemotePtr>>{changed_partition_set},
      [](RemotePtr) -> std::optional<distance_t> { return 1.0F; },
      [](RemotePtr, auto&&) {},
      [](RemotePtr, RemotePtr) -> distance_t { return 1.0F; });
  } catch (const std::invalid_argument&) {
    changed_set_rejected = true;
  }
  assert(changed_set_rejected);
}

}  // namespace

int main() {
  test_direct_and_two_stage_have_identical_final_neighbors();
  test_candidate_capacity_and_stage_boundary_are_enforced();
  return 0;
}
