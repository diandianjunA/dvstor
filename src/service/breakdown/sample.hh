#pragma once

#include <chrono>
#include <memory>

#include "service/breakdown/names.hh"

namespace service::breakdown {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;

struct StorageOwnerAnchorCounters {
  u64 hints{};
  u64 valid_hints{};
  u64 expansions{};
  u64 remote_expansions{};
};

struct SampleCounters {
  u64 storage_owner_anchor_hints{};
  u64 storage_owner_anchor_valid_hints{};
  u64 storage_owner_anchor_expansions{};
  u64 storage_owner_anchor_remote_expansions{};
};

struct Sample {
  explicit Sample(Operation operation, bool collect_fine_grained = true)
      : operation(operation), collect_fine_grained_breakdown(collect_fine_grained) {}

  Operation operation;
  bool collect_fine_grained_breakdown{};
  Clock::time_point enqueued_at{};
  Clock::time_point dequeued_at{};
  Clock::time_point started_at{};
  Clock::time_point finished_at{};
  std::array<u64, kCategoryCount> category_ns{};
  std::array<u64, kSubcategoryCount> subcategory_ns{};
  u64 queue_wait_ns{};
  u64 service_ns{};
  u64 end_to_end_ns{};
  std::shared_ptr<StorageOwnerAnchorCounters> storage_owner_anchor;
  bool started_flag{};
  bool finished_flag{};

  void mark_started(Clock::time_point dequeued, Clock::time_point started) {
    dequeued_at = dequeued;
    started_at = started;
    started_flag = true;
    queue_wait_ns = static_cast<u64>(
      std::chrono::duration_cast<Nanoseconds>(dequeued_at - enqueued_at).count());
  }

  void mark_finished(Clock::time_point finished) {
    finished_at = finished;
    finished_flag = true;
    service_ns = static_cast<u64>(
      std::chrono::duration_cast<Nanoseconds>(finished_at - started_at).count());
    end_to_end_ns = static_cast<u64>(
      std::chrono::duration_cast<Nanoseconds>(finished_at - enqueued_at).count());
  }

  [[nodiscard]] bool collects_breakdown() const {
    return collect_fine_grained_breakdown;
  }

  void add_subcategory(Subcategory subcategory, u64 nanoseconds) {
    if (!collect_fine_grained_breakdown) return;
    subcategory_ns[static_cast<size_t>(subcategory)] += nanoseconds;
    category_ns[static_cast<size_t>(parent_category(subcategory))] += nanoseconds;
  }

  [[nodiscard]] SampleCounters counters() const {
    SampleCounters result;
    if (!collect_fine_grained_breakdown || storage_owner_anchor == nullptr) return result;
    result.storage_owner_anchor_hints = storage_owner_anchor->hints;
    result.storage_owner_anchor_valid_hints = storage_owner_anchor->valid_hints;
    result.storage_owner_anchor_expansions = storage_owner_anchor->expansions;
    result.storage_owner_anchor_remote_expansions = storage_owner_anchor->remote_expansions;
    return result;
  }
};

}  // namespace service::breakdown
