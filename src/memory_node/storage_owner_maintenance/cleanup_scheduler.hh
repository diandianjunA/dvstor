#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace memory_node_storage_owner_maintenance_detail {

// Cleanup publication is predecessor-fenced: a sequence can run only after
// every earlier maintenance sequence is durable. Consequently the smallest
// queued sequence is the only cleanup item that can possibly be runnable.
// retry_not_before orders duplicate/same-sequence continuations without ever
// bypassing that durable-order fence.
template <class Task>
struct CleanupScheduleLater {
  bool operator()(const Task& lhs, const Task& rhs) const {
    if (lhs.maintenance_sequence != rhs.maintenance_sequence) {
      return lhs.maintenance_sequence > rhs.maintenance_sequence;
    }
    if (lhs.retry_not_before != rhs.retry_not_before) {
      return lhs.retry_not_before > rhs.retry_not_before;
    }
    return lhs.queued_at > rhs.queued_at;
  }
};

inline bool cleanup_predecessors_durable(std::uint64_t sequence,
                                         std::uint64_t durable_sequence) {
  return sequence <= 1 || durable_sequence >= sequence - 1;
}

template <class Task>
inline bool cleanup_schedule_ready(
    const Task& task,
    std::uint64_t durable_sequence,
    std::chrono::steady_clock::time_point now) {
  return cleanup_predecessors_durable(
           task.maintenance_sequence, durable_sequence) &&
    now >= task.retry_not_before;
}

template <class Container>
inline void cleanup_schedule_push(
    Container& heap, typename Container::value_type&& task) {
  using Task = typename Container::value_type;
  heap.push_back(std::move(task));
  std::push_heap(
    heap.begin(), heap.end(), CleanupScheduleLater<Task>{});
}

template <class Container>
inline typename Container::value_type cleanup_schedule_pop(Container& heap) {
  using Task = typename Container::value_type;
  if (heap.empty()) {
    throw std::logic_error("cleanup scheduler popped an empty heap");
  }
  std::pop_heap(
    heap.begin(), heap.end(), CleanupScheduleLater<Task>{});
  Task task = std::move(heap.back());
  heap.pop_back();
  return task;
}

template <class Container>
inline bool cleanup_schedule_valid(const Container& heap) {
  using Task = typename Container::value_type;
  return std::is_heap(
    heap.begin(), heap.end(), CleanupScheduleLater<Task>{});
}

}  // namespace memory_node_storage_owner_maintenance_detail
