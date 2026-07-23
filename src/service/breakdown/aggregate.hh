#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "service/breakdown/sample.hh"

namespace service::breakdown {

inline constexpr size_t kLatencyReservoirCapacity = 1u << 18;

inline u64 reservoir_hash(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct Aggregate {
  Operation operation{Operation::query};
  size_t count{};
  u64 total_queue_wait_ns{};
  u64 total_service_ns{};
  u64 total_end_to_end_ns{};
  bool fine_grained_breakdown_observed{};
  std::vector<u64> end_to_end_latencies_ns;
  std::vector<u64> service_latencies_ns;
  std::array<u64, kCategoryCount> category_ns{};
  std::array<u64, kSubcategoryCount> subcategory_ns{};

  [[nodiscard]] u64 cpu_other_ns() const {
    u64 explicit_cpu = 0;
    for (size_t index = 0; index < subcategory_ns.size(); ++index) {
      if (parent_category(static_cast<Subcategory>(index)) == Category::cpu) {
        explicit_cpu += subcategory_ns[index];
      }
    }
    const u64 rdma = category_ns[static_cast<size_t>(Category::rdma)];
    const u64 cpu_total = total_service_ns > rdma ? total_service_ns - rdma : 0;
    return cpu_total > explicit_cpu ? cpu_total - explicit_cpu : 0;
  }
};

struct Report {
  Aggregate query{};
  Aggregate insert{};

  [[nodiscard]] bool has_query() const { return query.count > 0; }
  [[nodiscard]] bool has_insert() const { return insert.count > 0; }
};

inline void add_sample(Aggregate& aggregate, const Sample& sample) {
  if (!sample.finished_flag) return;
  aggregate.operation = sample.operation;
  ++aggregate.count;
  aggregate.total_queue_wait_ns += sample.queue_wait_ns;
  aggregate.total_service_ns += sample.service_ns;
  aggregate.total_end_to_end_ns += sample.end_to_end_ns;
  if (aggregate.end_to_end_latencies_ns.size() < kLatencyReservoirCapacity) {
    aggregate.end_to_end_latencies_ns.push_back(sample.end_to_end_ns);
    aggregate.service_latencies_ns.push_back(sample.service_ns);
  } else {
    const size_t replacement = static_cast<size_t>(
      reservoir_hash(static_cast<u64>(aggregate.count)) % aggregate.count);
    if (replacement < kLatencyReservoirCapacity) {
      aggregate.end_to_end_latencies_ns[replacement] = sample.end_to_end_ns;
      aggregate.service_latencies_ns[replacement] = sample.service_ns;
    }
  }
  aggregate.fine_grained_breakdown_observed =
    aggregate.fine_grained_breakdown_observed || sample.collects_breakdown();
  if (!sample.collects_breakdown()) return;
  for (size_t index = 0; index < aggregate.category_ns.size(); ++index) {
    aggregate.category_ns[index] += sample.category_ns[index];
  }
  for (size_t index = 0; index < aggregate.subcategory_ns.size(); ++index) {
    aggregate.subcategory_ns[index] += sample.subcategory_ns[index];
  }
}

// The report returned to callers remains an ordinary value type. Hot-path
// aggregation uses a fixed set of shards so independent query/update threads
// do not serialize on one ComputeService mutex. The latency reservoir is
// shared by all shards and allocated exactly once; its memory does not grow
// with the number of producer threads.
inline constexpr size_t kConcurrentAggregateShardCount = 256;
inline constexpr size_t kLatencyReservoirStripeCount = 256;
static_assert((kConcurrentAggregateShardCount &
               (kConcurrentAggregateShardCount - 1)) == 0);
static_assert((kLatencyReservoirStripeCount &
               (kLatencyReservoirStripeCount - 1)) == 0);

class ConcurrentAggregate {
public:
  explicit ConcurrentAggregate(Operation operation)
      : operation_(operation),
        shards_(std::make_unique<Shard[]>(kConcurrentAggregateShardCount)),
        latency_reservoir_(
          std::make_unique<LatencyPair[]>(kLatencyReservoirCapacity)) {}

  ConcurrentAggregate(const ConcurrentAggregate&) = delete;
  ConcurrentAggregate& operator=(const ConcurrentAggregate&) = delete;

  void add(const Sample& sample) {
    if (!sample.finished_flag) return;

    Shard& shard = shards_[current_shard_index()];
    std::lock_guard<std::mutex> shard_lock(shard.mutex);
    const size_t ordinal =
      sample_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
    ++shard.count;
    shard.total_queue_wait_ns += sample.queue_wait_ns;
    shard.total_service_ns += sample.service_ns;
    shard.total_end_to_end_ns += sample.end_to_end_ns;
    shard.fine_grained_breakdown_observed =
      shard.fine_grained_breakdown_observed || sample.collects_breakdown();
    if (sample.collects_breakdown()) {
      for (size_t index = 0; index < shard.category_ns.size(); ++index) {
        shard.category_ns[index] += sample.category_ns[index];
      }
      for (size_t index = 0; index < shard.subcategory_ns.size(); ++index) {
        shard.subcategory_ns[index] += sample.subcategory_ns[index];
      }
    }

    size_t reservoir_index = kLatencyReservoirCapacity;
    if (ordinal <= kLatencyReservoirCapacity) {
      reservoir_index = ordinal - 1;
    } else {
      const size_t replacement = static_cast<size_t>(
        reservoir_hash(static_cast<u64>(ordinal)) % ordinal);
      if (replacement < kLatencyReservoirCapacity) {
        reservoir_index = replacement;
      }
    }
    if (reservoir_index == kLatencyReservoirCapacity) return;

    // Two latencies form one logical sample. A striped lock keeps the pair
    // coherent while avoiding a global reservoir lock. Holding the producer
    // shard until publication also lets reset()/collect() freeze writers by
    // locking only the fixed shard set.
    std::lock_guard<std::mutex> reservoir_lock(
      reservoir_mutexes_[reservoir_index &
        (kLatencyReservoirStripeCount - 1)].mutex);
    latency_reservoir_[reservoir_index] = {
      sample.end_to_end_ns, sample.service_ns};
  }

  void reset() {
    auto locks = lock_all_shards();
    sample_sequence_.store(0, std::memory_order_relaxed);
    for (size_t index = 0; index < kConcurrentAggregateShardCount; ++index) {
      Shard& shard = shards_[index];
      shard.count = 0;
      shard.total_queue_wait_ns = 0;
      shard.total_service_ns = 0;
      shard.total_end_to_end_ns = 0;
      shard.fine_grained_breakdown_observed = false;
      shard.category_ns.fill(0);
      shard.subcategory_ns.fill(0);
    }
  }

  [[nodiscard]] Aggregate collect() const {
    auto locks = lock_all_shards();
    Aggregate aggregate;
    aggregate.operation = operation_;
    for (size_t shard_index = 0;
         shard_index < kConcurrentAggregateShardCount; ++shard_index) {
      const Shard& shard = shards_[shard_index];
      aggregate.count += shard.count;
      aggregate.total_queue_wait_ns += shard.total_queue_wait_ns;
      aggregate.total_service_ns += shard.total_service_ns;
      aggregate.total_end_to_end_ns += shard.total_end_to_end_ns;
      aggregate.fine_grained_breakdown_observed =
        aggregate.fine_grained_breakdown_observed ||
        shard.fine_grained_breakdown_observed;
      for (size_t index = 0; index < aggregate.category_ns.size(); ++index) {
        aggregate.category_ns[index] += shard.category_ns[index];
      }
      for (size_t index = 0; index < aggregate.subcategory_ns.size(); ++index) {
        aggregate.subcategory_ns[index] += shard.subcategory_ns[index];
      }
    }

    const size_t reservoir_size = std::min(
      sample_sequence_.load(std::memory_order_relaxed),
      kLatencyReservoirCapacity);
    aggregate.end_to_end_latencies_ns.resize(reservoir_size);
    aggregate.service_latencies_ns.resize(reservoir_size);
    for (size_t index = 0; index < reservoir_size; ++index) {
      aggregate.end_to_end_latencies_ns[index] =
        latency_reservoir_[index].end_to_end_ns;
      aggregate.service_latencies_ns[index] =
        latency_reservoir_[index].service_ns;
    }
    return aggregate;
  }

private:
  struct alignas(64) Shard {
    mutable std::mutex mutex;
    size_t count{};
    u64 total_queue_wait_ns{};
    u64 total_service_ns{};
    u64 total_end_to_end_ns{};
    bool fine_grained_breakdown_observed{};
    std::array<u64, kCategoryCount> category_ns{};
    std::array<u64, kSubcategoryCount> subcategory_ns{};
  };

  struct LatencyPair {
    u64 end_to_end_ns{};
    u64 service_ns{};
  };

  struct alignas(64) ReservoirStripe {
    mutable std::mutex mutex;
  };

  using ShardLocks = std::array<std::unique_lock<std::mutex>,
                                kConcurrentAggregateShardCount>;

  [[nodiscard]] size_t current_shard_index() const noexcept {
    static thread_local const u64 thread_hash = reservoir_hash(
      static_cast<u64>(std::hash<std::thread::id>{}(
        std::this_thread::get_id())));
    const u64 aggregate_hash = reservoir_hash(
      thread_hash ^ static_cast<u64>(
        reinterpret_cast<std::uintptr_t>(this)));
    return static_cast<size_t>(aggregate_hash) &
      (kConcurrentAggregateShardCount - 1);
  }

  [[nodiscard]] ShardLocks lock_all_shards() const {
    ShardLocks locks{};
    for (size_t index = 0; index < kConcurrentAggregateShardCount; ++index) {
      locks[index] = std::unique_lock<std::mutex>(shards_[index].mutex);
    }
    return locks;
  }

  Operation operation_;
  std::unique_ptr<Shard[]> shards_;
  std::unique_ptr<LatencyPair[]> latency_reservoir_;
  mutable std::array<ReservoirStripe, kLatencyReservoirStripeCount>
    reservoir_mutexes_;
  std::atomic<size_t> sample_sequence_{0};
};

inline void add_sample(ConcurrentAggregate& aggregate, const Sample& sample) {
  aggregate.add(sample);
}

struct ConcurrentReport {
  ConcurrentAggregate query{Operation::query};
  ConcurrentAggregate insert{Operation::insert};

  void reset() {
    query.reset();
    insert.reset();
  }

  [[nodiscard]] Report collect() const {
    return {.query = query.collect(), .insert = insert.collect()};
  }
};

inline double ns_to_ms(u64 nanoseconds) {
  return static_cast<double>(nanoseconds) / 1'000'000.0;
}

inline u64 percentile_ns(std::vector<u64> values, double percentile) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const double index = percentile * static_cast<double>(values.size() - 1);
  return values[static_cast<size_t>(index)];
}

}  // namespace service::breakdown
