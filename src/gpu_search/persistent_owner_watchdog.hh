#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include "common/types.hh"

namespace gpu_search::owner_watchdog {

inline constexpr u64 kMinimumStallTimeoutNs = 100'000'000ULL;
inline constexpr u64 kTransportTimeoutMultiplier = 4;

inline constexpr u64 stall_timeout_ns(u64 transport_timeout_ns) {
  const u64 scaled = transport_timeout_ns >
      std::numeric_limits<u64>::max() / kTransportTimeoutMultiplier
    ? std::numeric_limits<u64>::max()
    : transport_timeout_ns * kTransportTimeoutMultiplier;
  return std::max(kMinimumStallTimeoutNs, scaled);
}

struct Observation {
  u64 announced{};
  u64 dequeued{};
  u64 completed{};
  u64 heartbeat{};
};

// Host-only policy for one GPU QP owner. New announcements deliberately do
// not refresh the deadline once work is outstanding: a stream of producers
// must not hide a dead owner. Only dequeue/completion progress does. An idle
// owner is never armed, regardless of how long it remains idle.
class Tracker {
 public:
  bool observe(const Observation& observation, u64 now_ns,
               u64 timeout_ns) {
    const bool counters_regressed =
      observation.announced < last_.announced ||
      observation.dequeued < last_.dequeued ||
      observation.completed < last_.completed || now_ns < last_sample_ns_;
    const bool inconsistent = observation.completed > observation.announced;
    if (counters_regressed || inconsistent) {
      reset(observation, now_ns);
      return false;
    }

    const bool pending = observation.announced != observation.completed;
    const bool owner_progressed =
      observation.dequeued != last_.dequeued ||
      observation.completed != last_.completed;
    if (!pending) {
      armed_ = false;
      last_progress_ns_ = now_ns;
    } else if (!armed_) {
      armed_ = true;
      last_progress_ns_ = now_ns;
    } else if (owner_progressed) {
      last_progress_ns_ = now_ns;
    }

    last_ = observation;
    last_sample_ns_ = now_ns;
    return armed_ && now_ns - last_progress_ns_ >= timeout_ns;
  }

  u64 stalled_for_ns(u64 now_ns) const {
    return armed_ && now_ns >= last_progress_ns_
      ? now_ns - last_progress_ns_ : 0;
  }

 private:
  void reset(const Observation& observation, u64 now_ns) {
    last_ = observation;
    last_sample_ns_ = now_ns;
    last_progress_ns_ = now_ns;
    armed_ = observation.announced > observation.completed;
  }

  Observation last_{};
  u64 last_sample_ns_{};
  u64 last_progress_ns_{};
  bool armed_{};
};

}  // namespace gpu_search::owner_watchdog
