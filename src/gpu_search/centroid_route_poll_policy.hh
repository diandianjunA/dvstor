#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace gpu_search::centroid_route_poll {

// Route publication is pull-based so an idle compute node cannot be notified
// by a storage node. Bound observation latency without turning every compute
// node into a synchronized high-frequency control-plane scanner.
inline constexpr auto kMinimumDelay = std::chrono::milliseconds(4);
inline constexpr auto kMaximumDelay = std::chrono::milliseconds(64);
inline constexpr auto kMaximumJitter = std::chrono::microseconds(750);

struct PublicationIdentity {
  std::uint64_t sequence{};
  std::uint64_t version{};
  std::uint64_t body_checksum{};
  std::uint64_t vector_count{};
  std::uint32_t live_entry_count{};

  friend bool operator==(const PublicationIdentity&,
                         const PublicationIdentity&) = default;
};

inline bool body_read_required(bool cache_valid,
                               const PublicationIdentity& cached,
                               const PublicationIdentity& observed) {
  // sequence detects ordinary publication. The remaining fields also detect a
  // storage restart that recreated the seqlock with the same sequence value.
  return !cache_valid || cached != observed;
}

enum class ProbeAction : std::uint8_t {
  retain_cached,
  read_body,
  retry,
};

inline ProbeAction classify_probe(bool cache_valid,
                                  const PublicationIdentity& cached,
                                  const PublicationIdentity& observed) {
  if (observed.sequence == 0 || (observed.sequence & 1u) != 0) {
    return ProbeAction::retry;
  }
  return body_read_required(cache_valid, cached, observed)
    ? ProbeAction::read_body
    : ProbeAction::retain_cached;
}

inline bool body_read_is_stable(const PublicationIdentity& probed,
                                const PublicationIdentity& body,
                                std::uint64_t sequence_after) {
  return (body.sequence & 1u) == 0 && body == probed &&
         body.sequence == sequence_after;
}

class AdaptiveIdleBackoff {
public:
  using duration = std::chrono::microseconds;

  explicit AdaptiveIdleBackoff(std::uint32_t poll_salt = 0)
      : poll_salt_(poll_salt) {}

  [[nodiscard]] duration delay() const {
    const duration base =
      std::chrono::duration_cast<duration>(base_delay_);
    const duration jitter{
      static_cast<duration::rep>(mix(
        (static_cast<std::uint64_t>(poll_salt_) << 32) ^ poll_round_) %
        (static_cast<std::uint64_t>(kMaximumJitter.count()) + 1))};
    // Add jitter below the ceiling and subtract it at the ceiling. This keeps
    // every wait in [4 ms, 64 ms] while still desynchronizing idle clients.
    return base_delay_ == kMaximumDelay ? base - jitter : base + jitter;
  }

  [[nodiscard]] std::chrono::milliseconds base_delay() const {
    return base_delay_;
  }

  // A route change or a transient seqlock read is activity. Retry those at the
  // minimum base delay. Only a completely stable poll grows the delay.
  void observe(bool activity) {
    ++poll_round_;
    if (activity) {
      base_delay_ = kMinimumDelay;
      return;
    }
    base_delay_ = std::min(kMaximumDelay, base_delay_ * 2);
  }

private:
  static std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  std::chrono::milliseconds base_delay_{kMinimumDelay};
  std::uint32_t poll_salt_{};
  std::uint64_t poll_round_{};
};

}  // namespace gpu_search::centroid_route_poll
