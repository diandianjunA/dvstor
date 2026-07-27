#include <cassert>
#include <chrono>

#include "gpu_search/centroid_route_poll_policy.hh"

int main() {
  using namespace std::chrono_literals;
  using gpu_search::centroid_route_poll::ProbeAction;
  using gpu_search::centroid_route_poll::PublicationIdentity;
  const PublicationIdentity baseline{
    .sequence = 18,
    .version = 7,
    .body_checksum = 123,
    .vector_count = 1000,
    .live_entry_count = 4,
  };
  assert(!gpu_search::centroid_route_poll::body_read_required(
    true, baseline, baseline));
  assert(gpu_search::centroid_route_poll::classify_probe(
           true, baseline, baseline) == ProbeAction::retain_cached);
  assert(gpu_search::centroid_route_poll::body_read_required(
    false, baseline, baseline));
  assert(gpu_search::centroid_route_poll::classify_probe(
           false, baseline, baseline) == ProbeAction::read_body);
  auto changed = baseline;
  changed.sequence += 2;
  assert(gpu_search::centroid_route_poll::body_read_required(
    true, baseline, changed));
  assert(gpu_search::centroid_route_poll::classify_probe(
           true, baseline, changed) == ProbeAction::read_body);
  // A restarted publisher can reuse a sequence; checksum/version identity
  // still forces complete validation rather than trusting stale cache state.
  changed = baseline;
  changed.body_checksum += 1;
  assert(gpu_search::centroid_route_poll::body_read_required(
    true, baseline, changed));

  auto torn = changed;
  torn.sequence |= 1;
  assert(gpu_search::centroid_route_poll::classify_probe(
           true, baseline, torn) == ProbeAction::retry);
  assert(!gpu_search::centroid_route_poll::body_read_is_stable(
    changed, changed, changed.sequence + 2));
  auto restarted_body = changed;
  restarted_body.body_checksum += 1;
  assert(!gpu_search::centroid_route_poll::body_read_is_stable(
    changed, restarted_body, changed.sequence));
  assert(!gpu_search::centroid_route_poll::body_read_is_stable(
    changed, torn, torn.sequence));
  assert(gpu_search::centroid_route_poll::body_read_is_stable(
    changed, changed, changed.sequence));

  gpu_search::centroid_route_poll::AdaptiveIdleBackoff backoff{17};
  gpu_search::centroid_route_poll::AdaptiveIdleBackoff same_client{17};
  gpu_search::centroid_route_poll::AdaptiveIdleBackoff other_client{18};

  assert(backoff.base_delay() == 4ms);
  assert(backoff.delay() >= 4ms && backoff.delay() < 5ms);
  assert(backoff.delay() == same_client.delay());
  bool observed_jitter_difference = backoff.delay() != other_client.delay();
  backoff.observe(false);
  same_client.observe(false);
  other_client.observe(false);
  assert(backoff.base_delay() == 8ms);
  assert(backoff.delay() == same_client.delay());
  observed_jitter_difference = observed_jitter_difference ||
    backoff.delay() != other_client.delay();
  backoff.observe(false);
  same_client.observe(false);
  other_client.observe(false);
  assert(backoff.base_delay() == 16ms);
  backoff.observe(false);
  same_client.observe(false);
  other_client.observe(false);
  assert(backoff.base_delay() == 32ms);
  backoff.observe(false);
  same_client.observe(false);
  other_client.observe(false);
  assert(backoff.base_delay() == 64ms);
  assert(backoff.delay() > 63ms && backoff.delay() <= 64ms);
  observed_jitter_difference = observed_jitter_difference ||
    backoff.delay() != other_client.delay();
  assert(observed_jitter_difference);

  backoff.observe(true);
  assert(backoff.base_delay() == 4ms);
  assert(backoff.delay() >= 4ms && backoff.delay() < 5ms);
  backoff.observe(false);
  assert(backoff.base_delay() == 8ms);
  backoff.observe(true);
  assert(backoff.base_delay() == 4ms);
}
