#include <cassert>

#include "gpu_search/persistent_owner_watchdog.hh"

using gpu_search::owner_watchdog::Observation;
using gpu_search::owner_watchdog::Tracker;

int main() {
  constexpr u64 ms = 1'000'000ULL;
  const u64 timeout = gpu_search::owner_watchdog::stall_timeout_ns(20 * ms);
  assert(timeout == 100 * ms);
  assert(gpu_search::owner_watchdog::stall_timeout_ns(40 * ms) == 160 * ms);
  assert(gpu_search::owner_watchdog::stall_timeout_ns(250 * ms) == 1'000 * ms);

  // An idle owner is never considered stalled.
  Tracker idle;
  assert(!idle.observe({}, 0, timeout));
  assert(!idle.observe({}, 10'000 * ms, timeout));

  // First outstanding work arms the timer. More producer announcements do
  // not hide an owner that never dequeues anything.
  Tracker dead_owner;
  assert(!dead_owner.observe({.announced = 1}, 0, timeout));
  assert(!dead_owner.observe({.announced = 100, .heartbeat = 9},
                             99 * ms, timeout));
  assert(dead_owner.observe({.announced = 101, .heartbeat = 10},
                            100 * ms, timeout));

  // Dequeue and completion are real progress and refresh the deadline even
  // while the queue remains continuously non-empty.
  Tracker loaded_owner;
  assert(!loaded_owner.observe({.announced = 20}, 0, timeout));
  assert(!loaded_owner.observe({.announced = 30, .dequeued = 8},
                               90 * ms, timeout));
  assert(!loaded_owner.observe(
    {.announced = 40, .dequeued = 8, .completed = 8}, 180 * ms, timeout));
  assert(!loaded_owner.observe(
    {.announced = 48, .dequeued = 16, .completed = 8}, 270 * ms, timeout));
  assert(!loaded_owner.observe(
    {.announced = 48, .dequeued = 16, .completed = 16}, 360 * ms, timeout));
  assert(loaded_owner.observe(
    {.announced = 48, .dequeued = 16, .completed = 16, .heartbeat = 20},
    460 * ms, timeout));

  // Draining disarms the timer; a later generation receives a fresh grace
  // period instead of inheriting the prior request's age.
  Tracker generations;
  assert(!generations.observe({.announced = 1}, 0, timeout));
  assert(!generations.observe(
    {.announced = 1, .dequeued = 1, .completed = 1}, 80 * ms, timeout));
  assert(!generations.observe(
    {.announced = 2, .dequeued = 1, .completed = 1}, 1'000 * ms, timeout));
  assert(!generations.observe(
    {.announced = 2, .dequeued = 1, .completed = 1}, 1'099 * ms, timeout));
  assert(generations.observe(
    {.announced = 2, .dequeued = 1, .completed = 1}, 1'100 * ms, timeout));

  // A torn/regressed sample is conservatively re-armed, never an immediate
  // false failure.
  Tracker inconsistent;
  assert(!inconsistent.observe({.announced = 4, .completed = 5},
                               500 * ms, timeout));
  assert(!inconsistent.observe({.announced = 6, .completed = 5},
                               501 * ms, timeout));
}
