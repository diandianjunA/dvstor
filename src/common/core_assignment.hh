#pragma once

#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <string_view>

#include <library/thread.hh>

#include "common/core_partition.hh"

// Specifically designed for our machines
// Please adjust accordingly
// Note that our NIC is attached to NUMA node 1
//
// w/out hyper-threading:
// NUMA node0 CPU(s):    0-7
// NUMA node1 CPU(s):    8-15
//
// w/ hyper-threading
// NUMA node0 CPU(s):   0-7,16-23
// NUMA node1 CPU(s):   8-15,24-31
//
// Strict policy: pin threads in the following order: 8-15, 0-7, 24-31, 16-23
// Interleaved policy: 8,0,9,1,...,24,16,25,17,...

enum AssignmentPolicy { interleaved, strict };

template <AssignmentPolicy>
class CoreAssignment {
public:
  CoreAssignment() : cores_(num_cores_) {
    set_core_sequence();
    apply_local_process_partition();
    restrict_current_thread_to_partition();
    print_hardware_info();
  }

  u32 get_available_core() { return cores_[assigned_cores_++ % cores_.size()]; }
  u32 available_core_count() const { return static_cast<u32>(cores_.size()); }
  bool hyperthreading_enabled() const { return num_cores_ == physical_cores_per_socket_ * num_sockets_ * 2; }
  void reset() { assigned_cores_ = 0; }

private:
  void set_core_sequence();
  void print_hardware_info() const;

  static std::optional<u32> read_partition_environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return std::nullopt;
    u32 parsed = 0;
    const std::string_view text{value};
    const auto [end, error] = std::from_chars(
      text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size()) {
      throw std::invalid_argument(std::string("invalid ") + name);
    }
    return parsed;
  }

  void apply_local_process_partition() {
    const auto rank = read_partition_environment("DVSTOR_LOCAL_PROCESS_RANK");
    const auto count = read_partition_environment("DVSTOR_LOCAL_PROCESS_COUNT");
    if (!rank.has_value() && !count.has_value()) return;
    if (!rank.has_value() || !count.has_value() || *count == 0 || *rank >= *count) {
      throw std::invalid_argument(
        "DVSTOR_LOCAL_PROCESS_RANK/COUNT must describe one valid partition");
    }
    local_process_rank_ = *rank;
    local_process_count_ = *count;
    if (*count == 1) return;
    cores_ = core_assignment_detail::partition_ordered_cores(
      cores_, hyperthreading_enabled(), *rank, *count);
  }

  void restrict_current_thread_to_partition() {
#ifdef __linux__
    cpu_set_t inherited;
    CPU_ZERO(&inherited);
    if (sched_getaffinity(0, sizeof(inherited), &inherited) != 0) {
      throw std::runtime_error(
        std::string("sched_getaffinity failed: ") + std::strerror(errno));
    }

    // Respect an outer taskset/cgroup mask. The intersection also ensures that
    // later pin_thread calls cannot escape the local shard partition.
    vec<u32> allowed;
    allowed.reserve(cores_.size());
    for (const u32 cpu : cores_) {
      if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &inherited)) {
        allowed.push_back(cpu);
      }
    }
    if (allowed.empty()) {
      throw std::runtime_error(
        "local process CPU partition does not intersect inherited affinity");
    }

    cpu_set_t partition;
    CPU_ZERO(&partition);
    for (const u32 cpu : allowed) CPU_SET(cpu, &partition);
    if (sched_setaffinity(0, sizeof(partition), &partition) != 0) {
      throw std::runtime_error(
        std::string("sched_setaffinity failed: ") + std::strerror(errno));
    }
    cores_ = std::move(allowed);
#endif
  }

private:
  const u32 num_cores_{std::thread::hardware_concurrency()};
  const u32 num_sockets_{2};
  const u32 physical_cores_per_socket_{num_cores_ > 16 ? num_cores_ / (2 * num_sockets_) : num_cores_ / num_sockets_};
  u32 assigned_cores_{0};
  vec<u32> cores_;
  u32 local_process_rank_{0};
  u32 local_process_count_{1};
};
