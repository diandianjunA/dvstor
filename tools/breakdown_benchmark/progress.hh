#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace tools::breakdown_benchmark {

bool can_start_timed_operation(std::chrono::steady_clock::time_point deadline,
                               std::chrono::nanoseconds avg_duration,
                               size_t completed_ops);
void update_avg_duration(std::chrono::nanoseconds& avg_duration,
                         std::chrono::steady_clock::time_point started_at,
                         size_t completed_ops);

class ProgressReporter {
public:
  ProgressReporter(std::string label, const std::atomic<size_t>& completed_ops, size_t total_ops = 0,
                   size_t total_seconds = 0);
  ~ProgressReporter();

  void finish();

private:
  void run();

  std::string label_;
  size_t total_ops_;
  size_t total_seconds_;
  const std::atomic<size_t>& completed_ops_;
  std::chrono::steady_clock::time_point start_;
  std::atomic<bool> finished_{false};
  std::mutex finish_mutex_;
  std::condition_variable finish_cv_;
  std::thread thread_;
};

}  // namespace tools::breakdown_benchmark
