#include "tools/breakdown_benchmark/progress.hh"

#include <algorithm>
#include <iostream>
#include <thread>
#include <utility>

namespace tools::breakdown_benchmark {

bool can_start_timed_operation(const std::chrono::steady_clock::time_point deadline,
                               const std::chrono::nanoseconds avg_duration,
                               size_t completed_ops) {
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) {
    return false;
  }
  if (completed_ops == 0 || avg_duration.count() <= 0) {
    return true;
  }

  const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now);
  return remaining >= avg_duration;
}

void update_avg_duration(std::chrono::nanoseconds& avg_duration,
                         const std::chrono::steady_clock::time_point started_at,
                         size_t completed_ops) {
  const auto observed = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now() - started_at);
  if (completed_ops == 0 || avg_duration.count() <= 0) {
    avg_duration = observed;
    return;
  }

  avg_duration = std::chrono::nanoseconds(
    (avg_duration.count() * 7 + observed.count()) / 8);
}

/* class declaration lives in progress.hh */
ProgressReporter::ProgressReporter(std::string label, const std::atomic<size_t>& completed_ops, size_t total_ops,
                                   size_t total_seconds)
    : label_(std::move(label)),
      total_ops_(total_ops),
      total_seconds_(total_seconds),
      completed_ops_(completed_ops),
      start_(std::chrono::steady_clock::now()),
      thread_([this]() { run(); }) {}

ProgressReporter::~ProgressReporter() { finish(); }

void ProgressReporter::finish() {
  finished_.store(true, std::memory_order_release);
  finish_cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void ProgressReporter::run() {
  size_t last_completed = 0;
  auto last_report = start_;
  while (!finished_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(finish_mutex_);
      if (finish_cv_.wait_for(lock, std::chrono::seconds(5), [&] {
            return finished_.load(std::memory_order_acquire);
          })) {
        break;
      }
    }
    const size_t completed = completed_ops_.load(std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(now - start_).count();
    const auto interval = std::chrono::duration<double>(now - last_report).count();
    const double rate = elapsed <= 0.0 ? 0.0 : static_cast<double>(completed) / elapsed;
    const double interval_rate = interval <= 0.0
                                   ? 0.0
                                   : static_cast<double>(completed - last_completed) / interval;
    if (total_seconds_ > 0) {
      std::cerr << "[breakdown][" << label_ << "] progress elapsed=" << elapsed << "s/" << total_seconds_
                << "s, completed=" << completed << " ops, rate=" << rate
                << " ops/s, interval_rate=" << interval_rate << " ops/s" << std::endl;
    } else {
      std::cerr << "[breakdown][" << label_ << "] progress " << completed << "/" << std::max<size_t>(total_ops_, 1)
                << " ops, rate=" << rate << " ops/s, interval_rate=" << interval_rate
                << " ops/s" << std::endl;
    }
    if (total_seconds_ == 0 && completed >= total_ops_) {
      break;
    }
    if (completed == last_completed && completed > 0) {
      std::cerr << "[breakdown][" << label_ << "] still running, no new completions in last interval"
                << std::endl;
    }
    last_completed = completed;
    last_report = now;
  }

  const size_t completed = completed_ops_.load(std::memory_order_relaxed);
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  const double rate = elapsed <= 0.0 ? 0.0 : static_cast<double>(completed) / elapsed;
  if (total_seconds_ > 0) {
    std::cerr << "[breakdown][" << label_ << "] done elapsed=" << elapsed << "s/" << total_seconds_
              << "s, completed=" << completed << " ops, avg_rate=" << rate << " ops/s" << std::endl;
  } else {
    std::cerr << "[breakdown][" << label_ << "] done " << completed << "/" << std::max<size_t>(total_ops_, 1)
              << " ops, avg_rate=" << rate << " ops/s" << std::endl;
  }
}

}  // namespace tools::breakdown_benchmark
