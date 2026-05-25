#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <unistd.h>

#include "common/types.hh"

namespace tools::vamana_offline {

size_t effective_thread_count(u32 configured_threads);
str format_duration(std::chrono::steady_clock::duration duration);

class ProgressReporter {
public:
  ProgressReporter(str label, size_t total)
      : label_(std::move(label)),
        total_(std::max<size_t>(total, 1)),
        interactive_(::isatty(fileno(stderr)) != 0),
        start_(std::chrono::steady_clock::now()),
        last_render_(start_),
        thread_([this]() { run(); }) {}

  ~ProgressReporter() { finish(); }

  void increment(size_t value = 1) { current_.fetch_add(value, std::memory_order_relaxed); }

  void finish() {
    if (!finished_.exchange(true, std::memory_order_relaxed)) {
      current_.store(total_, std::memory_order_relaxed);
      if (thread_.joinable()) thread_.join();
    }
  }

private:
  void run() {
    while (!finished_.load(std::memory_order_relaxed)) {
      render(false);
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    render(true);
  }

  void render(bool done) {
    const size_t current = std::min(current_.load(std::memory_order_relaxed), total_);
    const double ratio = static_cast<double>(current) / static_cast<double>(total_);
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - start_;

    std::ostringstream os;
    os << label_ << " ";

    if (interactive_) {
      constexpr size_t bar_width = 28;
      const size_t filled = static_cast<size_t>(ratio * static_cast<double>(bar_width));
      os << "[";
      for (size_t i = 0; i < bar_width; ++i) os << (i < filled ? '=' : ' ');
      os << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << "% ";
      os << "(" << current << "/" << total_ << ") ";
      os << "elapsed " << format_duration(elapsed);
      if (current > 0 && current < total_) {
        const auto estimated = std::chrono::duration_cast<std::chrono::steady_clock::duration>(elapsed / ratio);
        os << " eta " << format_duration(estimated - elapsed);
      }
      std::cerr << '\r' << os.str();
      if (done) std::cerr << '\n';
      std::cerr.flush();
      return;
    }

    const size_t bucket = done ? 20 : static_cast<size_t>(ratio * 20.0);
    const auto log_interval = std::chrono::seconds(15);
    if (!done && bucket <= last_bucket_ && (now - last_render_) < log_interval) return;
    last_bucket_ = std::max(last_bucket_, bucket);
    last_render_ = now;

    os << static_cast<int>(ratio * 100.0) << "% (" << current << "/" << total_ << ") elapsed "
       << format_duration(elapsed);
    if (done) os << " done";
    std::cerr << os.str() << '\n';
  }

  const str label_;
  const size_t total_;
  const bool interactive_;
  const std::chrono::steady_clock::time_point start_;
  std::atomic<size_t> current_{0};
  std::atomic<bool> finished_{false};
  size_t last_bucket_{0};
  std::chrono::steady_clock::time_point last_render_;
  std::thread thread_;
};

template <class Function>
void parallel_for(size_t begin, size_t end, size_t num_threads, Function&& fn) {
  num_threads = effective_thread_count(num_threads);
  if (num_threads == 1 || end <= begin + 1) {
    for (size_t i = begin; i < end; ++i) fn(i, 0);
    return;
  }
  std::atomic<size_t> current{begin};
  std::exception_ptr last_exception;
  std::mutex exception_mutex;
  vec<std::thread> threads;
  threads.reserve(num_threads);
  for (size_t tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      for (;;) {
        const size_t i = current.fetch_add(1);
        if (i >= end) return;
        try { fn(i, tid); }
        catch (...) {
          std::lock_guard<std::mutex> lock(exception_mutex);
          if (!last_exception) last_exception = std::current_exception();
          current.store(end);
          return;
        }
      }
    });
  }
  for (auto& t : threads) t.join();
  if (last_exception) std::rethrow_exception(last_exception);
}


}  // namespace tools::vamana_offline
