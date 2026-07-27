#include "tools/breakdown_benchmark/progress.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
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

PacedOperationDispatcher::PacedOperationDispatcher(double query_qps,
                                                   double write_qps)
    : query_{.rate = query_qps}, write_{.rate = write_qps} {
  if (!std::isfinite(query_qps) || !std::isfinite(write_qps) ||
      query_qps < 0.0 || write_qps < 0.0 ||
      (query_qps == 0.0 && write_qps == 0.0)) {
    throw std::invalid_argument(
      "paced operation rates must be finite, non-negative, and not both zero");
  }
}

void PacedOperationDispatcher::start(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point deadline) {
  if (deadline <= start) {
    throw std::invalid_argument("paced operation deadline must follow start");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  start_ = start;
  deadline_ = deadline;
  query_.next_ordinal = 0;
  write_.next_ordinal = 0;
  started_ = true;
}

std::chrono::steady_clock::time_point PacedOperationDispatcher::scheduled_at(
    const Stream& stream) const {
  if (stream.rate <= 0.0) {
    return std::chrono::steady_clock::time_point::max();
  }
  const long double seconds =
    static_cast<long double>(stream.next_ordinal) /
    static_cast<long double>(stream.rate);
  return start_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<long double>(seconds));
}

std::optional<PacedOperationClaim> PacedOperationDispatcher::claim() {
  PacedOperationClaim claim;
  std::chrono::steady_clock::time_point deadline;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_) {
      throw std::logic_error("paced operation dispatcher was not started");
    }
    if (std::chrono::steady_clock::now() >= deadline_) {
      return std::nullopt;
    }

    const auto query_time = scheduled_at(query_);
    const auto write_time = scheduled_at(write_);
    if (query_time >= deadline_ && write_time >= deadline_) {
      return std::nullopt;
    }
    if (query_time <= write_time) {
      claim = {
        .kind = PacedOperationKind::query,
        .ordinal = query_.next_ordinal++,
        .scheduled_at = query_time,
      };
    } else {
      claim = {
        .kind = PacedOperationKind::write,
        .ordinal = write_.next_ordinal++,
        .scheduled_at = write_time,
      };
    }
    deadline = deadline_;
  }

  std::this_thread::sleep_until(claim.scheduled_at);
  if (std::chrono::steady_clock::now() >= deadline) {
    return std::nullopt;
  }
  return claim;
}

uint64_t PacedOperationDispatcher::scheduled_count(double rate, size_t seconds) {
  if (!std::isfinite(rate) || rate < 0.0) {
    throw std::invalid_argument("scheduled rate must be finite and non-negative");
  }
  const long double product = static_cast<long double>(rate) *
    static_cast<long double>(seconds);
  if (product > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
    throw std::overflow_error("scheduled operation count exceeds uint64_t");
  }
  // Claims are scheduled at ordinal/rate for ordinal >= 0, strictly before
  // the deadline, so the count is ceil(rate * duration).
  return static_cast<uint64_t>(std::ceil(product));
}

/* class declaration lives in progress.hh */
ProgressReporter::ProgressReporter(std::string label, const std::atomic<size_t>& completed_ops, size_t total_ops,
                                   size_t total_seconds,
                                   const std::atomic<size_t>* completed_reads,
                                   const std::atomic<size_t>* completed_writes,
                                   std::chrono::milliseconds report_interval)
    : label_(std::move(label)),
      total_ops_(total_ops),
      total_seconds_(total_seconds),
      completed_ops_(completed_ops),
      completed_reads_(completed_reads),
      completed_writes_(completed_writes),
      report_interval_(std::max(report_interval,
                                std::chrono::milliseconds(1))),
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

std::vector<ProgressSample> ProgressReporter::samples() const {
  std::lock_guard<std::mutex> lock(samples_mutex_);
  return samples_;
}

void ProgressReporter::run() {
  size_t last_completed = 0;
  size_t last_reads = 0;
  size_t last_writes = 0;
  auto last_report = start_;
  auto record_sample = [&](size_t completed, size_t reads, size_t writes,
                           std::chrono::steady_clock::time_point sampled_at) {
    const double elapsed =
      std::chrono::duration<double>(sampled_at - start_).count();
    const double interval =
      std::chrono::duration<double>(sampled_at - last_report).count();
    if (interval <= 0.0) return false;
    const size_t interval_ops = completed - last_completed;
    const size_t interval_reads = reads - last_reads;
    const size_t interval_writes = writes - last_writes;
    {
      std::lock_guard<std::mutex> lock(samples_mutex_);
      samples_.push_back(ProgressSample{
        .elapsed_seconds = elapsed,
        .interval_seconds = interval,
        .completed_ops = completed,
        .interval_ops = interval_ops,
        .interval_reads = interval_reads,
        .interval_writes = interval_writes,
        .total_ops_per_sec = static_cast<double>(interval_ops) / interval,
        .query_ops_per_sec = static_cast<double>(interval_reads) / interval,
        .write_ops_per_sec = static_cast<double>(interval_writes) / interval,
      });
    }
    last_completed = completed;
    last_reads = reads;
    last_writes = writes;
    last_report = sampled_at;
    return true;
  };

  const bool timed = total_seconds_ > 0;
  const auto measurement_deadline =
    start_ + std::chrono::seconds(total_seconds_);
  auto next_report = start_ + report_interval_;
  bool deadline_sampled = false;
  while (!finished_.load(std::memory_order_acquire)) {
    const auto report_at = timed
      ? std::min(next_report, measurement_deadline)
      : next_report;
    {
      std::unique_lock<std::mutex> lock(finish_mutex_);
      if (finish_cv_.wait_until(lock, report_at, [&] {
            return finished_.load(std::memory_order_acquire);
          })) {
        break;
      }
    }
    const size_t completed = completed_ops_.load(std::memory_order_relaxed);
    const size_t reads = completed_reads_ == nullptr
      ? 0 : completed_reads_->load(std::memory_order_relaxed);
    const size_t writes = completed_writes_ == nullptr
      ? 0 : completed_writes_->load(std::memory_order_relaxed);
    // Use the scheduled boundary for the interval. In timed mode the last
    // boundary is exactly the measurement deadline; completions observed
    // while callers subsequently drain synchronous RPCs are deliberately not
    // turned into stability samples.
    const auto sampled_at = report_at;
    const auto elapsed =
      std::chrono::duration<double>(sampled_at - start_).count();
    const auto interval =
      std::chrono::duration<double>(sampled_at - last_report).count();
    const double rate = elapsed <= 0.0 ? 0.0 : static_cast<double>(completed) / elapsed;
    const double interval_rate = interval <= 0.0
                                   ? 0.0
                                   : static_cast<double>(completed - last_completed) / interval;
    const double interval_read_rate = interval <= 0.0
      ? 0.0 : static_cast<double>(reads - last_reads) / interval;
    const double interval_write_rate = interval <= 0.0
      ? 0.0 : static_cast<double>(writes - last_writes) / interval;
    const bool made_progress = completed != last_completed;
    record_sample(completed, reads, writes, sampled_at);
    if (total_seconds_ > 0) {
      std::cerr << "[breakdown][" << label_ << "] progress elapsed=" << elapsed << "s/" << total_seconds_
                << "s, completed=" << completed << " ops, rate=" << rate
                << " ops/s, interval_rate=" << interval_rate << " ops/s";
    } else {
      std::cerr << "[breakdown][" << label_ << "] progress " << completed << "/" << std::max<size_t>(total_ops_, 1)
                << " ops, rate=" << rate << " ops/s, interval_rate=" << interval_rate
                << " ops/s";
    }
    if (completed_reads_ != nullptr && completed_writes_ != nullptr) {
      std::cerr << ", interval_read_qps=" << interval_read_rate
                << ", interval_write_qps=" << interval_write_rate;
    }
    std::cerr << std::endl;
    if (total_seconds_ == 0 && completed >= total_ops_) {
      break;
    }
    if (!made_progress && completed > 0) {
      std::cerr << "[breakdown][" << label_ << "] still running, no new completions in last interval"
                << std::endl;
    }
    if (timed && report_at == measurement_deadline) {
      deadline_sampled = true;
      // Keep the reporter alive so finish() remains a synchronization point,
      // but never sample the post-deadline drain interval.
      std::unique_lock<std::mutex> lock(finish_mutex_);
      finish_cv_.wait(lock, [&] {
        return finished_.load(std::memory_order_acquire);
      });
      break;
    }
    next_report += report_interval_;
  }

  const size_t completed = completed_ops_.load(std::memory_order_relaxed);
  const size_t reads = completed_reads_ == nullptr
    ? 0 : completed_reads_->load(std::memory_order_relaxed);
  const size_t writes = completed_writes_ == nullptr
    ? 0 : completed_writes_->load(std::memory_order_relaxed);
  const auto finished_at = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration<double>(finished_at - start_).count();
  if (timed) {
    if (!deadline_sampled) {
      // Early termination still gets a final partial measurement sample. If
      // finish() races the deadline, cap its timestamp at the deadline.
      record_sample(completed, reads, writes,
                    std::min(finished_at, measurement_deadline));
    }
  } else if (completed != last_completed || samples().empty()) {
    record_sample(completed, reads, writes, finished_at);
  }
  const double measurement_elapsed =
    std::chrono::duration<double>(last_report - start_).count();
  const double rate = timed
    ? (measurement_elapsed <= 0.0 ? 0.0
                                 : static_cast<double>(last_completed) /
                                     measurement_elapsed)
    : (elapsed <= 0.0 ? 0.0 : static_cast<double>(completed) / elapsed);
  if (total_seconds_ > 0) {
    const size_t drain_completed =
      completed >= last_completed ? completed - last_completed : 0;
    const double drain_elapsed = std::max(0.0, elapsed - measurement_elapsed);
    std::cerr << "[breakdown][" << label_ << "] done measurement_elapsed="
              << measurement_elapsed << "s/" << total_seconds_
              << "s, measurement_completed=" << last_completed
              << " ops, measurement_avg_rate=" << rate
              << " ops/s, drain_elapsed=" << drain_elapsed
              << "s, drain_completed=" << drain_completed << " ops"
              << std::endl;
  } else {
    std::cerr << "[breakdown][" << label_ << "] done " << completed << "/" << std::max<size_t>(total_ops_, 1)
              << " ops, avg_rate=" << rate << " ops/s" << std::endl;
  }
}

}  // namespace tools::breakdown_benchmark
