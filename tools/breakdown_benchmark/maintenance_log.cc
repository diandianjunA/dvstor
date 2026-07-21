#include "tools/breakdown_benchmark/maintenance_log.hh"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace tools::breakdown_benchmark {
namespace {

using Fields = std::unordered_map<std::string, std::string>;

constexpr std::array<double, kMaintenanceLatencyBucketCount>
  kMaintenanceLatencyBucketUpperMs{
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
    512.0, 1000.0, 2000.0, 4000.0, 5000.0, 8000.0, 16000.0, 30000.0,
    std::numeric_limits<double>::infinity(),
  };

std::optional<uint64_t> parse_u64(const Fields& fields, const char* key) {
  const auto iterator = fields.find(key);
  if (iterator == fields.end()) return std::nullopt;
  uint64_t value = 0;
  const std::string_view text(iterator->second);
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

std::optional<double> parse_double(const Fields& fields, const char* key) {
  const auto iterator = fields.find(key);
  if (iterator == fields.end()) return std::nullopt;
  try {
    size_t consumed = 0;
    const double value = std::stod(iterator->second, &consumed);
    if (consumed != iterator->second.size() || !std::isfinite(value)) {
      return std::nullopt;
    }
    return value;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::array<uint64_t, kMaintenanceLatencyBucketCount>>
parse_histogram(const Fields& fields) {
  const auto iterator = fields.find("stage2_delay_histogram");
  if (iterator == fields.end()) return std::nullopt;

  std::array<uint64_t, kMaintenanceLatencyBucketCount> histogram{};
  const std::string_view text(iterator->second);
  size_t begin = 0;
  for (size_t bucket = 0; bucket < histogram.size(); ++bucket) {
    const size_t comma = text.find(',', begin);
    const size_t end = comma == std::string_view::npos ? text.size() : comma;
    if (end == begin) return std::nullopt;
    const std::string_view token = text.substr(begin, end - begin);
    const auto result = std::from_chars(
      token.data(), token.data() + token.size(), histogram[bucket]);
    if (result.ec != std::errc{} ||
        result.ptr != token.data() + token.size()) {
      return std::nullopt;
    }
    if (bucket + 1 == histogram.size()) {
      if (comma != std::string_view::npos) return std::nullopt;
    } else {
      if (comma == std::string_view::npos) return std::nullopt;
      begin = comma + 1;
    }
  }
  return histogram;
}

std::optional<MaintenanceObservation> parse_observation(const std::string& line) {
  constexpr std::string_view marker = "storage-owner maintenance ";
  const size_t marker_position = line.find(marker);
  if (marker_position == std::string::npos ||
      (line.find("observation:", marker_position) == std::string::npos &&
       line.find("summary:", marker_position) == std::string::npos)) {
    return std::nullopt;
  }

  Fields fields;
  size_t position = line.find(':', marker_position);
  if (position == std::string::npos) return std::nullopt;
  ++position;
  while (position < line.size()) {
    while (position < line.size() && line[position] == ' ') ++position;
    const size_t end = line.find(' ', position);
    const std::string_view token(
      line.data() + position,
      (end == std::string::npos ? line.size() : end) - position);
    const size_t equals = token.find('=');
    if (equals != std::string_view::npos && equals != 0 &&
        equals + 1 < token.size()) {
      fields.emplace(std::string(token.substr(0, equals)),
                     std::string(token.substr(equals + 1)));
    }
    if (end == std::string::npos) break;
    position = end + 1;
  }

  const auto stage2_enqueued = parse_u64(fields, "stage2_enqueued");
  const auto stage2_finalized_live = parse_u64(fields, "stage2_finalized_live");
  const auto remaining = parse_u64(fields, "remaining");
  if (!stage2_enqueued || !stage2_finalized_live || !remaining) {
    return std::nullopt;
  }
  const auto failed = parse_u64(fields, "failed");
  const auto peer_reverse_failed = parse_u64(fields, "peer_reverse_failed");
  const auto admission_window = parse_u64(fields, "admission_window");
  const auto completion_outstanding =
    parse_u64(fields, "completion_outstanding");
  const auto stage2_continuations =
    parse_u64(fields, "stage2_continuations");
  const auto stage2_remote_frontier_items =
    parse_u64(fields, "stage2_remote_frontier_items");
  const auto stage2_remote_expansions =
    parse_u64(fields, "stage2_remote_expansions");
  const auto stage2_scored_candidates =
    parse_u64(fields, "stage2_scored_candidates");
  const auto stage2_migrations =
    parse_u64(fields, "stage2_migrations");
  const auto stage2_final_edges =
    parse_u64(fields, "stage2_final_edges");
  const auto stage2_cross_edges_stage1_home =
    parse_u64(fields, "stage2_cross_edges_stage1_home");
  const auto stage2_cross_edges_final_home =
    parse_u64(fields, "stage2_cross_edges_final_home");
  const auto stage1_search_budget_exhausted =
    parse_u64(fields, "stage1_search_budget_exhausted");
  const auto stage2_search_budget_exhausted =
    parse_u64(fields, "stage2_search_budget_exhausted");
  const auto histogram = parse_histogram(fields);
  return MaintenanceObservation{
    .stage2_enqueued = *stage2_enqueued,
    .stage2_finalized_live = *stage2_finalized_live,
    .stale = parse_u64(fields, "stale").value_or(0),
    .remaining = *remaining,
    .peer_reverse_remaining =
      parse_u64(fields, "peer_reverse_remaining").value_or(0),
    .failed = failed.value_or(0),
    .peer_reverse_failed = peer_reverse_failed.value_or(0),
    .admission_window = admission_window.value_or(0),
    .completion_outstanding = completion_outstanding.value_or(0),
    .stage2_continuations = stage2_continuations.value_or(0),
    .stage2_remote_frontier_items =
      stage2_remote_frontier_items.value_or(0),
    .stage2_remote_expansions = stage2_remote_expansions.value_or(0),
    .stage2_scored_candidates = stage2_scored_candidates.value_or(0),
    .stage2_migrations = stage2_migrations.value_or(0),
    .stage2_final_edges = stage2_final_edges.value_or(0),
    .stage2_cross_edges_stage1_home =
      stage2_cross_edges_stage1_home.value_or(0),
    .stage2_cross_edges_final_home =
      stage2_cross_edges_final_home.value_or(0),
    .stage1_search_budget_exhausted =
      stage1_search_budget_exhausted.value_or(0),
    .stage2_search_budget_exhausted =
      stage2_search_budget_exhausted.value_or(0),
    .p99_stage2_delay_upper_ms =
      parse_double(fields, "p99_stage2_delay_upper_ms").value_or(0.0),
    .p99_stage2_delay_over_30s =
      fields.contains("p99_stage2_delay_over_30s") &&
      fields.at("p99_stage2_delay_over_30s") == "true",
    .stage2_delay_histogram = histogram.value_or(
      std::array<uint64_t, kMaintenanceLatencyBucketCount>{}),
    .failure_counters_available = failed.has_value() &&
      peer_reverse_failed.has_value(),
    .stage2_delay_histogram_available = histogram.has_value(),
    .completion_window_available = admission_window.has_value() &&
      completion_outstanding.has_value(),
    .locality_counters_available = stage2_continuations.has_value() &&
      stage2_remote_frontier_items.has_value() &&
      stage2_remote_expansions.has_value() &&
      stage2_scored_candidates.has_value() &&
      stage2_migrations.has_value() && stage2_final_edges.has_value() &&
      stage2_cross_edges_stage1_home.has_value() &&
      stage2_cross_edges_final_home.has_value(),
    .search_budget_counters_available =
      stage1_search_budget_exhausted.has_value() &&
      stage2_search_budget_exhausted.has_value(),
  };
}

struct ParsedLogSlice {
  bool readable{};
  bool rotated{};
  std::vector<MaintenanceObservation> observations;
};

ParsedLogSlice read_log_slice(const std::string& path,
                              uint64_t requested_begin,
                              std::optional<uint64_t> requested_end) {
  ParsedLogSlice result;
  std::ifstream input(path, std::ios::binary);
  if (!input) return result;
  result.readable = true;

  input.seekg(0, std::ios::end);
  const auto end_position = input.tellg();
  const uint64_t file_size = end_position < 0
    ? 0 : static_cast<uint64_t>(end_position);
  uint64_t begin = requested_begin;
  if (begin > file_size) {
    begin = 0;
    result.rotated = true;
  }
  const uint64_t end = std::min(
    requested_end.value_or(file_size), file_size);
  if (end <= begin) return result;

  bool begins_at_line_boundary = begin == 0;
  if (begin != 0) {
    input.clear();
    input.seekg(static_cast<std::streamoff>(begin - 1));
    char previous = 0;
    input.read(&previous, 1);
    begins_at_line_boundary = input && previous == '\n';
  }

  input.clear();
  input.seekg(static_cast<std::streamoff>(begin));
  std::string bytes(static_cast<size_t>(end - begin), '\0');
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  bytes.resize(static_cast<size_t>(input.gcount()));

  size_t position = 0;
  if (!begins_at_line_boundary) {
    const size_t newline = bytes.find('\n');
    if (newline == std::string::npos) return result;
    position = newline + 1;
  }
  while (position < bytes.size()) {
    const size_t newline = bytes.find('\n', position);
    if (newline == std::string::npos) break;
    size_t line_end = newline;
    if (line_end > position && bytes[line_end - 1] == '\r') --line_end;
    if (auto observation = parse_observation(
          bytes.substr(position, line_end - position))) {
      result.observations.push_back(*observation);
    }
    position = newline + 1;
  }
  return result;
}

double backlog_slope(const std::vector<MaintenanceObservation>& observations,
                     double period_seconds) {
  const size_t count = observations.size();
  if (count < 2 || period_seconds <= 0.0) return 0.0;
  const double mean_x = period_seconds * static_cast<double>(count - 1) / 2.0;
  double mean_y = 0.0;
  for (const auto& observation : observations) {
    mean_y += static_cast<double>(observation.backlog());
  }
  mean_y /= static_cast<double>(count);

  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t index = 0; index < count; ++index) {
    const double centered_x =
      period_seconds * static_cast<double>(index) - mean_x;
    numerator += centered_x *
      (static_cast<double>(observations[index].backlog()) - mean_y);
    denominator += centered_x * centered_x;
  }
  return denominator == 0.0 ? 0.0 : numerator / denominator;
}

bool counter_delta(uint64_t baseline, uint64_t latest, uint64_t* delta) {
  if (latest < baseline) return false;
  *delta = latest - baseline;
  return true;
}

bool histogram_delta(
    const std::array<uint64_t, kMaintenanceLatencyBucketCount>& baseline,
    const std::array<uint64_t, kMaintenanceLatencyBucketCount>& latest,
    std::array<uint64_t, kMaintenanceLatencyBucketCount>* delta) {
  for (size_t bucket = 0; bucket < delta->size(); ++bucket) {
    if (latest[bucket] < baseline[bucket]) return false;
    (*delta)[bucket] = latest[bucket] - baseline[bucket];
  }
  return true;
}

uint64_t histogram_sample_count(
    const std::array<uint64_t, kMaintenanceLatencyBucketCount>& histogram) {
  uint64_t count = 0;
  for (const uint64_t bucket : histogram) count += bucket;
  return count;
}

void include_histogram_p99(
    const std::array<uint64_t, kMaintenanceLatencyBucketCount>& histogram,
    MaintenanceLogSummary* summary) {
  const uint64_t samples = histogram_sample_count(histogram);
  summary->p99_stage2_delay_samples += samples;
  if (samples == 0) return;

  const uint64_t target = samples - samples / 100;
  uint64_t accumulated = 0;
  size_t bucket = 0;
  for (; bucket < histogram.size(); ++bucket) {
    accumulated += histogram[bucket];
    if (accumulated >= target) break;
  }
  const bool over_30s = bucket >= histogram.size() - 1;
  const size_t finite_bucket = std::min(bucket, histogram.size() - 2);
  summary->p99_stage2_delay_upper_ms = std::max(
    summary->p99_stage2_delay_upper_ms,
    kMaintenanceLatencyBucketUpperMs[finite_bucket]);
  summary->p99_stage2_delay_over_30s =
    summary->p99_stage2_delay_over_30s || over_30s;
}

std::optional<uint64_t> find_end_offset(
    const std::vector<MaintenanceLogCursor>* end_cursors,
    const std::string& path) {
  if (end_cursors == nullptr) return std::nullopt;
  const auto iterator = std::find_if(
    end_cursors->begin(), end_cursors->end(), [&](const auto& cursor) {
      return cursor.path == path;
    });
  if (iterator == end_cursors->end()) return std::nullopt;
  return iterator->offset;
}

MaintenanceLogSummary summarize_impl(
    const std::vector<MaintenanceLogCursor>& cursors,
    const std::vector<MaintenanceLogCursor>* end_cursors,
    double observation_period_seconds) {
  MaintenanceLogSummary summary;
  summary.requested_logs = cursors.size();
  for (const auto& cursor : cursors) {
    const ParsedLogSlice slice = read_log_slice(
      cursor.path, cursor.offset, find_end_offset(end_cursors, cursor.path));
    if (!slice.readable) {
      summary.unreadable_logs.push_back(cursor.path);
      continue;
    }
    ++summary.readable_logs;
    if (slice.observations.empty()) continue;

    ++summary.logs_with_observations;
    summary.observations += slice.observations.size();
    if (slice.observations.size() >= 2) {
      ++summary.logs_with_slope_observations;
      summary.backlog_slope_per_sec +=
        backlog_slope(slice.observations, observation_period_seconds);
    }
    for (const auto& observation : slice.observations) {
      summary.max_backlog_observed =
        std::max(summary.max_backlog_observed, observation.backlog());
      if (observation.completion_window_available) {
        summary.max_completion_outstanding_per_shard = std::max(
          summary.max_completion_outstanding_per_shard,
          observation.completion_outstanding);
      }
    }
    const auto& latest = slice.observations.back();
    summary.remaining += latest.backlog();
    if (latest.completion_window_available) {
      ++summary.logs_with_completion_window;
      summary.admission_window += latest.admission_window;
      summary.completion_outstanding += latest.completion_outstanding;
    }

    if (!slice.rotated && cursor.baseline_available &&
        cursor.baseline.failure_counters_available &&
        latest.failure_counters_available) {
      uint64_t failed_delta = 0;
      uint64_t peer_failed_delta = 0;
      if (counter_delta(cursor.baseline.failed, latest.failed, &failed_delta) &&
          counter_delta(cursor.baseline.peer_reverse_failed,
                        latest.peer_reverse_failed, &peer_failed_delta)) {
        ++summary.logs_with_failure_deltas;
        summary.failures += failed_delta + peer_failed_delta;
      }
    }

    if (!slice.rotated && cursor.baseline_available &&
        cursor.baseline.stage2_delay_histogram_available &&
        latest.stage2_delay_histogram_available) {
      std::array<uint64_t, kMaintenanceLatencyBucketCount> delta{};
      if (histogram_delta(cursor.baseline.stage2_delay_histogram,
                          latest.stage2_delay_histogram, &delta)) {
        ++summary.logs_with_histogram_deltas;
        include_histogram_p99(delta, &summary);
      }
    }

    if (!slice.rotated && cursor.baseline_available &&
        cursor.baseline.locality_counters_available &&
        latest.locality_counters_available) {
      uint64_t finalized = 0;
      uint64_t continuations = 0;
      uint64_t frontier = 0;
      uint64_t expansions = 0;
      uint64_t scored = 0;
      uint64_t migrations = 0;
      uint64_t final_edges = 0;
      uint64_t cross_before = 0;
      uint64_t cross_after = 0;
      const bool valid =
        counter_delta(cursor.baseline.stage2_finalized_live,
                      latest.stage2_finalized_live, &finalized) &&
        counter_delta(cursor.baseline.stage2_continuations,
                      latest.stage2_continuations, &continuations) &&
        counter_delta(cursor.baseline.stage2_remote_frontier_items,
                      latest.stage2_remote_frontier_items, &frontier) &&
        counter_delta(cursor.baseline.stage2_remote_expansions,
                      latest.stage2_remote_expansions, &expansions) &&
        counter_delta(cursor.baseline.stage2_scored_candidates,
                      latest.stage2_scored_candidates, &scored) &&
        counter_delta(cursor.baseline.stage2_migrations,
                      latest.stage2_migrations, &migrations) &&
        counter_delta(cursor.baseline.stage2_final_edges,
                      latest.stage2_final_edges, &final_edges) &&
        counter_delta(cursor.baseline.stage2_cross_edges_stage1_home,
                      latest.stage2_cross_edges_stage1_home,
                      &cross_before) &&
        counter_delta(cursor.baseline.stage2_cross_edges_final_home,
                      latest.stage2_cross_edges_final_home, &cross_after);
      if (valid) {
        ++summary.logs_with_locality_deltas;
        summary.stage2_finalized_live += finalized;
        summary.stage2_continuations += continuations;
        summary.stage2_remote_frontier_items += frontier;
        summary.stage2_remote_expansions += expansions;
        summary.stage2_scored_candidates += scored;
        summary.stage2_migrations += migrations;
        summary.stage2_final_edges += final_edges;
        summary.stage2_cross_edges_stage1_home += cross_before;
        summary.stage2_cross_edges_final_home += cross_after;
      }
    }

    if (!slice.rotated && cursor.baseline_available &&
        cursor.baseline.search_budget_counters_available &&
        latest.search_budget_counters_available) {
      uint64_t stage1_exhausted = 0;
      uint64_t stage2_exhausted = 0;
      if (counter_delta(cursor.baseline.stage1_search_budget_exhausted,
                        latest.stage1_search_budget_exhausted,
                        &stage1_exhausted) &&
          counter_delta(cursor.baseline.stage2_search_budget_exhausted,
                        latest.stage2_search_budget_exhausted,
                        &stage2_exhausted)) {
        ++summary.logs_with_search_budget_deltas;
        summary.stage1_search_budget_exhausted += stage1_exhausted;
        summary.stage2_search_budget_exhausted += stage2_exhausted;
      }
    }
  }
  summary.backlog_slope_available = summary.requested_logs != 0 &&
    summary.logs_with_slope_observations == summary.requested_logs;
  summary.failure_delta_available = summary.requested_logs != 0 &&
    summary.logs_with_failure_deltas == summary.requested_logs;
  summary.completion_window_available = summary.requested_logs != 0 &&
    summary.logs_with_completion_window == summary.requested_logs;
  summary.locality_delta_available = summary.requested_logs != 0 &&
    summary.logs_with_locality_deltas == summary.requested_logs;
  summary.search_budget_delta_available = summary.requested_logs != 0 &&
    summary.logs_with_search_budget_deltas == summary.requested_logs;
  summary.p99_stage2_delay_available = summary.requested_logs != 0 &&
    summary.logs_with_histogram_deltas == summary.requested_logs &&
    summary.p99_stage2_delay_samples != 0;
  return summary;
}

}  // namespace

uint64_t MaintenanceObservation::backlog() const {
  const uint64_t completed = stage2_finalized_live + stale;
  const uint64_t unfinished_stage2 =
    stage2_enqueued > completed ? stage2_enqueued - completed : 0;
  // These counters overlap for in-flight work. Taking the maximum avoids
  // double-counting while still covering queued, in-flight, and reverse work.
  return std::max({unfinished_stage2, remaining, peer_reverse_remaining});
}

std::vector<MaintenanceLogCursor> snapshot_maintenance_logs(
    const std::vector<std::string>& paths) {
  std::vector<MaintenanceLogCursor> cursors;
  cursors.reserve(paths.size());
  for (const auto& path : paths) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    MaintenanceLogCursor cursor{
      .path = path,
      .offset = error ? 0 : static_cast<uint64_t>(size),
      .baseline = {},
      .baseline_available = false,
    };
    if (!error) {
      const ParsedLogSlice baseline = read_log_slice(path, 0, cursor.offset);
      if (!baseline.observations.empty()) {
        cursor.baseline = baseline.observations.back();
        cursor.baseline_available = true;
      } else if (cursor.offset == 0) {
        // An empty, readable log represents a fresh zero-counter baseline.
        cursor.baseline.failure_counters_available = true;
        cursor.baseline.stage2_delay_histogram_available = true;
        cursor.baseline.locality_counters_available = true;
        cursor.baseline_available = true;
      }
    }
    cursors.push_back(std::move(cursor));
  }
  return cursors;
}

MaintenanceLogSummary summarize_maintenance_logs(
    const std::vector<MaintenanceLogCursor>& cursors,
    double observation_period_seconds) {
  return summarize_impl(cursors, nullptr, observation_period_seconds);
}

MaintenanceLogSummary summarize_maintenance_log_window(
    const std::vector<MaintenanceLogCursor>& begin_cursors,
    const std::vector<MaintenanceLogCursor>& end_cursors,
    double observation_period_seconds) {
  return summarize_impl(
    begin_cursors, &end_cursors, observation_period_seconds);
}

MaintenanceLogSummary summarize_maintenance_snapshot_window(
    const std::vector<std::optional<
      gpu_search::maintenance_telemetry::Snapshot>>& begin,
    const std::vector<std::optional<
      gpu_search::maintenance_telemetry::Snapshot>>& end) {
  MaintenanceLogSummary summary;
  summary.requested_logs = std::max(begin.size(), end.size());
  for (size_t shard = 0; shard < summary.requested_logs; ++shard) {
    if (shard >= begin.size() || shard >= end.size() ||
        !begin[shard].has_value() || !end[shard].has_value()) {
      summary.unreadable_logs.push_back(
        "in-band-control-page:shard-" + std::to_string(shard));
      continue;
    }
    const auto& first = *begin[shard];
    const auto& latest = *end[shard];
    ++summary.readable_logs;
    ++summary.logs_with_observations;
    summary.observations += 2;

    const auto backlog = [](const auto& snapshot) {
      const uint64_t completed = snapshot.stage2_finalized_live +
        snapshot.stale;
      const uint64_t unfinished = snapshot.stage2_enqueued > completed
        ? snapshot.stage2_enqueued - completed : 0;
      return std::max({unfinished, snapshot.remaining,
                       snapshot.peer_reverse_remaining});
    };
    const uint64_t first_backlog = backlog(first);
    const uint64_t latest_backlog = backlog(latest);
    summary.remaining += latest_backlog;
    summary.max_backlog_observed = std::max({
      summary.max_backlog_observed, first_backlog, latest_backlog,
      latest.max_backlog});
    if (latest.published_steady_ns > first.published_steady_ns) {
      const double elapsed_s = static_cast<double>(
        latest.published_steady_ns - first.published_steady_ns) / 1e9;
      summary.backlog_slope_per_sec +=
        (static_cast<double>(latest_backlog) -
         static_cast<double>(first_backlog)) / elapsed_s;
      ++summary.logs_with_slope_observations;
    }

    ++summary.logs_with_completion_window;
    summary.admission_window += latest.admission_window;
    summary.completion_outstanding += latest.completion_outstanding;
    summary.max_completion_outstanding_per_shard = std::max(
      summary.max_completion_outstanding_per_shard,
      latest.completion_outstanding);

    uint64_t failed = 0;
    uint64_t peer_failed = 0;
    if (counter_delta(first.failed, latest.failed, &failed) &&
        counter_delta(first.peer_reverse_failed,
                      latest.peer_reverse_failed, &peer_failed)) {
      ++summary.logs_with_failure_deltas;
      summary.failures += failed + peer_failed;
    }

    std::array<uint64_t, kMaintenanceLatencyBucketCount> latency_delta{};
    static_assert(kMaintenanceLatencyBucketCount ==
      gpu_search::maintenance_telemetry::kLatencyBucketCount);
    if (histogram_delta(first.stage2_delay_histogram,
                        latest.stage2_delay_histogram, &latency_delta)) {
      ++summary.logs_with_histogram_deltas;
      include_histogram_p99(latency_delta, &summary);
    }

    uint64_t finalized = 0;
    uint64_t continuations = 0;
    uint64_t frontier = 0;
    uint64_t expansions = 0;
    uint64_t scored = 0;
    uint64_t migrations = 0;
    uint64_t final_edges = 0;
    uint64_t cross_before = 0;
    uint64_t cross_after = 0;
    const bool locality_valid =
      counter_delta(first.stage2_finalized_live,
                    latest.stage2_finalized_live, &finalized) &&
      counter_delta(first.stage2_continuations,
                    latest.stage2_continuations, &continuations) &&
      counter_delta(first.stage2_remote_frontier_items,
                    latest.stage2_remote_frontier_items, &frontier) &&
      counter_delta(first.stage2_remote_expansions,
                    latest.stage2_remote_expansions, &expansions) &&
      counter_delta(first.stage2_scored_candidates,
                    latest.stage2_scored_candidates, &scored) &&
      counter_delta(first.stage2_migrations,
                    latest.stage2_migrations, &migrations) &&
      counter_delta(first.stage2_final_edges,
                    latest.stage2_final_edges, &final_edges) &&
      counter_delta(first.stage2_cross_edges_stage1_home,
                    latest.stage2_cross_edges_stage1_home,
                    &cross_before) &&
      counter_delta(first.stage2_cross_edges_final_home,
                    latest.stage2_cross_edges_final_home, &cross_after);
    if (locality_valid) {
      ++summary.logs_with_locality_deltas;
      summary.stage2_finalized_live += finalized;
      summary.stage2_continuations += continuations;
      summary.stage2_remote_frontier_items += frontier;
      summary.stage2_remote_expansions += expansions;
      summary.stage2_scored_candidates += scored;
      summary.stage2_migrations += migrations;
      summary.stage2_final_edges += final_edges;
      summary.stage2_cross_edges_stage1_home += cross_before;
      summary.stage2_cross_edges_final_home += cross_after;
    }

    uint64_t stage1_exhausted = 0;
    uint64_t stage2_exhausted = 0;
    if (counter_delta(first.stage1_search_budget_exhausted,
                      latest.stage1_search_budget_exhausted,
                      &stage1_exhausted) &&
        counter_delta(first.stage2_search_budget_exhausted,
                      latest.stage2_search_budget_exhausted,
                      &stage2_exhausted)) {
      ++summary.logs_with_search_budget_deltas;
      summary.stage1_search_budget_exhausted += stage1_exhausted;
      summary.stage2_search_budget_exhausted += stage2_exhausted;
    }
  }
  summary.backlog_slope_available = summary.requested_logs != 0 &&
    summary.logs_with_slope_observations == summary.requested_logs;
  summary.failure_delta_available = summary.requested_logs != 0 &&
    summary.logs_with_failure_deltas == summary.requested_logs;
  summary.completion_window_available = summary.requested_logs != 0 &&
    summary.logs_with_completion_window == summary.requested_logs;
  summary.locality_delta_available = summary.requested_logs != 0 &&
    summary.logs_with_locality_deltas == summary.requested_logs;
  summary.search_budget_delta_available = summary.requested_logs != 0 &&
    summary.logs_with_search_budget_deltas == summary.requested_logs;
  summary.p99_stage2_delay_available = summary.requested_logs != 0 &&
    summary.logs_with_histogram_deltas == summary.requested_logs &&
    summary.p99_stage2_delay_samples != 0;
  return summary;
}

}  // namespace tools::breakdown_benchmark
