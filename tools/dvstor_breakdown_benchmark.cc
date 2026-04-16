#include <algorithm>
#include <atomic>
#include <barrier>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include "common/configuration.hh"
#include "common/distance.hh"
#include "service/breakdown.hh"
#include "service/compute_service.hh"

namespace {

using ConfigMap = std::unordered_map<std::string, std::string>;

struct Args {
  std::string service_config_path;
  std::string workload{"both"};
  size_t warmup_ops{100};
  size_t measure_ops{1000};
  size_t warmup_seconds{0};
  size_t measure_seconds{0};
  size_t client_threads{4};
  double read_ratio{0.5};
  std::string query_file;
  std::string insert_file;
  bool synthetic{false};
  std::string report_json_path;
  std::string report_text_path;
};

struct MixedPhaseStats {
  uint32_t next_insert_id{};
  size_t issued_reads{};
  size_t issued_writes{};
  size_t completed_reads{};
  size_t completed_writes{};
};

std::string trim(std::string value) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

ConfigMap read_config(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open config: " + path);
  }

  ConfigMap config;
  std::string line;
  while (std::getline(input, line)) {
    const auto comment_pos = line.find_first_of("#;");
    if (comment_pos != std::string::npos) {
      line.erase(comment_pos);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }

    const auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      continue;
    }
    auto key = trim(line.substr(0, eq_pos));
    auto value = trim(line.substr(eq_pos + 1));
    if (!key.empty()) {
      config[std::move(key)] = std::move(value);
    }
  }
  return config;
}

bool is_truthy(const std::string& value) {
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

std::vector<std::string> split_tokens(const std::string& value) {
  std::string normalized = value;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::stringstream ss(normalized);
  std::vector<std::string> tokens;
  std::string token;
  while (ss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<char*> make_argv(std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  return argv;
}

std::vector<std::string> build_service_argv(const std::string& service_config_path) {
  const auto config = read_config(service_config_path);
  std::vector<std::string> args;
  args.emplace_back("dvstor_breakdown_benchmark");

  static const std::vector<std::string> multi_keys = {"servers", "clients"};
  static const std::vector<std::string> flag_keys = {
    "initiator", "cache", "routing", "load-index", "store-index", "disable-thread-pinning", "no-recall", "ip-dist",
    "gpudirect-rdma", "gpu-cache"};

  for (const auto& [key, value] : config) {
    const std::string option = "--" + key;
    if (std::find(flag_keys.begin(), flag_keys.end(), key) != flag_keys.end()) {
      if (is_truthy(value)) {
        args.push_back(option);
      }
      continue;
    }

    if (std::find(multi_keys.begin(), multi_keys.end(), key) != multi_keys.end()) {
      const auto tokens = split_tokens(value);
      if (!tokens.empty()) {
        args.push_back(option);
        args.insert(args.end(), tokens.begin(), tokens.end());
      }
      continue;
    }

    args.push_back(option);
    args.push_back(value);
  }

  return args;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (flag == "--service-config") {
      args.service_config_path = require_value("--service-config");
    } else if (flag == "--workload") {
      args.workload = require_value("--workload");
    } else if (flag == "--warmup-ops") {
      args.warmup_ops = std::stoull(require_value("--warmup-ops"));
    } else if (flag == "--measure-ops") {
      args.measure_ops = std::stoull(require_value("--measure-ops"));
    } else if (flag == "--warmup-seconds") {
      args.warmup_seconds = std::stoull(require_value("--warmup-seconds"));
    } else if (flag == "--measure-seconds") {
      args.measure_seconds = std::stoull(require_value("--measure-seconds"));
    } else if (flag == "--client-threads") {
      args.client_threads = std::stoull(require_value("--client-threads"));
    } else if (flag == "--read-ratio") {
      args.read_ratio = std::stod(require_value("--read-ratio"));
    } else if (flag == "--query-file") {
      args.query_file = require_value("--query-file");
    } else if (flag == "--insert-file") {
      args.insert_file = require_value("--insert-file");
    } else if (flag == "--synthetic") {
      args.synthetic = true;
    } else if (flag == "--report-json") {
      args.report_json_path = require_value("--report-json");
    } else if (flag == "--report-text") {
      args.report_text_path = require_value("--report-text");
    } else {
      throw std::runtime_error("unknown argument: " + flag);
    }
  }

  if (args.service_config_path.empty()) {
    throw std::runtime_error("--service-config is required");
  }
  if (args.report_json_path.empty()) {
    throw std::runtime_error("--report-json is required");
  }
  if (args.workload != "query" && args.workload != "insert" && args.workload != "both" &&
      args.workload != "mixed") {
    throw std::runtime_error("--workload must be query, insert, both, or mixed");
  }
  if (args.client_threads == 0) {
    throw std::runtime_error("--client-threads must be > 0");
  }
  if (args.read_ratio < 0.0 || args.read_ratio > 1.0) {
    throw std::runtime_error("--read-ratio must be in [0, 1]");
  }
  const bool use_time_mode = args.warmup_seconds > 0 || args.measure_seconds > 0;
  if (use_time_mode && (args.warmup_seconds == 0 || args.measure_seconds == 0)) {
    throw std::runtime_error("--warmup-seconds and --measure-seconds must both be > 0 when using time-based mode");
  }
  return args;
}

std::vector<float> make_deterministic_vector(uint32_t seed, size_t dim) {
  std::vector<float> vector(dim, 0.0f);
  uint64_t state = 1469598103934665603ull ^ static_cast<uint64_t>(seed);
  for (size_t i = 0; i < dim; ++i) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const uint32_t value = static_cast<uint32_t>((state * 2685821657736338717ull) >> 32);
    vector[i] = static_cast<float>(value % 10000) / 10000.0f;
  }
  vector[seed % dim] += 4.0f;
  vector[(seed * 17 + 3) % dim] += 1.0f;
  return vector;
}

std::vector<float> make_dataset(const std::vector<uint32_t>& ids, size_t dim) {
  std::vector<float> vectors;
  vectors.reserve(ids.size() * dim);
  for (uint32_t id : ids) {
    auto vec = make_deterministic_vector(id, dim);
    vectors.insert(vectors.end(), vec.begin(), vec.end());
  }
  return vectors;
}

std::vector<float> read_fbin(const std::string& path, uint32_t* dim_out, size_t* count_out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }
  uint32_t count = 0;
  uint32_t dim = 0;
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!input) {
    throw std::runtime_error("failed to read fbin header: " + path);
  }
  std::vector<float> data(static_cast<size_t>(count) * dim);
  input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("failed to read fbin payload: " + path);
  }
  if (dim_out) {
    *dim_out = dim;
  }
  if (count_out) {
    *count_out = count;
  }
  return data;
}

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

class ProgressReporter {
public:
  ProgressReporter(std::string label, const std::atomic<size_t>& completed_ops, size_t total_ops = 0,
                   size_t total_seconds = 0)
      : label_(std::move(label)), total_ops_(total_ops), total_seconds_(total_seconds), completed_ops_(completed_ops),
        start_(std::chrono::steady_clock::now()),
        thread_([this]() { run(); }) {}

  ~ProgressReporter() { finish(); }

  void finish() {
    finished_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

private:
  void run() {
    size_t last_completed = 0;
    while (!finished_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      const size_t completed = completed_ops_.load(std::memory_order_relaxed);
      const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
      const double rate = elapsed <= 0.0 ? 0.0 : static_cast<double>(completed) / elapsed;
      if (total_seconds_ > 0) {
        std::cerr << "[breakdown][" << label_ << "] progress elapsed=" << elapsed << "s/" << total_seconds_
                  << "s, completed=" << completed << " ops, rate=" << rate << " ops/s" << std::endl;
      } else {
        std::cerr << "[breakdown][" << label_ << "] progress " << completed << "/" << std::max<size_t>(total_ops_, 1)
                  << " ops, rate=" << rate << " ops/s" << std::endl;
      }
      if (total_seconds_ == 0 && completed >= total_ops_) {
        break;
      }
      if (completed == last_completed && completed > 0) {
        std::cerr << "[breakdown][" << label_ << "] still running, no new completions in last interval"
                  << std::endl;
      }
      last_completed = completed;
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

  std::string label_;
  size_t total_ops_;
  size_t total_seconds_;
  const std::atomic<size_t>& completed_ops_;
  std::chrono::steady_clock::time_point start_;
  std::atomic<bool> finished_{false};
  std::thread thread_;
};

template <class Distance>
nlohmann::json run_benchmark(ComputeService<Distance>& service, const Args& args) {
  using SampleReport = service::breakdown::Report;
  using service::breakdown::aggregate_text_summary;
  using service::breakdown::report_to_json;

  nlohmann::json root;
  root["meta"] = {
    {"workload", args.workload},
    {"warmup_ops", args.warmup_ops},
    {"measure_ops", args.measure_ops},
    {"warmup_seconds", args.warmup_seconds},
    {"measure_seconds", args.measure_seconds},
    {"run_mode", (args.warmup_seconds > 0 || args.measure_seconds > 0) ? "time" : "ops"},
    {"time_completion_policy", "drain"},
    {"time_issue_policy", "bounded_by_observed_call_latency"},
    {"operation_granularity", "single_vector"},
    {"client_threads", args.client_threads},
    {"read_ratio", args.read_ratio},
    {"dim", service.config().dim},
    {"threads", service.config().num_threads},
    {"coroutines", service.config().num_coroutines},
    {"search_mode", service.config().search_mode},
  };
  const size_t dim = service.config().dim;
  const size_t bootstrap_work = args.measure_seconds > 0
                                  ? std::max<size_t>(4096, args.client_threads * 256)
                                  : std::max<size_t>(2048, args.measure_ops);
  const size_t bootstrap_count = bootstrap_work;
  const bool needs_query_data = (args.workload == "query" || args.workload == "both" || args.workload == "mixed");
  const bool requires_rabitq_artifacts = service.config().use_rabitq_search() &&
                                         (args.workload == "insert" || args.workload == "both" || args.workload == "mixed");

  if (requires_rabitq_artifacts && !service.config().load_index) {
    throw std::runtime_error(
      "mixed/insert benchmark with search-mode=rabitq_gpu requires a preloaded offline index. "
      "Enable --load-index and provide a valid index-prefix so the .meta.json and .rotation.bin artifacts are loaded.");
  }

  std::vector<uint32_t> bootstrap_ids(bootstrap_count);
  std::iota(bootstrap_ids.begin(), bootstrap_ids.end(), 1);
  const auto bootstrap_vectors = make_dataset(bootstrap_ids, dim);

  if (needs_query_data && !service.config().load_index) {
    vec<typename ComputeService<Distance>::InsertItem> bootstrap_batch;
    bootstrap_batch.reserve(bootstrap_count);
    for (size_t i = 0; i < bootstrap_count; ++i) {
      const auto begin = bootstrap_vectors.begin() + static_cast<std::ptrdiff_t>(i * dim);
      bootstrap_batch.push_back({bootstrap_ids[i], vec<element_t>(begin, begin + static_cast<std::ptrdiff_t>(dim))});
    }
    service.insert(bootstrap_batch);
    root["meta"]["bootstrap_vectors"] = bootstrap_count;
  }

  auto run_insert_phase_ops = [&](const std::string& label, size_t ops, uint32_t start_id) {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, ops, 0);
    for (size_t op = 0; op < ops; ++op) {
      const uint32_t id = start_id + static_cast<uint32_t>(op);
      auto values = make_deterministic_vector(id, dim);
      vec<typename ComputeService<Distance>::InsertItem> insert_items;
      insert_items.reserve(1);
      insert_items.push_back({id, vec<element_t>(values.begin(), values.end())});
      service.insert(insert_items);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
    }
    reporter.finish();
  };

  auto run_insert_phase_seconds = [&](const std::string& label, size_t seconds, uint32_t start_id) -> uint32_t {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, 0, seconds);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    uint32_t current_id = start_id;
    std::chrono::nanoseconds avg_insert_duration{0};
    size_t local_completed = 0;
    while (can_start_timed_operation(deadline, avg_insert_duration, local_completed)) {
      const uint32_t id = current_id++;
      auto values = make_deterministic_vector(id, dim);
      vec<typename ComputeService<Distance>::InsertItem> insert_items;
      insert_items.reserve(1);
      insert_items.push_back({id, vec<element_t>(values.begin(), values.end())});
      const auto started_at = std::chrono::steady_clock::now();
      service.insert(insert_items);
      update_avg_duration(avg_insert_duration, started_at, local_completed);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
      ++local_completed;
    }
    reporter.finish();
    return current_id;
  };

  std::vector<float> query_data;
  size_t query_count = 0;
  if (!args.query_file.empty()) {
    std::ifstream probe(args.query_file, std::ios::binary);
    if (!probe.good()) {
      throw std::runtime_error("query file does not exist: " + args.query_file);
    }
    uint32_t file_dim = 0;
    query_data = read_fbin(args.query_file, &file_dim, &query_count);
    if (file_dim != dim) {
      throw std::runtime_error("query dim mismatch with service config");
    }
  } else {
    query_count = std::max<size_t>(
      bootstrap_count,
      args.measure_seconds > 0 ? args.client_threads * 4096 : args.measure_ops * args.client_threads);
    std::vector<uint32_t> query_ids(query_count);
    std::iota(query_ids.begin(), query_ids.end(), bootstrap_count + 1);
    query_data = make_dataset(query_ids, dim);
    root["meta"]["synthetic_query_vectors"] = query_count;
  }

  auto run_query_phase_ops = [&](const std::string& label, size_t ops) {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, ops, 0);
    for (size_t op = 0; op < ops; ++op) {
      const size_t idx = op % query_count;
      std::vector<float> query(query_data.begin() + static_cast<std::ptrdiff_t>(idx * dim),
                               query_data.begin() + static_cast<std::ptrdiff_t>((idx + 1) * dim));
      (void)service.search(query, service.config().k);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
    }
    reporter.finish();
  };

  auto run_query_phase_seconds = [&](const std::string& label, size_t seconds) {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, 0, seconds);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    size_t op = 0;
    std::chrono::nanoseconds avg_query_duration{0};
    while (can_start_timed_operation(deadline, avg_query_duration, op)) {
      const size_t idx = op % query_count;
      std::vector<float> query(query_data.begin() + static_cast<std::ptrdiff_t>(idx * dim),
                               query_data.begin() + static_cast<std::ptrdiff_t>((idx + 1) * dim));
      const auto started_at = std::chrono::steady_clock::now();
      (void)service.search(query, service.config().k);
      update_avg_duration(avg_query_duration, started_at, op);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
      ++op;
    }
    reporter.finish();
  };

  auto run_mixed_phase_ops = [&](const std::string& label, size_t ops, uint32_t start_id) -> MixedPhaseStats {
    std::atomic<size_t> next_op{0};
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> next_insert_id{start_id};
    std::atomic<size_t> next_query_idx{0};
    std::atomic<size_t> issued_reads{0};
    std::atomic<size_t> issued_writes{0};
    std::atomic<size_t> completed_reads{0};
    std::atomic<size_t> completed_writes{0};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.client_threads));
    std::vector<std::thread> threads;
    threads.reserve(args.client_threads);
    ProgressReporter reporter(label, completed_ops, ops, 0);

    const size_t read_period = args.read_ratio >= 1.0 ? 1 : (args.read_ratio <= 0.0 ? std::numeric_limits<size_t>::max()
                                                                                    : static_cast<size_t>(1.0 / (1.0 - args.read_ratio)));

    for (size_t tid = 0; tid < args.client_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        start_barrier.arrive_and_wait();
        for (;;) {
          const size_t op_index = next_op.fetch_add(1, std::memory_order_relaxed);
          if (op_index >= ops) {
            break;
          }

          bool do_read = true;
          if (args.read_ratio <= 0.0) {
            do_read = false;
          } else if (args.read_ratio < 1.0) {
            do_read = ((op_index + 1) % read_period) != 0;
          }

          if (do_read) {
            issued_reads.fetch_add(1, std::memory_order_relaxed);
            const size_t query_idx = next_query_idx.fetch_add(1, std::memory_order_relaxed) % query_count;
            std::vector<float> query(query_data.begin() + static_cast<std::ptrdiff_t>(query_idx * dim),
                                     query_data.begin() + static_cast<std::ptrdiff_t>((query_idx + 1) * dim));
            (void)service.search(query, service.config().k);
            completed_reads.fetch_add(1, std::memory_order_relaxed);
          } else {
            issued_writes.fetch_add(1, std::memory_order_relaxed);
            const uint32_t id = next_insert_id.fetch_add(1, std::memory_order_relaxed);
            auto values = make_deterministic_vector(id, dim);
            vec<typename ComputeService<Distance>::InsertItem> insert_items;
            insert_items.reserve(1);
            insert_items.push_back({id, vec<element_t>(values.begin(), values.end())});
            (void)service.insert(insert_items);
            completed_writes.fetch_add(1, std::memory_order_relaxed);
          }
          completed_ops.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    reporter.finish();
    return MixedPhaseStats{
      .next_insert_id = next_insert_id.load(std::memory_order_relaxed),
      .issued_reads = issued_reads.load(std::memory_order_relaxed),
      .issued_writes = issued_writes.load(std::memory_order_relaxed),
      .completed_reads = completed_reads.load(std::memory_order_relaxed),
      .completed_writes = completed_writes.load(std::memory_order_relaxed),
    };
  };

  auto run_mixed_phase_seconds = [&](const std::string& label, size_t seconds, uint32_t start_id) -> MixedPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> next_insert_id{start_id};
    std::atomic<size_t> next_query_idx{0};
    std::atomic<size_t> issued_reads{0};
    std::atomic<size_t> issued_writes{0};
    std::atomic<size_t> completed_reads{0};
    std::atomic<size_t> completed_writes{0};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.client_threads));
    std::vector<std::thread> threads;
    threads.reserve(args.client_threads);
    ProgressReporter reporter(label, completed_ops, 0, seconds);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);

    const size_t read_period = args.read_ratio >= 1.0 ? 1 : (args.read_ratio <= 0.0 ? std::numeric_limits<size_t>::max()
                                                                                    : static_cast<size_t>(1.0 / (1.0 - args.read_ratio)));

    for (size_t tid = 0; tid < args.client_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        start_barrier.arrive_and_wait();
        size_t local_op_index = tid;
        std::chrono::nanoseconds avg_read_duration{0};
        std::chrono::nanoseconds avg_write_duration{0};
        size_t local_reads = 0;
        size_t local_writes = 0;
        for (;;) {
          bool do_read = true;
          if (args.read_ratio <= 0.0) {
            do_read = false;
          } else if (args.read_ratio < 1.0) {
            do_read = ((local_op_index + 1) % read_period) != 0;
          }

          if (do_read) {
            if (!can_start_timed_operation(deadline, avg_read_duration, local_reads)) {
              break;
            }
          } else {
            if (!can_start_timed_operation(deadline, avg_write_duration, local_writes)) {
              break;
            }
          }

          if (do_read) {
            issued_reads.fetch_add(1, std::memory_order_relaxed);
            const size_t query_idx = next_query_idx.fetch_add(1, std::memory_order_relaxed) % query_count;
            std::vector<float> query(query_data.begin() + static_cast<std::ptrdiff_t>(query_idx * dim),
                                     query_data.begin() + static_cast<std::ptrdiff_t>((query_idx + 1) * dim));
            const auto started_at = std::chrono::steady_clock::now();
            (void)service.search(query, service.config().k);
            update_avg_duration(avg_read_duration, started_at, local_reads);
            ++local_reads;
            completed_reads.fetch_add(1, std::memory_order_relaxed);
          } else {
            issued_writes.fetch_add(1, std::memory_order_relaxed);
            const uint32_t id = next_insert_id.fetch_add(1, std::memory_order_relaxed);
            auto values = make_deterministic_vector(id, dim);
            vec<typename ComputeService<Distance>::InsertItem> insert_items;
            insert_items.reserve(1);
            insert_items.push_back({id, vec<element_t>(values.begin(), values.end())});
            const auto started_at = std::chrono::steady_clock::now();
            (void)service.insert(insert_items);
            update_avg_duration(avg_write_duration, started_at, local_writes);
            ++local_writes;
            completed_writes.fetch_add(1, std::memory_order_relaxed);
          }
          completed_ops.fetch_add(1, std::memory_order_relaxed);
          local_op_index += args.client_threads;
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    reporter.finish();
    return MixedPhaseStats{
      .next_insert_id = next_insert_id.load(std::memory_order_relaxed),
      .issued_reads = issued_reads.load(std::memory_order_relaxed),
      .issued_writes = issued_writes.load(std::memory_order_relaxed),
      .completed_reads = completed_reads.load(std::memory_order_relaxed),
      .completed_writes = completed_writes.load(std::memory_order_relaxed),
    };
  };

  uint32_t next_insert_id = static_cast<uint32_t>(bootstrap_count + 10'000);
  const bool use_time_mode = args.warmup_seconds > 0 || args.measure_seconds > 0;
  MixedPhaseStats warmup_mixed_stats{};
  MixedPhaseStats measure_mixed_stats{};

  if (args.workload == "insert" || args.workload == "both") {
    if (use_time_mode) {
      next_insert_id = run_insert_phase_seconds("warmup-insert", args.warmup_seconds, next_insert_id) + 1024;
    } else {
      run_insert_phase_ops("warmup-insert", args.warmup_ops, next_insert_id);
      next_insert_id += static_cast<uint32_t>(args.warmup_ops + 1024);
    }
  }
  if (args.workload == "query" || args.workload == "both") {
    if (use_time_mode) {
      run_query_phase_seconds("warmup-query", args.warmup_seconds);
    } else {
      run_query_phase_ops("warmup-query", args.warmup_ops);
    }
  }
  if (args.workload == "mixed") {
    if (use_time_mode) {
      warmup_mixed_stats = run_mixed_phase_seconds("warmup-mixed", args.warmup_seconds, next_insert_id);
      next_insert_id = warmup_mixed_stats.next_insert_id + 1024;
    } else {
      warmup_mixed_stats = run_mixed_phase_ops("warmup-mixed", args.warmup_ops, next_insert_id);
      next_insert_id = warmup_mixed_stats.next_insert_id + 1024;
    }
  }

  service.clear_thread_statistics();
  service.reset_breakdown_state();

  if (args.workload == "insert" || args.workload == "both") {
    if (use_time_mode) {
      next_insert_id = run_insert_phase_seconds("measure-insert", args.measure_seconds, next_insert_id);
    } else {
      run_insert_phase_ops("measure-insert", args.measure_ops, next_insert_id);
    }
  }
  if (args.workload == "query" || args.workload == "both") {
    if (use_time_mode) {
      run_query_phase_seconds("measure-query", args.measure_seconds);
    } else {
      run_query_phase_ops("measure-query", args.measure_ops);
    }
  }
  if (args.workload == "mixed") {
    if (use_time_mode) {
      measure_mixed_stats = run_mixed_phase_seconds("measure-mixed", args.measure_seconds, next_insert_id);
      next_insert_id = measure_mixed_stats.next_insert_id;
    } else {
      measure_mixed_stats = run_mixed_phase_ops("measure-mixed", args.measure_ops, next_insert_id);
      next_insert_id = measure_mixed_stats.next_insert_id;
    }
  }

  if (args.workload == "mixed") {
    root["meta"]["warmup_mixed"] = {
      {"issued_reads", warmup_mixed_stats.issued_reads},
      {"issued_writes", warmup_mixed_stats.issued_writes},
      {"completed_reads", warmup_mixed_stats.completed_reads},
      {"completed_writes", warmup_mixed_stats.completed_writes},
    };
    root["meta"]["measure_mixed"] = {
      {"issued_reads", measure_mixed_stats.issued_reads},
      {"issued_writes", measure_mixed_stats.issued_writes},
      {"completed_reads", measure_mixed_stats.completed_reads},
      {"completed_writes", measure_mixed_stats.completed_writes},
    };
    std::cerr << "[breakdown][measure-mixed] reads issued/completed=" << measure_mixed_stats.issued_reads << "/"
              << measure_mixed_stats.completed_reads << ", writes issued/completed="
              << measure_mixed_stats.issued_writes << "/" << measure_mixed_stats.completed_writes << std::endl;
    lib_assert(measure_mixed_stats.completed_reads > 0, "mixed benchmark completed zero reads");
    lib_assert(measure_mixed_stats.completed_writes > 0, "mixed benchmark completed zero writes");
  }

  const SampleReport report = service.collect_breakdown_report();
  root.update(report_to_json(report));

  nlohmann::json summaries = nlohmann::json::object();
  std::ostringstream text_summary;
  if (report.has_insert()) {
    const auto summary = aggregate_text_summary(report.insert);
    summaries["insert"] = summary;
    text_summary << summary;
  }
  if (report.has_query()) {
    const auto summary = aggregate_text_summary(report.query);
    summaries["query"] = summary;
    text_summary << summary;
  }
  root["bottleneck_summary"] = std::move(summaries);
  root["system_counters"] = {
    {"rdma_read_bytes", report.query.counters.rdma_read_bytes + report.insert.counters.rdma_read_bytes},
    {"rdma_write_bytes", report.query.counters.rdma_write_bytes + report.insert.counters.rdma_write_bytes},
    {"h2d_bytes", report.query.counters.h2d_bytes + report.insert.counters.h2d_bytes},
    {"d2h_bytes", report.query.counters.d2h_bytes + report.insert.counters.d2h_bytes},
  };

  std::ofstream json_output(args.report_json_path);
  json_output << root.dump(2) << '\n';
  if (!json_output) {
    throw std::runtime_error("failed to write report json");
  }

  if (!args.report_text_path.empty()) {
    std::ofstream text_output(args.report_text_path);
    text_output << text_summary.str();
  }

  std::cout << text_summary.str();
  return root;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  auto service_args = build_service_argv(args.service_config_path);
  auto service_argv = make_argv(service_args);
  configuration::IndexConfiguration config(static_cast<int>(service_argv.size()), service_argv.data());

  try {
    if (config.ip_distance) {
      ComputeService<IPDistance> service(config, false);
      (void)run_benchmark(service, args);
    } else {
      ComputeService<L2Distance> service(config, false);
      (void)run_benchmark(service, args);
    }
  } catch (const std::exception& e) {
    std::cerr << "breakdown benchmark failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
