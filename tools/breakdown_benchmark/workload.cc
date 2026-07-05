#include "tools/breakdown_benchmark/workload.hh"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/distance.hh"
#include "common/vector_dtype.hh"
#include "service/breakdown.hh"
#include "tools/breakdown_benchmark/progress.hh"
#include "vamana/vamana_node.hh"

namespace tools::breakdown_benchmark {

struct MixedPhaseStats {
  uint32_t next_insert_id{};
  size_t issued_reads{};
  size_t issued_writes{};
  size_t issued_inserts{};
  size_t issued_upserts{};
  size_t issued_deletes{};
  size_t completed_reads{};
  size_t completed_writes{};
  size_t completed_inserts{};
  size_t completed_upserts{};
  size_t completed_deletes{};
};

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

struct VectorRows {
  VectorDType dtype{VectorDType::float32};
  uint32_t dim{};
  size_t count{};
  size_t vector_bytes{};
  std::vector<byte_t> raw;
  std::vector<float> decoded;

  const byte_t* raw_row(size_t index) const {
    return raw.data() + index * vector_bytes;
  }
};

VectorRows read_vector_rows(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }

  uint32_t count = 0;
  uint32_t dim = 0;
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!input) {
    throw std::runtime_error("failed to read vector file header: " + path);
  }

  VectorRows rows;
  rows.dtype = resolve_vector_dtype_config("auto", filepath_t{path});
  rows.dim = dim;
  rows.count = count;
  rows.vector_bytes = vector_dtype_bytes(rows.dtype, dim);
  rows.raw.resize(static_cast<size_t>(count) * rows.vector_bytes);
  rows.decoded.resize(static_cast<size_t>(count) * dim);

  input.read(reinterpret_cast<char*>(rows.raw.data()), static_cast<std::streamsize>(rows.raw.size()));
  if (!input) {
    throw std::runtime_error("failed to read vector payload: " + path);
  }

  for (size_t row = 0; row < rows.count; ++row) {
    decode_storage_vector_to_float(rows.raw_row(row), rows.dtype, rows.dim,
                                   rows.decoded.data() + row * rows.dim);
  }
  return rows;
}

VectorRows make_float_query_rows(const std::vector<float>& values, uint32_t dim) {
  VectorRows rows;
  rows.dtype = VectorDType::float32;
  rows.dim = dim;
  rows.count = dim == 0 ? 0 : values.size() / dim;
  rows.vector_bytes = vector_dtype_bytes(rows.dtype, dim);
  rows.decoded = values;
  rows.raw.resize(values.size() * sizeof(float));
  std::memcpy(rows.raw.data(), values.data(), rows.raw.size());
  return rows;
}


struct GroundTruth {
  uint32_t rows{};
  uint32_t top_k{};
  std::vector<uint32_t> ids;

  const uint32_t* row(size_t index) const {
    return ids.data() + index * top_k;
  }
};

GroundTruth read_groundtruth_bin(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open groundtruth file: " + path);
  }

  GroundTruth gt;
  input.read(reinterpret_cast<char*>(&gt.rows), sizeof(gt.rows));
  input.read(reinterpret_cast<char*>(&gt.top_k), sizeof(gt.top_k));
  if (!input || gt.rows == 0 || gt.top_k == 0) {
    throw std::runtime_error("failed to read groundtruth header: " + path);
  }

  gt.ids.resize(static_cast<size_t>(gt.rows) * gt.top_k);
  input.read(reinterpret_cast<char*>(gt.ids.data()),
             static_cast<std::streamsize>(gt.ids.size() * sizeof(uint32_t)));
  if (!input) {
    throw std::runtime_error("failed to read groundtruth ids: " + path);
  }
  return gt;
}

double recall_at(const std::vector<uint32_t>& results, const uint32_t* gt, uint32_t k) {
  std::unordered_set<uint32_t> truth;
  truth.reserve(k);
  for (uint32_t i = 0; i < k; ++i) {
    truth.insert(gt[i]);
  }

  uint32_t hits = 0;
  const size_t result_count = std::min<size_t>(results.size(), k);
  for (size_t i = 0; i < result_count; ++i) {
    if (truth.find(results[i]) != truth.end()) {
      ++hits;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}


template <class Distance>
nlohmann::json run_benchmark(ComputeService<Distance>& service, const Args& args) {
  using SampleReport = service::breakdown::Report;
  using service::breakdown::aggregate_text_summary;
  using service::breakdown::report_to_json;

  const bool use_insert_file = !args.insert_file.empty();

  nlohmann::json root;
  root["meta"] = {
    {"workload", args.workload},
    {"warmup_ops", args.warmup_ops},
    {"measure_ops", args.measure_ops},
    {"warmup_seconds", args.warmup_seconds},
    {"measure_seconds", args.measure_seconds},
    {"run_mode", (args.warmup_seconds > 0 || args.measure_seconds > 0) ? "time" : "ops"},
    {"time_completion_policy", "drain"},
    {"time_issue_policy", args.mixed_mode == "fixed_threads" ? "fixed_read_write_threads_until_deadline"
                                                        : "probabilistic_read_write_per_thread_until_deadline"},
    {"mixed_dispatch_policy", args.mixed_mode},
    {"vector_data_type", VamanaNode::vector_dtype_name()},
    {"vector_component_size", VamanaNode::vector_component_size()},
    {"vector_bytes", VamanaNode::vector_bytes()},
    {"node_size", VamanaNode::total_size()},
    {"candidate_vector_rdma_bytes", VamanaNode::vector_bytes()},
    {"effective_bytes_per_vector", VamanaNode::vector_bytes()},
    {"operation_granularity", "single_vector"},
    {"insert_vector_source", use_insert_file ? args.insert_file : "deterministic_synthetic_from_insert_id"},
    {"client_threads", args.client_threads},
    {"read_ratio", args.read_ratio},
    {"write_insert_ratio", args.write_insert_ratio},
    {"write_upsert_ratio", args.write_upsert_ratio},
    {"write_delete_ratio", args.write_delete_ratio},
    {"insert_start_id", args.insert_start_id},
    {"dim", service.config().dim},
    {"threads", service.config().num_threads},
    {"coroutines", service.config().num_coroutines},
    {"fine_grained_breakdown_enabled", service.config().enable_breakdown},
    {"device_utilization_observation_enabled",
     service.config().enable_breakdown && service.config().observe_device_utilization},
    {"search", std::string(service.config().credit_aware_expansion ? "credit_aware_" : "") +
        (service.config().use_rabitq
          ? "rabitq_rfq5_" + service.config().rabitq_mode : "exact")},
    {"credit_aware_expansion", service.config().credit_aware_expansion},
    {"credit_aware_min_k", service.config().credit_aware_expansion
        ? service.config().credit_aware_min_k : 0},
    {"credit_aware_max_k", service.config().credit_aware_expansion
        ? (service.config().credit_aware_max_k == 0
            ? service.config().expansion_batch : service.config().credit_aware_max_k)
        : 0},
    {"credit_aware_target_candidates", service.config().credit_aware_expansion
        ? service.config().credit_aware_target_candidates : 0},
    {"credit_aware_max_lookahead", service.config().credit_aware_expansion
        ? service.config().credit_aware_max_lookahead : 0},
    {"credit_aware_cost_guard", service.config().credit_aware_expansion
        ? service.config().credit_aware_cost_guard : false},
    {"credit_aware_cost_max_extra_ratio", service.config().credit_aware_expansion
        ? service.config().credit_aware_cost_max_extra_ratio : 0.0},
    {"credit_aware_cost_probe_rounds", service.config().credit_aware_expansion
        ? service.config().credit_aware_cost_probe_rounds : 0},
    {"rabitq_cache_bytes", service.rabitq_cache_bytes()},
    {"rabitq_cache_entries", service.rabitq_cache_entries()},
    {"rabitq_cache_numa_interleaved", service.rabitq_cache_numa_interleaved()},
    {"rabitq_cache_ratio",
      service.rabitq_cache_entries() == 0 ? 0.0 :
        static_cast<double>(service.rabitq_cache_bytes()) /
          (service.rabitq_cache_entries() * VamanaNode::vector_bytes())},
    {"rabitq_cache_entry_bytes", service.rabitq_cache_entry_bytes()},
    {"rabitq_cache_code_bits", service.rabitq_cache_code_bits()},
    {"rabitq_cache_override_bitmap_bytes",
     service.rabitq_cache_override_bitmap_bytes()},
    {"rabitq_cache_dynamic_live", service.rabitq_cache_dynamic_live()},
    {"rabitq_cache_dynamic_overflow", service.rabitq_cache_dynamic_overflow()},
    {"rabitq_gate_width", service.config().use_rabitq
        ? service.config().rabitq_gate_width : 0},
    {"rabitq_gate_max_width", service.config().use_rabitq
        ? service.config().rabitq_gate_max_width : 0},
    {"rabitq_gate_margin", service.config().use_rabitq
        ? service.config().rabitq_gate_margin : 0.0},
    {"rabitq_cache_max_ratio", service.config().use_rabitq
        ? service.config().rabitq_cache_max_ratio : 0.0},
    {"rabitq_mode", service.config().use_rabitq
        ? service.config().rabitq_mode : ""},
    {"rabitq_coalesce_target", service.config().use_rabitq
        ? service.config().rabitq_coalesce_target : 0},
    {"rabitq_coalesce_min", service.config().use_rabitq
        ? service.config().rabitq_coalesce_min : 0},
    {"rabitq_coalesce_wait_us", service.config().use_rabitq
        ? service.config().rabitq_coalesce_wait_us : 0},
    {"rabitq_prefetch_width", service.config().use_rabitq
        ? service.config().rabitq_prefetch_width : 0},
    {"rabitq_prefetch_min_samples", service.config().use_rabitq
        ? service.config().rabitq_prefetch_min_samples : 0},
    {"rabitq_prefetch_min_hit_ratio", service.config().use_rabitq
        ? service.config().rabitq_prefetch_min_hit_ratio : 0.0},
    {"rabitq_warmup_exact_expansions", service.config().use_rabitq
        ? service.config().rabitq_warmup_exact_expansions : 0},
    {"rabitq_audit_period", service.config().use_rabitq
        ? service.config().rabitq_audit_period : 0},
    {"rabitq_safe_epsilon", service.config().use_rabitq
        ? service.config().rabitq_safe_epsilon : 0.0},
    {"rabitq_strict_recall", service.config().use_rabitq
        ? service.config().rabitq_strict_recall : false},
  };
  const size_t dim = service.config().dim;
  const double write_ratio_sum = args.write_insert_ratio + args.write_upsert_ratio + args.write_delete_ratio;
  const double normalized_insert_ratio = args.write_insert_ratio / write_ratio_sum;
  const double normalized_upsert_ratio = args.write_upsert_ratio / write_ratio_sum;
  const double normalized_delete_ratio = args.write_delete_ratio / write_ratio_sum;
  root["meta"]["normalized_write_mix"] = {
    {"insert", normalized_insert_ratio},
    {"upsert", normalized_upsert_ratio},
    {"delete", normalized_delete_ratio},
  };
  size_t fixed_read_threads = 0;
  size_t fixed_write_threads = 0;
  if (args.workload == "mixed" && args.mixed_mode == "fixed_threads") {
    if (args.read_ratio <= 0.0) {
      fixed_read_threads = 0;
    } else if (args.read_ratio >= 1.0) {
      fixed_read_threads = args.client_threads;
    } else {
      fixed_read_threads = static_cast<size_t>(std::llround(static_cast<double>(args.client_threads) * args.read_ratio));
      fixed_read_threads = std::clamp<size_t>(fixed_read_threads, 1, args.client_threads - 1);
    }
    fixed_write_threads = args.client_threads - fixed_read_threads;
    root["meta"]["mixed_fixed_threads"] = {
      {"read_threads", fixed_read_threads},
      {"write_threads", fixed_write_threads},
    };
    std::cerr << "[breakdown] mixed fixed thread split: reads=" << fixed_read_threads
              << ", writes=" << fixed_write_threads << std::endl;
  }
  const size_t bootstrap_work = args.measure_seconds > 0
                                  ? std::max<size_t>(4096, args.client_threads * 256)
                                  : std::max<size_t>(2048, args.measure_ops);
  const size_t bootstrap_count = bootstrap_work;
  std::cerr << "[breakdown] preparing workload: bootstrap_count=" << bootstrap_count
            << ", workload=" << args.workload << std::endl;
  const bool needs_query_data = (args.workload == "query" || args.workload == "both" || args.workload == "mixed");

  // Load insert vectors from file if specified, otherwise use synthetic data.
  VectorRows insert_rows;
  if (use_insert_file) {
    insert_rows = read_vector_rows(args.insert_file);
    if (insert_rows.dim != dim) {
      throw std::runtime_error("insert-file dim mismatch: " + std::to_string(insert_rows.dim)
                               + " vs config " + std::to_string(dim));
    }
    std::cerr << "[breakdown] insert-file vectors: count=" << insert_rows.count
              << " dim=" << insert_rows.dim << " dtype=" << vector_dtype_name(insert_rows.dtype)
              << std::endl;
  }

  auto get_insert_vector = [&](uint32_t id) -> vec<element_t> {
    if (use_insert_file) {
      size_t row = id % insert_rows.count;
      return vec<element_t>(insert_rows.decoded.begin() + row * dim,
                            insert_rows.decoded.begin() + (row + 1) * dim);
    }
    auto gen = make_deterministic_vector(id, dim);
    return vec<element_t>(gen.begin(), gen.end());
  };

  auto get_update_vector = [&](uint32_t target_id, uint32_t version) -> vec<element_t> {
    return get_insert_vector(target_id ^ (0x9e3779b9u * (version + 1u)));
  };

  std::vector<uint32_t> bootstrap_ids(bootstrap_count);
  std::iota(bootstrap_ids.begin(), bootstrap_ids.end(), 1);
  const auto bootstrap_vectors = make_dataset(bootstrap_ids, dim);
  std::cerr << "[breakdown] bootstrap vectors ready (synthetic)" << std::endl;

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
      vec<element_t> values = get_insert_vector(id);
      vec<typename ComputeService<Distance>::InsertItem> insert_items;
      insert_items.reserve(1);
      insert_items.push_back({id, std::move(values)});
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
      vec<element_t> values = get_insert_vector(id);
      vec<typename ComputeService<Distance>::InsertItem> insert_items;
      insert_items.reserve(1);
      insert_items.push_back({id, std::move(values)});
      const auto started_at = std::chrono::steady_clock::now();
      service.insert(insert_items);
      update_avg_duration(avg_insert_duration, started_at, local_completed);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
      ++local_completed;
    }
    reporter.finish();
    return current_id;
  };

  VectorRows query_rows;
  size_t query_count = 0;
  if (!args.query_file.empty()) {
    std::ifstream probe(args.query_file, std::ios::binary);
    if (!probe.good()) {
      throw std::runtime_error("query file does not exist: " + args.query_file);
    }
    query_rows = read_vector_rows(args.query_file);
    query_count = query_rows.count;
    if (query_rows.dim != dim) {
      throw std::runtime_error("query dim mismatch with service config");
    }
  } else {
    query_count = std::max<size_t>(
      bootstrap_count,
      args.measure_seconds > 0 ? args.client_threads * 4096 : args.measure_ops * args.client_threads);
    std::vector<uint32_t> query_ids(query_count);
    std::iota(query_ids.begin(), query_ids.end(), bootstrap_count + 1);
    query_rows = make_float_query_rows(make_dataset(query_ids, dim), static_cast<uint32_t>(dim));
    root["meta"]["synthetic_query_vectors"] = query_count;
  }
  root["meta"]["query_data_type"] = vector_dtype_name(query_rows.dtype);
  root["meta"]["query_vector_bytes"] = query_rows.vector_bytes;
  std::cerr << "[breakdown] query data ready: count=" << query_count
            << " dtype=" << vector_dtype_name(query_rows.dtype)
            << " vector_bytes=" << query_rows.vector_bytes << std::endl;

  bool recall_below_threshold = false;
  auto run_recall_check = [&](const char* phase,
                              const char* key,
                              bool reset_after,
                              bool enforce_threshold) {
    if (args.groundtruth_file.empty()) {
      return;
    }
    if (query_count == 0) {
      throw std::runtime_error("recall requires query vectors");
    }
    const GroundTruth gt = read_groundtruth_bin(args.groundtruth_file);
    if (gt.rows != query_count) {
      throw std::runtime_error("query/groundtruth row count mismatch");
    }
    const uint32_t recall_k = args.recall_k == 0 ? std::min<uint32_t>(service.config().k, gt.top_k) : args.recall_k;
    if (recall_k == 0 || recall_k > gt.top_k) {
      throw std::runtime_error("invalid recall k");
    }
    const size_t recall_queries = args.recall_queries == 0 ? query_count : std::min<size_t>(args.recall_queries, query_count);
    double total_recall = 0.0;
    std::atomic<size_t> recall_completed{0};
    ProgressReporter recall_reporter(key, recall_completed, recall_queries, 0);
    for (size_t qi = 0; qi < recall_queries; ++qi) {
      const auto results = service.search_raw(query_rows.dtype, query_rows.raw_row(qi), dim, recall_k);
      std::vector<uint32_t> result_ids;
      result_ids.reserve(results.size());
      for (const auto id : results) {
        result_ids.push_back(static_cast<uint32_t>(id));
      }
      total_recall += recall_at(result_ids, gt.row(qi), recall_k);
      recall_completed.fetch_add(1, std::memory_order_relaxed);
    }
    recall_reporter.finish();
    const double recall = recall_queries > 0 ? total_recall / static_cast<double>(recall_queries) : 0.0;
    root[key] = {
      {"phase", phase},
      {"groundtruth_file", args.groundtruth_file},
      {"queries", recall_queries},
      {"k", recall_k},
      {"recall", recall},
      {"min_recall", args.min_recall},
      {"passed", args.min_recall < 0.0 || recall >= args.min_recall},
    };
    std::cerr << "[breakdown][recall] " << phase << " recall@" << recall_k << "=" << recall
              << " queries=" << recall_queries << std::endl;
    if (enforce_threshold && args.min_recall >= 0.0 && recall < args.min_recall) {
      recall_below_threshold = true;
    }

    if (reset_after) {
      service.clear_thread_statistics();
      service.reset_breakdown_state();
    }
  };
  run_recall_check("before_performance", "recall", true, true);

  auto run_query_phase_ops = [&](const std::string& label, size_t ops) -> size_t {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, ops, 0);
    for (size_t op = 0; op < ops; ++op) {
      const size_t idx = op % query_count;
      (void)service.search_raw(query_rows.dtype, query_rows.raw_row(idx), dim, service.config().k);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
    }
    reporter.finish();
    return completed_ops.load(std::memory_order_relaxed);
  };

  auto run_query_phase_seconds = [&](const std::string& label, size_t seconds) -> size_t {
    std::atomic<size_t> completed_ops{0};
    ProgressReporter reporter(label, completed_ops, 0, seconds);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    size_t op = 0;
    std::chrono::nanoseconds avg_query_duration{0};
    while (can_start_timed_operation(deadline, avg_query_duration, op)) {
      const size_t idx = op % query_count;
      const auto started_at = std::chrono::steady_clock::now();
      (void)service.search_raw(query_rows.dtype, query_rows.raw_row(idx), dim, service.config().k);
      update_avg_duration(avg_query_duration, started_at, op);
      completed_ops.fetch_add(1, std::memory_order_relaxed);
      ++op;
    }
    reporter.finish();
    return completed_ops.load(std::memory_order_relaxed);
  };

  auto choose_mixed_read = [&](std::mt19937_64& rng) {
    if (args.read_ratio <= 0.0) {
      return false;
    }
    if (args.read_ratio >= 1.0) {
      return true;
    }
    std::bernoulli_distribution read_dist(args.read_ratio);
    return read_dist(rng);
  };

  enum class WriteKind { insert, upsert, erase };
  auto choose_write_kind = [&](std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double pick = dist(rng);
    if (pick < normalized_insert_ratio) return WriteKind::insert;
    if (pick < normalized_insert_ratio + normalized_upsert_ratio) return WriteKind::upsert;
    return WriteKind::erase;
  };

  const uint32_t base_id_count = static_cast<uint32_t>(
    std::max<size_t>(1, service.config().max_vectors));
  auto sample_existing_id = [&](std::mt19937_64& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, base_id_count - 1);
    return dist(rng);
  };

  auto issue_mixed_write = [&](std::mt19937_64& rng,
                               std::atomic<uint32_t>& next_insert_id,
                               std::atomic<uint32_t>& next_update_version,
                               std::atomic<size_t>& issued_inserts,
                               std::atomic<size_t>& issued_upserts,
                               std::atomic<size_t>& issued_deletes,
                               std::atomic<size_t>& completed_inserts,
                               std::atomic<size_t>& completed_upserts,
                               std::atomic<size_t>& completed_deletes) {
    switch (choose_write_kind(rng)) {
      case WriteKind::insert: {
        issued_inserts.fetch_add(1, std::memory_order_relaxed);
        const uint32_t id = next_insert_id.fetch_add(1, std::memory_order_relaxed);
        vec<element_t> values = get_insert_vector(id);
        vec<typename ComputeService<Distance>::InsertItem> items;
        items.push_back({id, std::move(values)});
        completed_inserts.fetch_add(service.insert(items), std::memory_order_relaxed);
        break;
      }
      case WriteKind::upsert: {
        issued_upserts.fetch_add(1, std::memory_order_relaxed);
        const uint32_t id = sample_existing_id(rng);
        const uint32_t version = next_update_version.fetch_add(1, std::memory_order_relaxed);
        vec<element_t> values = get_update_vector(id, version);
        vec<typename ComputeService<Distance>::InsertItem> items;
        items.push_back({id, std::move(values)});
        completed_upserts.fetch_add(service.upsert(items), std::memory_order_relaxed);
        break;
      }
      case WriteKind::erase: {
        issued_deletes.fetch_add(1, std::memory_order_relaxed);
        const uint32_t id = sample_existing_id(rng);
        vec<node_t> ids;
        ids.push_back(id);
        completed_deletes.fetch_add(service.erase(ids), std::memory_order_relaxed);
        break;
      }
    }
  };

  auto run_mixed_phase_ops = [&](const std::string& label, size_t ops, uint32_t start_id) -> MixedPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> next_insert_id{start_id};
    std::atomic<size_t> next_query_idx{0};
    std::atomic<size_t> issued_reads{0};
    std::atomic<size_t> issued_writes{0};
    std::atomic<size_t> issued_inserts{0};
    std::atomic<size_t> issued_upserts{0};
    std::atomic<size_t> issued_deletes{0};
    std::atomic<size_t> completed_reads{0};
    std::atomic<size_t> completed_writes{0};
    std::atomic<size_t> completed_inserts{0};
    std::atomic<size_t> completed_upserts{0};
    std::atomic<size_t> completed_deletes{0};
    std::atomic<size_t> next_op{0};
    std::atomic<uint32_t> next_update_version{0};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.client_threads));
    std::vector<std::thread> threads;
    threads.reserve(args.client_threads);
    ProgressReporter reporter(label, completed_ops, ops, 0);

    for (size_t tid = 0; tid < args.client_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        std::mt19937_64 rng(0x9e3779b97f4a7c15ull ^
                            (static_cast<uint64_t>(tid) << 32) ^
                            static_cast<uint64_t>(std::hash<std::string>{}(label)));
        start_barrier.arrive_and_wait();
        for (;;) {
          const size_t op_index = next_op.fetch_add(1, std::memory_order_relaxed);
          if (op_index >= ops) {
            break;
          }

          const bool read_op = args.mixed_mode == "fixed_threads" ? tid < fixed_read_threads : choose_mixed_read(rng);
          if (read_op) {
            issued_reads.fetch_add(1, std::memory_order_relaxed);
            const size_t query_idx = next_query_idx.fetch_add(1, std::memory_order_relaxed) % query_count;
            (void)service.search_raw(query_rows.dtype, query_rows.raw_row(query_idx), dim, service.config().k);
            completed_reads.fetch_add(1, std::memory_order_relaxed);
          } else {
            issued_writes.fetch_add(1, std::memory_order_relaxed);
            issue_mixed_write(rng, next_insert_id, next_update_version,
                              issued_inserts, issued_upserts, issued_deletes,
                              completed_inserts, completed_upserts, completed_deletes);
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
      .issued_inserts = issued_inserts.load(std::memory_order_relaxed),
      .issued_upserts = issued_upserts.load(std::memory_order_relaxed),
      .issued_deletes = issued_deletes.load(std::memory_order_relaxed),
      .completed_reads = completed_reads.load(std::memory_order_relaxed),
      .completed_writes = completed_writes.load(std::memory_order_relaxed),
      .completed_inserts = completed_inserts.load(std::memory_order_relaxed),
      .completed_upserts = completed_upserts.load(std::memory_order_relaxed),
      .completed_deletes = completed_deletes.load(std::memory_order_relaxed),
    };
  };

  auto run_mixed_phase_seconds = [&](const std::string& label, size_t seconds, uint32_t start_id) -> MixedPhaseStats {
    std::atomic<size_t> completed_ops{0};
    std::atomic<uint32_t> next_insert_id{start_id};
    std::atomic<size_t> next_query_idx{0};
    std::atomic<size_t> issued_reads{0};
    std::atomic<size_t> issued_writes{0};
    std::atomic<size_t> issued_inserts{0};
    std::atomic<size_t> issued_upserts{0};
    std::atomic<size_t> issued_deletes{0};
    std::atomic<size_t> completed_reads{0};
    std::atomic<size_t> completed_writes{0};
    std::atomic<size_t> completed_inserts{0};
    std::atomic<size_t> completed_upserts{0};
    std::atomic<size_t> completed_deletes{0};
    std::atomic<uint32_t> next_update_version{0};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.client_threads));
    std::vector<std::thread> threads;
    threads.reserve(args.client_threads);
    ProgressReporter reporter(label, completed_ops, 0, seconds);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);

    for (size_t tid = 0; tid < args.client_threads; ++tid) {
      threads.emplace_back([&, tid]() {
        std::mt19937_64 rng(0xd1b54a32d192ed03ull ^
                            (static_cast<uint64_t>(tid) << 32) ^
                            static_cast<uint64_t>(std::hash<std::string>{}(label)));
        start_barrier.arrive_and_wait();
        for (;;) {
          if (std::chrono::steady_clock::now() >= deadline) {
            break;
          }

          const bool read_op = args.mixed_mode == "fixed_threads" ? tid < fixed_read_threads : choose_mixed_read(rng);
          if (read_op) {
            issued_reads.fetch_add(1, std::memory_order_relaxed);
            const size_t query_idx = next_query_idx.fetch_add(1, std::memory_order_relaxed) % query_count;
            (void)service.search_raw(query_rows.dtype, query_rows.raw_row(query_idx), dim, service.config().k);
            completed_reads.fetch_add(1, std::memory_order_relaxed);
          } else {
            issued_writes.fetch_add(1, std::memory_order_relaxed);
            issue_mixed_write(rng, next_insert_id, next_update_version,
                              issued_inserts, issued_upserts, issued_deletes,
                              completed_inserts, completed_upserts, completed_deletes);
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
      .issued_inserts = issued_inserts.load(std::memory_order_relaxed),
      .issued_upserts = issued_upserts.load(std::memory_order_relaxed),
      .issued_deletes = issued_deletes.load(std::memory_order_relaxed),
      .completed_reads = completed_reads.load(std::memory_order_relaxed),
      .completed_writes = completed_writes.load(std::memory_order_relaxed),
      .completed_inserts = completed_inserts.load(std::memory_order_relaxed),
      .completed_upserts = completed_upserts.load(std::memory_order_relaxed),
      .completed_deletes = completed_deletes.load(std::memory_order_relaxed),
    };
  };

  uint32_t next_insert_id = args.insert_start_id;
  if (next_insert_id == 0) {
    const uint64_t default_start = service.config().load_index
                                     ? static_cast<uint64_t>(service.config().max_vectors) + 10'000ull
                                     : static_cast<uint64_t>(bootstrap_count) + 10'000ull;
    if (default_start > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("default insert start id exceeds uint32_t; pass --insert-start-id explicitly");
    }
    next_insert_id = static_cast<uint32_t>(default_start);
  }
  root["meta"]["effective_insert_start_id"] = next_insert_id;
  std::cerr << "[breakdown] effective insert start id=" << next_insert_id << std::endl;
  const bool use_time_mode = args.warmup_seconds > 0 || args.measure_seconds > 0;
  MixedPhaseStats warmup_mixed_stats{};
  MixedPhaseStats measure_mixed_stats{};

  if (args.workload == "insert" || args.workload == "both") {
    std::cerr << "[breakdown] starting warmup insert" << std::endl;
    if (use_time_mode) {
      next_insert_id = run_insert_phase_seconds("warmup-insert", args.warmup_seconds, next_insert_id) + 1024;
    } else {
      run_insert_phase_ops("warmup-insert", args.warmup_ops, next_insert_id);
      next_insert_id += static_cast<uint32_t>(args.warmup_ops + 1024);
    }
  }
  if (args.workload == "query" || args.workload == "both") {
    std::cerr << "[breakdown] starting warmup query" << std::endl;
    if (use_time_mode) {
      (void)run_query_phase_seconds("warmup-query", args.warmup_seconds);
    } else {
      (void)run_query_phase_ops("warmup-query", args.warmup_ops);
    }
  }
  if (args.workload == "mixed") {
    std::cerr << "[breakdown] starting warmup mixed" << std::endl;
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
  std::cerr << "[breakdown] starting measure phase" << std::endl;
  size_t measured_query_operations = 0;
  size_t measured_insert_operations = 0;

  if (args.workload == "insert" || args.workload == "both") {
    if (use_time_mode) {
      const uint32_t start_id = next_insert_id;
      next_insert_id = run_insert_phase_seconds("measure-insert", args.measure_seconds, next_insert_id);
      measured_insert_operations = next_insert_id - start_id;
    } else {
      run_insert_phase_ops("measure-insert", args.measure_ops, next_insert_id);
      measured_insert_operations = args.measure_ops;
    }
  }
  if (args.workload == "query" || args.workload == "both") {
    if (use_time_mode) {
      measured_query_operations = run_query_phase_seconds("measure-query", args.measure_seconds);
    } else {
      measured_query_operations = run_query_phase_ops("measure-query", args.measure_ops);
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
      {"issued_inserts", warmup_mixed_stats.issued_inserts},
      {"issued_upserts", warmup_mixed_stats.issued_upserts},
      {"issued_deletes", warmup_mixed_stats.issued_deletes},
      {"completed_reads", warmup_mixed_stats.completed_reads},
      {"completed_writes", warmup_mixed_stats.completed_writes},
      {"completed_inserts", warmup_mixed_stats.completed_inserts},
      {"completed_upserts", warmup_mixed_stats.completed_upserts},
      {"completed_deletes", warmup_mixed_stats.completed_deletes},
    };
    root["meta"]["measure_mixed"] = {
      {"issued_reads", measure_mixed_stats.issued_reads},
      {"issued_writes", measure_mixed_stats.issued_writes},
      {"issued_inserts", measure_mixed_stats.issued_inserts},
      {"issued_upserts", measure_mixed_stats.issued_upserts},
      {"issued_deletes", measure_mixed_stats.issued_deletes},
      {"completed_reads", measure_mixed_stats.completed_reads},
      {"completed_writes", measure_mixed_stats.completed_writes},
      {"completed_inserts", measure_mixed_stats.completed_inserts},
      {"completed_upserts", measure_mixed_stats.completed_upserts},
      {"completed_deletes", measure_mixed_stats.completed_deletes},
    };
    std::cerr << "[breakdown][measure-mixed] reads issued/completed=" << measure_mixed_stats.issued_reads << "/"
              << measure_mixed_stats.completed_reads << ", writes issued/completed="
              << measure_mixed_stats.issued_writes << "/" << measure_mixed_stats.completed_writes
              << " (insert=" << measure_mixed_stats.completed_inserts
              << ", upsert=" << measure_mixed_stats.completed_upserts
              << ", delete=" << measure_mixed_stats.completed_deletes << ")" << std::endl;
    if (args.read_ratio > 0.0) {
      lib_assert(measure_mixed_stats.completed_reads > 0, "mixed benchmark completed zero reads");
    }
    if (args.read_ratio < 1.0) {
      lib_assert(measure_mixed_stats.completed_writes > 0, "mixed benchmark completed zero writes");
    }
  }

  const SampleReport report = service.collect_breakdown_report();
  root.update(report_to_json(report));


  const bool has_throughput_duration = use_time_mode && args.measure_seconds > 0;
  const double throughput_duration = has_throughput_duration ? static_cast<double>(args.measure_seconds) : 0.0;
  const size_t throughput_query_ops = args.workload == "mixed"
    ? measure_mixed_stats.completed_reads
    : (service.config().enable_breakdown ? report.query.count : measured_query_operations);
  const size_t throughput_write_ops = args.workload == "mixed"
    ? measure_mixed_stats.completed_writes
    : (service.config().enable_breakdown ? report.insert.count : measured_insert_operations);
  const double query_throughput = has_throughput_duration
                                    ? static_cast<double>(throughput_query_ops) / throughput_duration
                                    : 0.0;
  const double write_throughput = has_throughput_duration
                                    ? static_cast<double>(throughput_write_ops) / throughput_duration
                                    : 0.0;
  const double total_throughput = query_throughput + write_throughput;
  root["throughput"] = {
    {"duration_seconds", throughput_duration},
    {"total_ops", throughput_query_ops + throughput_write_ops},
    {"total_ops_per_sec", total_throughput},
    {"query_ops", throughput_query_ops},
    {"query_ops_per_sec", query_throughput},
    {"write_ops", throughput_write_ops},
    {"write_ops_per_sec", write_throughput},
    {"insert_ops", args.workload == "mixed" ? measure_mixed_stats.completed_inserts
                                                : (service.config().enable_breakdown
                                                    ? report.insert.count : measured_insert_operations)},
    {"insert_ops_per_sec", has_throughput_duration
      ? static_cast<double>(args.workload == "mixed"
          ? measure_mixed_stats.completed_inserts
          : (service.config().enable_breakdown ? report.insert.count
                                               : measured_insert_operations)) / throughput_duration
      : 0.0},
    {"upsert_ops", args.workload == "mixed" ? measure_mixed_stats.completed_upserts : 0},
    {"upsert_ops_per_sec", has_throughput_duration && args.workload == "mixed"
      ? static_cast<double>(measure_mixed_stats.completed_upserts) / throughput_duration
      : 0.0},
    {"delete_ops", args.workload == "mixed" ? measure_mixed_stats.completed_deletes : 0},
    {"delete_ops_per_sec", has_throughput_duration && args.workload == "mixed"
      ? static_cast<double>(measure_mixed_stats.completed_deletes) / throughput_duration
      : 0.0},
  };

  run_recall_check("after_performance", "static_gt_post_recall", false, false);

  nlohmann::json summaries = nlohmann::json::object();
  std::ostringstream text_summary;
  if (has_throughput_duration) {
    text_summary << "throughput\n";
    text_summary << "  duration_seconds: " << throughput_duration << '\n';
    text_summary << "  total_ops_per_sec: " << total_throughput
                 << " (ops=" << (throughput_query_ops + throughput_write_ops) << ")\n";
    text_summary << "  query_ops_per_sec: " << query_throughput
                 << " (ops=" << throughput_query_ops << ")\n";
    text_summary << "  write_ops_per_sec: " << write_throughput
                 << " (ops=" << throughput_write_ops << ")\n";
    if (args.workload == "mixed") {
      text_summary << "  write_mix_completed: insert=" << measure_mixed_stats.completed_inserts
                   << " upsert=" << measure_mixed_stats.completed_upserts
                   << " delete=" << measure_mixed_stats.completed_deletes << '\n';
    }
  }
  if (root.contains("recall")) {
    const auto& recall = root["recall"];
    text_summary << "recall\n";
    text_summary << "  recall@" << recall.value("k", 0) << ": "
                 << recall.value("recall", 0.0) << '\n';
    text_summary << "  queries: " << recall.value("queries", 0) << '\n';
    text_summary << "  passed: " << (recall.value("passed", false) ? "true" : "false") << '\n';
    text_summary << "  groundtruth_file: " << recall.value("groundtruth_file", "") << '\n';
  }
  if (root.contains("static_gt_post_recall")) {
    const auto& recall = root["static_gt_post_recall"];
    text_summary << "static_gt_post_recall\n";
    text_summary << "  recall@" << recall.value("k", 0) << ": "
                 << recall.value("recall", 0.0) << '\n';
    text_summary << "  queries: " << recall.value("queries", 0) << '\n';
    text_summary << "  groundtruth_file: " << recall.value("groundtruth_file", "") << '\n';
  }
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
  if (recall_below_threshold) {
    throw std::runtime_error("recall below threshold");
  }
  return root;
}



template nlohmann::json run_benchmark<L2Distance>(ComputeService<L2Distance>& service, const Args& args);
template nlohmann::json run_benchmark<IPDistance>(ComputeService<IPDistance>& service, const Args& args);

}  // namespace tools::breakdown_benchmark
