#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <execinfo.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "common/configuration.hh"
#include "common/vector_dtype.hh"
#include "nlohmann/json.hh"
#include "service/compute_service.hh"
#include "tools/breakdown_benchmark/args.hh"
#include "tools/breakdown_benchmark/progress.hh"

namespace {

using tools::breakdown_benchmark::ProgressReporter;

struct Args {
  std::string service_config_path;
  std::filesystem::path insert_file;
  uint64_t insert_start_id{100000000ull};
  size_t insert_count{1000000};
  size_t insert_row_offset{0};
  size_t insert_threads{16};
  size_t insert_batch_size{16};
  size_t reset_breakdown_every{50000};

  std::filesystem::path query_file;
  std::filesystem::path baseline_groundtruth_file;
  std::filesystem::path groundtruth_file;
  size_t recall_queries{10000};
  u32 recall_k{10};
  size_t settle_seconds{300};

  std::filesystem::path report_json_path;
  std::filesystem::path report_text_path;
};

struct VectorRows {
  VectorDType dtype{VectorDType::float32};
  u32 dim{};
  size_t count{};
  size_t vector_bytes{};
  size_t row_stride{};
  size_t row_payload_offset{};
  std::vector<byte_t> raw;

  const byte_t* row_payload(size_t index) const {
    return raw.data() + index * row_stride + row_payload_offset;
  }

  vec<element_t> decode_row(size_t index) const {
    vec<element_t> values(dim);
    decode_storage_vector_to_float(row_payload(index), dtype, dim, values.data());
    return values;
  }
};

struct GroundTruth {
  u32 rows{};
  u32 top_k{};
  std::vector<u32> ids;

  const u32* row(size_t index) const {
    return ids.data() + static_cast<size_t>(top_k) * index;
  }
};

struct RecallResult {
  double recall{};
  size_t queries{};
  u32 k{};
};

struct InsertStats {
  size_t attempted{};
  size_t inserted{};
  size_t failed{};
  double duration_seconds{};
  double inserts_per_second{};
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void signal_handler(int signal) {
  void* frames[64];
  const int count = backtrace(frames, 64);
  const char header[] = "\n[sift101m-long-insert] fatal signal, backtrace:\n";
  const ssize_t ignored = ::write(STDERR_FILENO, header, sizeof(header) - 1);
  (void)ignored;
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  _exit(128 + signal);
}

std::string extension_lower(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return ext;
}

uint64_t path_size(const std::filesystem::path& path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    fail("cannot stat " + path.string() + ": " + ec.message());
  }
  return size;
}

u32 read_u32(std::ifstream& input, const std::string& what) {
  u32 value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) {
    fail("failed to read " + what);
  }
  return value;
}

u32 load_u32_unaligned(const byte_t* data) {
  u32 value{};
  std::memcpy(&value, data, sizeof(value));
  return value;
}

void read_exact(std::ifstream& input, void* dst, size_t bytes, const std::string& what) {
  input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!input) {
    fail("failed to read " + what);
  }
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
  if (path.empty()) {
    return;
  }
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream output(path);
  output << text;
  if (!output) {
    fail("failed to write " + path.string());
  }
}

void write_json_file(const std::filesystem::path& path, const nlohmann::json& root) {
  if (path.empty()) {
    fail("--report-json is required");
  }
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream output(path);
  output << root.dump(2) << '\n';
  if (!output) {
    fail("failed to write " + path.string());
  }
}

VectorRows read_header_vector_rows(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("failed to open vector file: " + path.string());
  }

  const u32 count = read_u32(input, "vector count");
  const u32 dim = read_u32(input, "vector dim");
  if (count == 0 || dim == 0) {
    fail("invalid vector header in " + path.string());
  }

  VectorRows rows;
  rows.dtype = resolve_vector_dtype_config("auto", filepath_t{path});
  rows.dim = dim;
  rows.count = count;
  rows.vector_bytes = vector_dtype_bytes(rows.dtype, dim);
  rows.row_stride = rows.vector_bytes;
  rows.row_payload_offset = 0;

  const uint64_t expected_size = 2ull * sizeof(u32) + static_cast<uint64_t>(rows.count) * rows.vector_bytes;
  const uint64_t actual_size = path_size(path);
  if (actual_size != expected_size) {
    fail("unexpected vector file size for " + path.string() + ": got " + std::to_string(actual_size) +
         ", expected " + std::to_string(expected_size));
  }

  rows.raw.resize(static_cast<size_t>(rows.count) * rows.vector_bytes);
  read_exact(input, rows.raw.data(), rows.raw.size(), "vector payload");
  return rows;
}

VectorRows read_vecs_vector_rows(const std::filesystem::path& path, VectorDType dtype) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("failed to open vecs file: " + path.string());
  }

  const uint64_t size = path_size(path);
  const u32 dim = read_u32(input, "vecs dim");
  if (dim == 0) {
    fail("invalid vecs dim in " + path.string());
  }

  const size_t vector_bytes = vector_dtype_bytes(dtype, dim);
  const size_t row_stride = sizeof(u32) + vector_bytes;
  if (size % row_stride != 0) {
    fail("vecs file size is not divisible by row size: " + path.string());
  }

  VectorRows rows;
  rows.dtype = dtype;
  rows.dim = dim;
  rows.count = static_cast<size_t>(size / row_stride);
  rows.vector_bytes = vector_bytes;
  rows.row_stride = row_stride;
  rows.row_payload_offset = sizeof(u32);
  rows.raw.resize(static_cast<size_t>(size));

  input.seekg(0, std::ios::beg);
  read_exact(input, rows.raw.data(), rows.raw.size(), "vecs payload");
  for (size_t row = 0; row < rows.count; ++row) {
    const u32 row_dim = load_u32_unaligned(rows.raw.data() + row * row_stride);
    if (row_dim != dim) {
      fail("vecs row dim mismatch in " + path.string() + " row " + std::to_string(row));
    }
  }
  return rows;
}

VectorRows read_vector_rows(const std::filesystem::path& path) {
  const std::string ext = extension_lower(path);
  if (ext == ".bvecs") {
    return read_vecs_vector_rows(path, VectorDType::uint8);
  }
  if (ext == ".fvecs") {
    return read_vecs_vector_rows(path, VectorDType::float32);
  }
  if (ext == ".u8bin" || ext == ".i8bin" || ext == ".fbin" || ext == ".bin") {
    return read_header_vector_rows(path);
  }
  fail("unsupported vector file extension: " + path.string());
}

GroundTruth read_project_groundtruth_bin(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("failed to open groundtruth file: " + path.string());
  }

  GroundTruth gt;
  gt.rows = read_u32(input, "groundtruth rows");
  gt.top_k = read_u32(input, "groundtruth top_k");
  if (gt.rows == 0 || gt.top_k == 0) {
    fail("invalid groundtruth header: " + path.string());
  }

  const uint64_t expected_size = 2ull * sizeof(u32) +
                                 static_cast<uint64_t>(gt.rows) * gt.top_k * sizeof(u32);
  const uint64_t actual_size = path_size(path);
  if (actual_size != expected_size) {
    fail("unexpected groundtruth file size for " + path.string() + ": got " + std::to_string(actual_size) +
         ", expected " + std::to_string(expected_size));
  }

  gt.ids.resize(static_cast<size_t>(gt.rows) * gt.top_k);
  read_exact(input, gt.ids.data(), gt.ids.size() * sizeof(u32), "groundtruth ids");
  return gt;
}

GroundTruth read_ivecs_groundtruth(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail("failed to open ivecs groundtruth file: " + path.string());
  }
  const uint64_t size = path_size(path);
  const u32 top_k = read_u32(input, "ivecs top_k");
  if (top_k == 0) {
    fail("invalid ivecs top_k in " + path.string());
  }
  const uint64_t row_bytes = sizeof(u32) + static_cast<uint64_t>(top_k) * sizeof(u32);
  if (size % row_bytes != 0) {
    fail("ivecs file size is not divisible by row size: " + path.string());
  }

  GroundTruth gt;
  gt.rows = static_cast<u32>(size / row_bytes);
  gt.top_k = top_k;
  gt.ids.resize(static_cast<size_t>(gt.rows) * gt.top_k);

  input.seekg(0, std::ios::beg);
  for (u32 row = 0; row < gt.rows; ++row) {
    const u32 row_top_k = read_u32(input, "ivecs row top_k");
    if (row_top_k != top_k) {
      fail("ivecs top_k mismatch in row " + std::to_string(row));
    }
    read_exact(input, gt.ids.data() + static_cast<size_t>(row) * gt.top_k,
               static_cast<size_t>(gt.top_k) * sizeof(u32), "ivecs row ids");
  }
  return gt;
}

GroundTruth read_groundtruth(const std::filesystem::path& path) {
  const std::string ext = extension_lower(path);
  if (ext == ".ivecs") {
    return read_ivecs_groundtruth(path);
  }
  if (ext == ".bin") {
    return read_project_groundtruth_bin(path);
  }
  fail("unsupported groundtruth file extension: " + path.string());
}

double recall_at_k(const std::vector<u32>& results, const u32* truth, u32 k) {
  u32 hits = 0;
  const size_t result_count = std::min<size_t>(results.size(), k);
  for (size_t i = 0; i < result_count; ++i) {
    for (u32 j = 0; j < k; ++j) {
      if (results[i] == truth[j]) {
        ++hits;
        break;
      }
    }
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --service-config <ini> --insert-file <path> "
      << "--query-file <path> --groundtruth-file <path> --report-json <path> [options]\n"
      << "  --insert-start-id <id>          First inserted ID, default 100000000\n"
      << "  --insert-count <n>              Number of rows to insert, default 1000000; 0 means all rows\n"
      << "  --insert-row-offset <n>         First row in insert file, default 0\n"
      << "  --insert-threads <n>            Concurrent application threads, default 16\n"
      << "  --insert-batch-size <n>         Vectors per ComputeService::insert call, default 16\n"
      << "  --baseline-groundtruth-file <p> Optional pre-insert GT, e.g. gnd/idx_100M.ivecs\n"
      << "  --recall-queries <n>            Queries for recall, default 10000; 0 means all\n"
      << "  --recall-k <k>                  Recall K, default 10\n"
      << "  --settle-seconds <n>            Sleep after insertion before post recall, default 300\n"
      << "  --reset-breakdown-every <n>     Clear samples every n attempted inserts, default 50000; 0 disables\n"
      << "  --report-text <path>            Optional text summary\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        fail(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (flag == "--service-config") {
      args.service_config_path = require_value("--service-config");
    } else if (flag == "--insert-file") {
      args.insert_file = require_value("--insert-file");
    } else if (flag == "--insert-start-id") {
      args.insert_start_id = std::stoull(require_value("--insert-start-id"));
    } else if (flag == "--insert-count") {
      args.insert_count = std::stoull(require_value("--insert-count"));
    } else if (flag == "--insert-row-offset") {
      args.insert_row_offset = std::stoull(require_value("--insert-row-offset"));
    } else if (flag == "--insert-threads") {
      args.insert_threads = std::stoull(require_value("--insert-threads"));
    } else if (flag == "--insert-batch-size") {
      args.insert_batch_size = std::stoull(require_value("--insert-batch-size"));
    } else if (flag == "--query-file") {
      args.query_file = require_value("--query-file");
    } else if (flag == "--baseline-groundtruth-file") {
      args.baseline_groundtruth_file = require_value("--baseline-groundtruth-file");
    } else if (flag == "--groundtruth-file") {
      args.groundtruth_file = require_value("--groundtruth-file");
    } else if (flag == "--recall-queries") {
      args.recall_queries = std::stoull(require_value("--recall-queries"));
    } else if (flag == "--recall-k") {
      args.recall_k = static_cast<u32>(std::stoul(require_value("--recall-k")));
    } else if (flag == "--settle-seconds") {
      args.settle_seconds = std::stoull(require_value("--settle-seconds"));
    } else if (flag == "--reset-breakdown-every") {
      args.reset_breakdown_every = std::stoull(require_value("--reset-breakdown-every"));
    } else if (flag == "--report-json") {
      args.report_json_path = require_value("--report-json");
    } else if (flag == "--report-text") {
      args.report_text_path = require_value("--report-text");
    } else if (flag == "--help" || flag == "-h") {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      fail("unknown argument: " + flag);
    }
  }

  if (args.service_config_path.empty()) fail("--service-config is required");
  if (args.insert_file.empty()) fail("--insert-file is required");
  if (args.query_file.empty()) fail("--query-file is required");
  if (args.groundtruth_file.empty()) fail("--groundtruth-file is required");
  if (args.report_json_path.empty()) fail("--report-json is required");
  if (args.insert_threads == 0) fail("--insert-threads must be > 0");
  if (args.insert_batch_size == 0) fail("--insert-batch-size must be > 0");
  if (args.recall_k == 0) fail("--recall-k must be > 0");
  if (args.insert_start_id > std::numeric_limits<node_t>::max()) {
    fail("--insert-start-id exceeds node_t range");
  }
  return args;
}

RecallResult run_recall(ComputeService& service,
                        const std::string& label,
                        const VectorRows& queries,
                        const GroundTruth& gt,
                        size_t requested_queries,
                        u32 requested_k) {
  if (queries.dim != service.config().dim) {
    fail("query dim mismatch: query=" + std::to_string(queries.dim) +
         " config=" + std::to_string(service.config().dim));
  }
  if (gt.rows != queries.count) {
    fail("query/groundtruth row count mismatch: query=" + std::to_string(queries.count) +
         " gt=" + std::to_string(gt.rows));
  }
  if (requested_k > gt.top_k) {
    fail("recall k exceeds groundtruth top_k");
  }

  const size_t recall_queries = requested_queries == 0
                                  ? queries.count
                                  : std::min<size_t>(requested_queries, queries.count);
  std::atomic<size_t> completed{0};
  ProgressReporter reporter(label, completed, recall_queries, 0);
  double total_recall = 0.0;
  for (size_t qi = 0; qi < recall_queries; ++qi) {
    const auto results = service.search_raw(queries.dtype, queries.row_payload(qi),
                                            queries.dim, requested_k);
    std::vector<u32> result_ids;
    result_ids.reserve(results.size());
    for (const node_t id : results) {
      result_ids.push_back(static_cast<u32>(id));
    }
    total_recall += recall_at_k(result_ids, gt.row(qi), requested_k);
    completed.fetch_add(1, std::memory_order_relaxed);
  }
  reporter.finish();

  RecallResult result;
  result.queries = recall_queries;
  result.k = requested_k;
  result.recall = recall_queries == 0 ? 0.0 : total_recall / static_cast<double>(recall_queries);
  std::cerr << "[sift101m-long-insert][recall] " << label << " recall@" << requested_k
            << "=" << result.recall << " queries=" << result.queries << std::endl;
  return result;
}

InsertStats run_insert_phase(ComputeService& service,
                             const VectorRows& insert_rows,
                             const Args& args,
                             size_t effective_insert_count) {
  if (insert_rows.dim != service.config().dim) {
    fail("insert dim mismatch: insert=" + std::to_string(insert_rows.dim) +
         " config=" + std::to_string(service.config().dim));
  }
  if (args.insert_row_offset + effective_insert_count > insert_rows.count) {
    fail("insert range exceeds insert-file row count");
  }
  if (args.insert_start_id + effective_insert_count > static_cast<uint64_t>(std::numeric_limits<node_t>::max()) + 1ull) {
    fail("insert id range exceeds node_t range");
  }

  std::atomic<size_t> next_row{0};
  std::atomic<size_t> attempted{0};
  std::atomic<size_t> inserted{0};
  std::atomic<size_t> failed{0};
  std::atomic<bool> stop{false};
  std::exception_ptr first_error;
  std::mutex error_mutex;
  std::mutex reset_mutex;
  std::barrier start_barrier(static_cast<std::ptrdiff_t>(args.insert_threads));
  std::vector<std::thread> threads;
  threads.reserve(args.insert_threads);

  ProgressReporter reporter("insert-100m-to-101m", attempted, effective_insert_count, 0);
  const auto start = std::chrono::steady_clock::now();

  for (size_t tid = 0; tid < args.insert_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      (void)tid;
      try {
        start_barrier.arrive_and_wait();
        vec<ComputeService::InsertItem> batch;
        batch.reserve(args.insert_batch_size);
        while (!stop.load(std::memory_order_acquire)) {
          const size_t begin = next_row.fetch_add(args.insert_batch_size, std::memory_order_relaxed);
          if (begin >= effective_insert_count) {
            break;
          }
          const size_t end = std::min(begin + args.insert_batch_size, effective_insert_count);
          batch.clear();
          for (size_t row = begin; row < end; ++row) {
            const uint64_t id64 = args.insert_start_id + row;
            batch.push_back({
              static_cast<node_t>(id64),
              insert_rows.decode_row(args.insert_row_offset + row),
            });
          }

          const size_t ok = service.insert(batch);
          inserted.fetch_add(ok, std::memory_order_relaxed);
          if (ok != batch.size()) {
            failed.fetch_add(batch.size() - ok, std::memory_order_relaxed);
          }
          const size_t before = attempted.fetch_add(batch.size(), std::memory_order_relaxed);
          const size_t after = before + batch.size();
          if (args.reset_breakdown_every > 0 &&
              before / args.reset_breakdown_every != after / args.reset_breakdown_every) {
            std::lock_guard<std::mutex> lock(reset_mutex);
            service.clear_thread_statistics();
            service.reset_breakdown_state();
          }
        }
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!first_error) {
            first_error = std::current_exception();
          }
        }
        stop.store(true, std::memory_order_release);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
  reporter.finish();
  if (first_error) {
    std::rethrow_exception(first_error);
  }

  const auto stop_time = std::chrono::steady_clock::now();
  InsertStats stats;
  stats.attempted = attempted.load(std::memory_order_relaxed);
  stats.inserted = inserted.load(std::memory_order_relaxed);
  stats.failed = failed.load(std::memory_order_relaxed);
  stats.duration_seconds = std::chrono::duration<double>(stop_time - start).count();
  stats.inserts_per_second = stats.duration_seconds <= 0.0
                               ? 0.0
                               : static_cast<double>(stats.inserted) / stats.duration_seconds;
  return stats;
}

void wait_for_settle(size_t seconds) {
  if (seconds == 0) {
    return;
  }
  std::atomic<size_t> elapsed{0};
  ProgressReporter reporter("settle-after-insert", elapsed, 0, seconds);
  for (size_t i = 0; i < seconds; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    elapsed.fetch_add(1, std::memory_order_relaxed);
  }
  reporter.finish();
}

nlohmann::json recall_json(const RecallResult& result,
                           const std::filesystem::path& query_file,
                           const std::filesystem::path& gt_file) {
  return {
    {"query_file", query_file.string()},
    {"groundtruth_file", gt_file.string()},
    {"queries", result.queries},
    {"k", result.k},
    {"recall", result.recall},
  };
}

nlohmann::json insert_stats_json(const InsertStats& stats) {
  return {
    {"attempted", stats.attempted},
    {"inserted", stats.inserted},
    {"failed", stats.failed},
    {"duration_seconds", stats.duration_seconds},
    {"inserts_per_second", stats.inserts_per_second},
  };
}

int run_with_service(ComputeService& service, const Args& args) {
  std::cerr << "[sift101m-long-insert] loading insert vectors: " << args.insert_file << std::endl;
  const VectorRows insert_rows = read_vector_rows(args.insert_file);
  std::cerr << "[sift101m-long-insert] insert rows=" << insert_rows.count
            << " dim=" << insert_rows.dim
            << " dtype=" << vector_dtype_name(insert_rows.dtype)
            << " vector_bytes=" << insert_rows.vector_bytes << std::endl;

  std::cerr << "[sift101m-long-insert] loading queries: " << args.query_file << std::endl;
  const VectorRows queries = read_vector_rows(args.query_file);
  std::cerr << "[sift101m-long-insert] query rows=" << queries.count
            << " dim=" << queries.dim
            << " dtype=" << vector_dtype_name(queries.dtype) << std::endl;

  if (args.insert_row_offset > insert_rows.count) {
    fail("--insert-row-offset exceeds insert-file row count");
  }
  const size_t available_insert_rows = insert_rows.count - args.insert_row_offset;
  const size_t effective_insert_count = args.insert_count == 0
                                          ? available_insert_rows
                                          : std::min(args.insert_count, available_insert_rows);
  if (effective_insert_count == 0) {
    fail("effective insert count is zero");
  }

  nlohmann::json root;
  root["meta"] = {
    {"service_config", args.service_config_path},
    {"insert_file", args.insert_file.string()},
    {"insert_start_id", args.insert_start_id},
    {"insert_count_requested", args.insert_count},
    {"insert_count_effective", effective_insert_count},
    {"insert_row_offset", args.insert_row_offset},
    {"insert_threads", args.insert_threads},
    {"insert_batch_size", args.insert_batch_size},
    {"query_file", args.query_file.string()},
    {"baseline_groundtruth_file", args.baseline_groundtruth_file.empty() ? "" : args.baseline_groundtruth_file.string()},
    {"post_groundtruth_file", args.groundtruth_file.string()},
    {"recall_queries", args.recall_queries},
    {"recall_k", args.recall_k},
    {"settle_seconds", args.settle_seconds},
    {"reset_breakdown_every", args.reset_breakdown_every},
    {"dim", service.config().dim},
    {"max_vectors_config", service.config().max_vectors},
    {"storage_owner_update_protocol", "centroid_home_two_stage"},
    {"storage_owner_maintenance_workers", service.config().storage_owner_maintenance_workers},
    {"fine_grained_breakdown_enabled", service.config().enable_breakdown},
    {"search", "gpu_persistent_opq_pq"},
    {"navigation_quantizer", "opq_pq"},
  };
  root["input"] = {
    {"insert_rows", insert_rows.count},
    {"insert_dim", insert_rows.dim},
    {"insert_dtype", vector_dtype_name(insert_rows.dtype)},
    {"query_rows", queries.count},
    {"query_dim", queries.dim},
    {"query_dtype", vector_dtype_name(queries.dtype)},
  };

  std::optional<RecallResult> baseline_recall;
  if (!args.baseline_groundtruth_file.empty()) {
    std::cerr << "[sift101m-long-insert] loading baseline groundtruth: "
              << args.baseline_groundtruth_file << std::endl;
    const GroundTruth baseline_gt = read_groundtruth(args.baseline_groundtruth_file);
    baseline_recall = run_recall(service, "baseline-100m", queries, baseline_gt,
                                 args.recall_queries, args.recall_k);
    root["baseline_recall"] = recall_json(*baseline_recall, args.query_file,
                                          args.baseline_groundtruth_file);
    service.clear_thread_statistics();
    service.reset_breakdown_state();
  }

  std::cerr << "[sift101m-long-insert] inserting ID range [" << args.insert_start_id
            << ", " << (args.insert_start_id + effective_insert_count) << ")" << std::endl;
  const InsertStats insert_stats = run_insert_phase(service, insert_rows, args, effective_insert_count);
  root["insert"] = insert_stats_json(insert_stats);
  service.clear_thread_statistics();
  service.reset_breakdown_state();

  if (insert_stats.failed != 0 || insert_stats.inserted != effective_insert_count) {
    root["execution_error"] = "one or more inserts failed";
    std::ostringstream text;
    text << "insert failed\n"
         << "  attempted: " << insert_stats.attempted << '\n'
         << "  inserted: " << insert_stats.inserted << '\n'
         << "  failed: " << insert_stats.failed << '\n';
    write_json_file(args.report_json_path, root);
    write_text_file(args.report_text_path, text.str());
    std::cout << text.str();
    return EXIT_FAILURE;
  }

  wait_for_settle(args.settle_seconds);

  std::cerr << "[sift101m-long-insert] loading post-insert groundtruth: "
            << args.groundtruth_file << std::endl;
  const GroundTruth post_gt = read_groundtruth(args.groundtruth_file);
  const RecallResult post_recall = run_recall(service, "post-101m", queries, post_gt,
                                              args.recall_queries, args.recall_k);
  root["post_insert_recall"] = recall_json(post_recall, args.query_file, args.groundtruth_file);

  if (baseline_recall.has_value()) {
    const double drop = baseline_recall->recall - post_recall.recall;
    root["recall_delta"] = {
      {"baseline_minus_post", drop},
      {"post_minus_baseline", post_recall.recall - baseline_recall->recall},
    };
  }

  std::ostringstream text;
  text << std::fixed << std::setprecision(6);
  text << "sift101m long insert recall\n";
  text << "  inserted: " << insert_stats.inserted << "/" << effective_insert_count
       << " in " << insert_stats.duration_seconds << "s"
       << " (" << insert_stats.inserts_per_second << " inserts/s)\n";
  if (baseline_recall.has_value()) {
    text << "  baseline recall@" << baseline_recall->k << ": "
         << baseline_recall->recall << " (" << baseline_recall->queries << " queries)\n";
  }
  text << "  post recall@" << post_recall.k << ": "
       << post_recall.recall << " (" << post_recall.queries << " queries)\n";
  if (baseline_recall.has_value()) {
    text << "  recall drop: " << (baseline_recall->recall - post_recall.recall) << '\n';
  }

  write_json_file(args.report_json_path, root);
  write_text_file(args.report_text_path, text.str());
  std::cout << text.str();
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGSEGV, signal_handler);
  signal(SIGABRT, signal_handler);

  try {
    const Args args = parse_args(argc, argv);
    auto service_args = tools::breakdown_benchmark::build_service_argv(args.service_config_path);
    auto service_argv = tools::breakdown_benchmark::make_argv(service_args);
    configuration::IndexConfiguration config(static_cast<int>(service_argv.size()), service_argv.data());

    ComputeService service(config);
    return run_with_service(service, args);
  } catch (const std::exception& e) {
    std::cerr << "sift101m long insert recall failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
