#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

struct Args {
  std::filesystem::path sift_root{"/data/xjs/datasets/sift1b"};
  std::filesystem::path base_path{};
  std::filesystem::path query_bvecs{};
  std::filesystem::path gt100_idx{};
  std::filesystem::path gt100_dist{};
  std::filesystem::path out_query{"sift101m_query.fbin"};
  std::filesystem::path out_groundtruth{"sift101m_groundtruth.bin"};
  u32 topk{100};
  u64 insert_start{100000000};
  u64 insert_count{1000000};
  u32 threads{0};
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

u64 path_file_size(const std::filesystem::path& path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) fail("cannot stat " + path.string() + ": " + ec.message());
  return size;
}

template <class T>
void read_exact(std::ifstream& input, T* dst, size_t count, const std::string& what) {
  const size_t bytes = count * sizeof(T);
  if (!input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes))) {
    fail("failed to read " + what);
  }
}

u32 read_u32(std::ifstream& input, const std::string& what) {
  u32 value{};
  read_exact(input, &value, 1, what);
  return value;
}

void write_exact(std::ofstream& output, const void* data, size_t bytes, const std::string& what) {
  if (!output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes))) {
    fail("failed to write " + what);
  }
}

std::string format_duration(double seconds) {
  if (seconds < 0.0 || seconds > 365.0 * 24.0 * 3600.0) return "--";
  u64 total = static_cast<u64>(seconds + 0.5);
  const u64 hours = total / 3600;
  total %= 3600;
  const u64 minutes = total / 60;
  const u64 secs = total % 60;

  std::ostringstream out;
  if (hours > 0) out << hours << "h";
  if (hours > 0 || minutes > 0) out << minutes << "m";
  out << secs << "s";
  return out.str();
}

void print_progress(const std::string& label, u64 current, u64 total,
                    std::chrono::steady_clock::time_point start) {
  const auto now = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(now - start).count();
  const double fraction = total == 0 ? 1.0 : static_cast<double>(current) / static_cast<double>(total);
  const double rate = elapsed > 0.0 ? static_cast<double>(current) / elapsed : 0.0;
  const double eta = rate > 0.0 ? static_cast<double>(total - current) / rate : -1.0;
  constexpr u32 bar_width = 30;
  const u32 filled = static_cast<u32>(std::min<double>(bar_width, fraction * bar_width));

  std::cerr << "\r" << label << " [";
  for (u32 i = 0; i < bar_width; ++i) std::cerr << (i < filled ? '#' : '.');
  std::cerr << "] " << std::fixed << std::setprecision(1) << (fraction * 100.0)
            << "% " << current << "/" << total
            << " rate=" << std::setprecision(2) << rate << "/s"
            << " elapsed=" << format_duration(elapsed)
            << " eta=" << format_duration(eta) << "   " << std::flush;
  if (current >= total) std::cerr << "\n";
}

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --sift-root <dir>          SIFT1B root (default: /data/xjs/datasets/sift1b)\n"
      << "  --base <path>              bigann_base.bvecs path\n"
      << "  --query-bvecs <path>       bigann_query.bvecs path\n"
      << "  --gt100-idx <path>         gnd/idx_100M.ivecs path\n"
      << "  --gt100-dist <path>        gnd/dis_100M.fvecs path\n"
      << "  --out-query <path>         output query .fbin (u32 n, u32 dim, float payload)\n"
      << "  --out-groundtruth <path>   output groundtruth .bin (u32 n, u32 k, u32 ids)\n"
      << "  --topk <k>                 output top-k (default: 100)\n"
      << "  --insert-start <id>        first inserted base id (default: 100000000)\n"
      << "  --insert-count <n>         inserted vector count (default: 1000000)\n"
      << "  --threads <n>              worker threads (default: hardware_concurrency)\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const char* option) -> const char* {
      if (i + 1 >= argc) fail(std::string("missing value for ") + option);
      return argv[++i];
    };
    if (key == "--sift-root") {
      args.sift_root = need_value("--sift-root");
    } else if (key == "--base") {
      args.base_path = need_value("--base");
    } else if (key == "--query-bvecs") {
      args.query_bvecs = need_value("--query-bvecs");
    } else if (key == "--gt100-idx") {
      args.gt100_idx = need_value("--gt100-idx");
    } else if (key == "--gt100-dist") {
      args.gt100_dist = need_value("--gt100-dist");
    } else if (key == "--out-query") {
      args.out_query = need_value("--out-query");
    } else if (key == "--out-groundtruth") {
      args.out_groundtruth = need_value("--out-groundtruth");
    } else if (key == "--topk") {
      args.topk = static_cast<u32>(std::stoul(need_value("--topk")));
    } else if (key == "--insert-start") {
      args.insert_start = std::stoull(need_value("--insert-start"));
    } else if (key == "--insert-count") {
      args.insert_count = std::stoull(need_value("--insert-count"));
    } else if (key == "--threads") {
      args.threads = static_cast<u32>(std::stoul(need_value("--threads")));
    } else if (key == "--help" || key == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      fail("unknown option: " + key);
    }
  }

  if (args.base_path.empty()) args.base_path = args.sift_root / "bigann_base.bvecs";
  if (args.query_bvecs.empty()) args.query_bvecs = args.sift_root / "bigann_query.bvecs";
  if (args.gt100_idx.empty()) args.gt100_idx = args.sift_root / "gnd" / "idx_100M.ivecs";
  if (args.gt100_dist.empty()) args.gt100_dist = args.sift_root / "gnd" / "dis_100M.fvecs";
  if (args.topk == 0) fail("--topk must be positive");
  return args;
}

struct QuerySet {
  u32 count{};
  u32 dim{};
  std::vector<u8> raw_u8;
  std::vector<float> floats;
};

QuerySet read_queries_and_write_fbin(const std::filesystem::path& input_path,
                                     const std::filesystem::path& output_path) {
  std::ifstream input(input_path, std::ios::binary);
  if (!input) fail("cannot open query bvecs: " + input_path.string());
  const u64 size = path_file_size(input_path);
  const u32 dim = read_u32(input, "query dim");
  const u64 row_bytes = sizeof(u32) + dim;
  if (dim == 0 || size % row_bytes != 0) {
    fail("query bvecs size is not divisible by row size");
  }
  const u32 count = static_cast<u32>(size / row_bytes);
  input.seekg(0, std::ios::beg);

  QuerySet queries;
  queries.count = count;
  queries.dim = dim;
  queries.raw_u8.resize(static_cast<size_t>(count) * dim);
  queries.floats.resize(static_cast<size_t>(count) * dim);

  for (u32 row = 0; row < count; ++row) {
    const u32 row_dim = read_u32(input, "query row dim");
    if (row_dim != dim) fail("query dim mismatch in row " + std::to_string(row));
    u8* raw = queries.raw_u8.data() + static_cast<size_t>(row) * dim;
    read_exact(input, raw, dim, "query row");
    float* dst = queries.floats.data() + static_cast<size_t>(row) * dim;
    for (u32 j = 0; j < dim; ++j) dst[j] = static_cast<float>(raw[j]);
  }

  std::filesystem::create_directories(output_path.parent_path().empty() ? "." : output_path.parent_path());
  std::ofstream output(output_path, std::ios::binary);
  if (!output) fail("cannot open output query: " + output_path.string());
  write_exact(output, &count, sizeof(count), "query count");
  write_exact(output, &dim, sizeof(dim), "query dim");
  write_exact(output, queries.floats.data(), queries.floats.size() * sizeof(float), "query payload");
  std::cerr << "wrote query fbin: " << output_path << " (" << count << " x " << dim << ")\n";
  return queries;
}

std::vector<u8> load_insert_vectors(const std::filesystem::path& base_path,
                                    u64 insert_start,
                                    u64 insert_count,
                                    u32 expected_dim) {
  std::ifstream input(base_path, std::ios::binary);
  if (!input) fail("cannot open base bvecs: " + base_path.string());
  const u64 size = path_file_size(base_path);
  const u32 dim = read_u32(input, "base dim");
  if (dim != expected_dim) fail("base/query dim mismatch");
  const u64 row_bytes = sizeof(u32) + dim;
  if (size % row_bytes != 0) fail("base bvecs size is not divisible by row size");
  const u64 total = size / row_bytes;
  if (insert_start + insert_count > total) fail("insert range exceeds base vector count");

  std::vector<u8> vectors(static_cast<size_t>(insert_count) * dim);
  input.seekg(static_cast<std::streamoff>(insert_start * row_bytes), std::ios::beg);
  const auto progress_start = std::chrono::steady_clock::now();
  const u64 progress_step = std::max<u64>(1, insert_count / 100);
  for (u64 row = 0; row < insert_count; ++row) {
    const u32 row_dim = read_u32(input, "insert row dim");
    if (row_dim != dim) fail("base dim mismatch at row " + std::to_string(insert_start + row));
    read_exact(input, vectors.data() + static_cast<size_t>(row) * dim, dim, "insert row");
    if ((row + 1) % progress_step == 0 || row + 1 == insert_count) {
      print_progress("loading inserts", row + 1, insert_count, progress_start);
    }
  }
  return vectors;
}

struct VecsHeader {
  u32 rows{};
  u32 dim{};
};

VecsHeader inspect_vecs(const std::filesystem::path& path, size_t element_bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open vecs file: " + path.string());
  const u64 size = path_file_size(path);
  const u32 dim = read_u32(input, "vecs dim");
  const u64 row_bytes = sizeof(u32) + static_cast<u64>(dim) * element_bytes;
  if (dim == 0 || size % row_bytes != 0) fail("invalid vecs file size: " + path.string());
  return VecsHeader{static_cast<u32>(size / row_bytes), dim};
}

void read_gt100_row(std::ifstream& idx_input,
                    std::ifstream& dist_input,
                    u32 row,
                    u32 source_topk,
                    u32 out_topk,
                    std::vector<std::pair<float, u32>>& candidates) {
  const u64 idx_row_bytes = sizeof(u32) + static_cast<u64>(source_topk) * sizeof(u32);
  const u64 dist_row_bytes = sizeof(u32) + static_cast<u64>(source_topk) * sizeof(float);
  idx_input.seekg(static_cast<std::streamoff>(static_cast<u64>(row) * idx_row_bytes), std::ios::beg);
  dist_input.seekg(static_cast<std::streamoff>(static_cast<u64>(row) * dist_row_bytes), std::ios::beg);
  const u32 idx_dim = read_u32(idx_input, "idx row dim");
  const u32 dist_dim = read_u32(dist_input, "dist row dim");
  if (idx_dim != source_topk || dist_dim != source_topk) fail("groundtruth row dim mismatch");

  std::vector<u32> ids(source_topk);
  std::vector<float> distances(source_topk);
  read_exact(idx_input, ids.data(), source_topk, "idx row");
  read_exact(dist_input, distances.data(), source_topk, "dist row");

  candidates.clear();
  candidates.reserve(out_topk);
  for (u32 i = 0; i < out_topk; ++i) {
    candidates.emplace_back(distances[i], ids[i]);
  }
}

u32 l2_u8_bounded(const u8* query, const u8* base, u32 dim, u32 limit) {
  u32 sum = 0;
  for (u32 i = 0; i < dim; ++i) {
    const int diff = static_cast<int>(query[i]) - static_cast<int>(base[i]);
    sum += static_cast<u32>(diff * diff);
    if (sum >= limit) return sum;
  }
  return sum;
}

std::vector<u32> build_groundtruth101(const Args& args,
                                      const QuerySet& queries,
                                      const std::vector<u8>& inserts) {
  const VecsHeader idx_header = inspect_vecs(args.gt100_idx, sizeof(u32));
  const VecsHeader dist_header = inspect_vecs(args.gt100_dist, sizeof(float));
  if (idx_header.rows != queries.count || dist_header.rows != queries.count) {
    fail("groundtruth/query count mismatch");
  }
  if (idx_header.dim != dist_header.dim) fail("idx/dist topk mismatch");
  if (args.topk > idx_header.dim) {
    fail("--topk exceeds source GT topk " + std::to_string(idx_header.dim));
  }

  std::vector<u32> output_ids(static_cast<size_t>(queries.count) * args.topk);
  const u32 worker_count =
      args.threads == 0 ? std::max(1u, std::thread::hardware_concurrency()) : args.threads;
  std::atomic<u32> next_query{0};
  std::atomic<u32> done{0};
  std::atomic<bool> failed{false};
  std::string first_error;
  std::mutex error_mutex;
  std::mutex progress_mutex;
  const auto start = std::chrono::steady_clock::now();
  std::cerr << "groundtruth scan work: " << queries.count << " queries x "
            << args.insert_count << " inserted vectors = "
            << (static_cast<u64>(queries.count) * args.insert_count)
            << " distances\n";
  print_progress("building GT", 0, queries.count, start);

  auto worker = [&]() {
    try {
      std::ifstream idx_input(args.gt100_idx, std::ios::binary);
      std::ifstream dist_input(args.gt100_dist, std::ios::binary);
      if (!idx_input || !dist_input) fail("cannot open GT files in worker");

      std::vector<std::pair<float, u32>> base_candidates;
      using Candidate = std::pair<float, u32>;
      std::priority_queue<Candidate> heap;
      while (!failed.load(std::memory_order_relaxed)) {
        const u32 q = next_query.fetch_add(1);
        if (q >= queries.count) break;

        read_gt100_row(idx_input, dist_input, q, idx_header.dim, args.topk, base_candidates);
        heap = std::priority_queue<Candidate>{};
        for (const auto& candidate : base_candidates) heap.push(candidate);

        const u8* query = queries.raw_u8.data() + static_cast<size_t>(q) * queries.dim;
        for (u64 offset = 0; offset < args.insert_count; ++offset) {
          const auto worst = heap.top();
          const u32 limit = worst.first >= static_cast<float>(std::numeric_limits<u32>::max())
                                ? std::numeric_limits<u32>::max()
                                : static_cast<u32>(worst.first);
          const u8* base = inserts.data() + static_cast<size_t>(offset) * queries.dim;
          const u32 dist = l2_u8_bounded(query, base, queries.dim, limit);
          const Candidate candidate{static_cast<float>(dist), static_cast<u32>(args.insert_start + offset)};
          if (candidate < worst) {
            heap.pop();
            heap.push(candidate);
          }
        }

        std::vector<Candidate> row;
        row.reserve(args.topk);
        while (!heap.empty()) {
          row.push_back(heap.top());
          heap.pop();
        }
        std::sort(row.begin(), row.end());
        u32* dst = output_ids.data() + static_cast<size_t>(q) * args.topk;
        for (u32 i = 0; i < args.topk; ++i) dst[i] = row[i].second;

        const u32 finished = done.fetch_add(1) + 1;
        if (finished % 100 == 0 || finished == queries.count) {
          std::lock_guard<std::mutex> lock(progress_mutex);
          print_progress("building GT", finished, queries.count, start);
        }
      }
    } catch (const std::exception& e) {
      failed.store(true);
      std::lock_guard<std::mutex> lock(error_mutex);
      if (first_error.empty()) first_error = e.what();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(worker_count);
  for (u32 i = 0; i < worker_count; ++i) threads.emplace_back(worker);
  for (auto& thread : threads) thread.join();
  std::cerr << "\n";
  if (failed.load()) fail(first_error.empty() ? "worker failed" : first_error);
  return output_ids;
}

void write_groundtruth_bin(const std::filesystem::path& path,
                           u32 query_count,
                           u32 topk,
                           const std::vector<u32>& ids) {
  std::filesystem::create_directories(path.parent_path().empty() ? "." : path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) fail("cannot open output groundtruth: " + path.string());
  write_exact(output, &query_count, sizeof(query_count), "groundtruth query count");
  write_exact(output, &topk, sizeof(topk), "groundtruth topk");
  write_exact(output, ids.data(), ids.size() * sizeof(u32), "groundtruth ids");
  std::cerr << "wrote groundtruth: " << path << " (" << query_count << " x " << topk << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    std::cerr << "base: " << args.base_path << "\n"
              << "queries: " << args.query_bvecs << "\n"
              << "gt100 ids: " << args.gt100_idx << "\n"
              << "gt100 distances: " << args.gt100_dist << "\n"
              << "insert range: [" << args.insert_start << ", "
              << (args.insert_start + args.insert_count) << ")\n"
              << "topk: " << args.topk << "\n";

    QuerySet queries = read_queries_and_write_fbin(args.query_bvecs, args.out_query);
    std::vector<u8> inserts =
        load_insert_vectors(args.base_path, args.insert_start, args.insert_count, queries.dim);
    std::vector<u32> ids = build_groundtruth101(args, queries, inserts);
    write_groundtruth_bin(args.out_groundtruth, queries.count, args.topk, ids);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
