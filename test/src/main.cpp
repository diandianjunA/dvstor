#include "shine_index.h"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// Accumulate all inserted (id, vector) pairs for recall ground truth
std::vector<uint32_t> g_all_ids;
std::vector<float> g_all_vecs;

struct ConcurrentThreadResult {
  size_t inserted{};
  double elapsed_ms{};
  std::string error;
};

std::vector<float> make_deterministic_dataset(const std::vector<uint32_t>& ids, size_t dim);

std::vector<float> make_sparse_dataset(size_t num_vectors, size_t dim) {
  std::vector<uint32_t> local_ids(num_vectors);
  for (size_t i = 0; i < num_vectors; ++i) {
    local_ids[i] = static_cast<uint32_t>(i + 1);
  }
  return make_deterministic_dataset(local_ids, dim);
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

std::vector<float> make_deterministic_dataset(const std::vector<uint32_t>& ids, size_t dim) {
  std::vector<float> vectors;
  vectors.reserve(ids.size() * dim);
  for (uint32_t id : ids) {
    auto vector = make_deterministic_vector(id, dim);
    vectors.insert(vectors.end(), vector.begin(), vector.end());
  }
  return vectors;
}

void require_contains(const std::vector<uint32_t>& ids, uint32_t expected, const std::string& stage) {
  if (std::find(ids.begin(), ids.end(), expected) == ids.end()) {
    std::string joined = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i > 0) {
        joined += ", ";
      }
      joined += std::to_string(ids[i]);
    }
    joined += "]";
    throw std::runtime_error(stage + " failed: expected id " + std::to_string(expected) + " not found in " + joined);
  }
}

void run_smoke_test(ShineIndex& index, const std::string& index_prefix, uint32_t base_id, bool test_reload) {
  const size_t dim = index.dimension();
  const size_t num_vectors = std::min<size_t>(32, std::max<size_t>(8, dim));
  const size_t smoke_top_k = num_vectors;

  std::vector<uint32_t> ids(num_vectors);
  for (size_t i = 0; i < num_vectors; ++i) {
    ids[i] = static_cast<uint32_t>(base_id + i);
  }

  const auto vectors = make_sparse_dataset(num_vectors, dim);
  const size_t inserted = index.insert_count(vectors, ids);
  std::cout << "[smoke] inserted=" << inserted << " expected=" << num_vectors << std::endl;
  if (inserted != num_vectors) {
    throw std::runtime_error("smoke insert count mismatch: inserted=" + std::to_string(inserted) +
                             " expected=" + std::to_string(num_vectors));
  }

  g_all_ids.insert(g_all_ids.end(), ids.begin(), ids.end());
  g_all_vecs.insert(g_all_vecs.end(), vectors.begin(), vectors.end());

  std::vector<uint32_t> search_ids;
  std::vector<float> distances;
  std::vector<float> query(vectors.begin(), vectors.begin() + static_cast<std::ptrdiff_t>(dim));

  index.search(query, smoke_top_k, search_ids, distances);
  std::cout << "[smoke] initial search ids=";
  for (uint32_t id : search_ids) {
    std::cout << id << ' ';
  }
  std::cout << std::endl;
  require_contains(search_ids, ids.front(), "initial search");

  if (test_reload) {
    index.save(index_prefix);
    index.load(index_prefix);

    search_ids.clear();
    distances.clear();
    index.search(query, smoke_top_k, search_ids, distances);
    std::cout << "[smoke] reload search ids=";
    for (uint32_t id : search_ids) {
      std::cout << id << ' ';
    }
    std::cout << std::endl;
    require_contains(search_ids, ids.front(), "reload search");
  }

  std::cout << "[smoke] passed" << std::endl;
  std::cout << "  vectors: " << num_vectors << std::endl;
  std::cout << "  prefix:  " << index_prefix << std::endl;
}

void run_concurrent_insert_test(ShineIndex& index,
                                size_t thread_count,
                                size_t vectors_per_thread,
                                uint32_t base_id) {
  const size_t dim = index.dimension();

  if (thread_count == 1) {
    std::vector<uint32_t> ids(vectors_per_thread);
    for (size_t i = 0; i < vectors_per_thread; ++i) {
      ids[i] = base_id + static_cast<uint32_t>(i);
    }
    const auto vectors = make_deterministic_dataset(ids, dim);

    for (size_t i = 0; i < vectors_per_thread; ++i) {
      const auto begin = std::chrono::steady_clock::now();
      const std::vector<uint32_t> single_id = {ids[i]};
      const std::vector<float> single_vector(vectors.begin() + static_cast<std::ptrdiff_t>(i * dim),
                                             vectors.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim));
      const size_t inserted = index.insert_count(single_vector, single_id);
      const auto end = std::chrono::steady_clock::now();
      std::cout << "[serial] insert " << i << " id=" << ids[i]
                << " inserted=" << inserted
                << " elapsed_ms="
                << std::chrono::duration<double, std::milli>(end - begin).count()
                << std::endl;
      if (inserted != 1) {
        throw std::runtime_error("serial insert failed for id " + std::to_string(ids[i]));
      }
    }

    std::vector<uint32_t> search_ids;
    std::vector<float> distances;
    for (size_t i = 0; i < vectors_per_thread; ++i) {
      const std::vector<float> query(vectors.begin() + static_cast<std::ptrdiff_t>(i * dim),
                                     vectors.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim));
      search_ids.clear();
      distances.clear();
      index.search(query, 10, search_ids, distances);
      if (std::find(search_ids.begin(), search_ids.end(), ids[i]) == search_ids.end()) {
        std::cerr << "[serial] missing self id " << ids[i] << " results=";
        for (uint32_t result_id : search_ids) {
          std::cerr << result_id << ' ';
        }
        std::cerr << std::endl;
        throw std::runtime_error("serial insert verification failed for id " + std::to_string(ids[i]));
      }
    }

    std::cout << "[serial] aggregate inserted=" << vectors_per_thread << std::endl;
    return;
  }

  std::barrier start_barrier(static_cast<std::ptrdiff_t>(thread_count));

  std::vector<std::vector<uint32_t>> thread_ids(thread_count);
  std::vector<std::vector<float>> thread_vectors(thread_count);
  std::vector<ConcurrentThreadResult> thread_results(thread_count);
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (size_t tid = 0; tid < thread_count; ++tid) {
    auto& ids = thread_ids[tid];
    ids.resize(vectors_per_thread);
    for (size_t i = 0; i < vectors_per_thread; ++i) {
      ids[i] = base_id + static_cast<uint32_t>(tid * vectors_per_thread + i);
    }
    thread_vectors[tid] = make_deterministic_dataset(ids, dim);
  }

  for (size_t tid = 0; tid < thread_count; ++tid) {
    threads.emplace_back([&, tid]() {
      auto begin = std::chrono::steady_clock::now();
      start_barrier.arrive_and_wait();

      try {
        thread_results[tid].inserted = index.insert_count(thread_vectors[tid], thread_ids[tid]);
      } catch (const std::exception& e) {
        thread_results[tid].error = e.what();
      }

      auto end = std::chrono::steady_clock::now();
      thread_results[tid].elapsed_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  size_t inserted_total = 0;
  for (size_t tid = 0; tid < thread_count; ++tid) {
    const auto& result = thread_results[tid];
    inserted_total += result.inserted;
    std::cout << "[concurrent] thread " << tid
              << " inserted=" << result.inserted
              << " elapsed_ms=" << result.elapsed_ms;
    if (!result.error.empty()) {
      std::cout << " error=\"" << result.error << '"';
    }
    std::cout << std::endl;
  }

  const size_t expected_total = thread_count * vectors_per_thread;
  if (inserted_total != expected_total) {
    throw std::runtime_error("concurrent insert count mismatch: inserted=" + std::to_string(inserted_total) +
                             " expected=" + std::to_string(expected_total));
  }

  for (size_t tid = 0; tid < thread_count; ++tid) {
    g_all_ids.insert(g_all_ids.end(), thread_ids[tid].begin(), thread_ids[tid].end());
    g_all_vecs.insert(g_all_vecs.end(), thread_vectors[tid].begin(), thread_vectors[tid].end());
  }

  size_t checked = 0;
  size_t misses = 0;
  std::vector<uint32_t> search_ids;
  std::vector<float> distances;
  for (size_t tid = 0; tid < thread_count; ++tid) {
    const auto& ids = thread_ids[tid];
    const auto& vectors = thread_vectors[tid];
    const std::vector<size_t> sample_offsets = {0, ids.size() - 1};

    for (size_t offset : sample_offsets) {
      std::vector<float> query(vectors.begin() + static_cast<std::ptrdiff_t>(offset * dim),
                               vectors.begin() + static_cast<std::ptrdiff_t>((offset + 1) * dim));
      search_ids.clear();
      distances.clear();
      index.search(query, 10, search_ids, distances);
      ++checked;
      if (std::find(search_ids.begin(), search_ids.end(), ids[offset]) == search_ids.end()) {
        ++misses;
        std::cerr << "[concurrent] missing self id " << ids[offset] << " results=";
        for (uint32_t result_id : search_ids) {
          std::cerr << result_id << ' ';
        }
        std::cerr << std::endl;
      }
    }
  }

  std::cout << "[concurrent] aggregate inserted=" << inserted_total
            << " expected=" << expected_total
            << " sampled_queries=" << checked
            << " misses=" << misses
            << std::endl;

  if (misses > 0) {
    throw std::runtime_error("concurrent insert verification failed: " + std::to_string(misses) +
                             " sampled self-queries missed their id");
  }
}

struct ReadWriteThreadResult {
  size_t count{};
  size_t misses{};
  double elapsed_ms{};
  std::string error;
};

void run_concurrent_read_write_test(ShineIndex& index, uint32_t base_id) {
  const size_t dim = index.dimension();
  const size_t base_count = 128;
  const size_t writers = 2;
  const size_t readers = 4;
  const size_t write_per_thread = 64;
  const size_t read_per_thread = 32;
  const size_t top_k = 10;

  // Phase 1: insert base vectors synchronously
  std::vector<uint32_t> base_ids(base_count);
  for (size_t i = 0; i < base_count; ++i) {
    base_ids[i] = base_id + static_cast<uint32_t>(i);
  }
  const auto base_vectors = make_deterministic_dataset(base_ids, dim);
  const size_t base_inserted = index.insert_count(base_vectors, base_ids);
  std::cout << "[rw] base inserted=" << base_inserted << " expected=" << base_count << std::endl;
  if (base_inserted != base_count) {
    throw std::runtime_error("rw base insert count mismatch");
  }

  g_all_ids.insert(g_all_ids.end(), base_ids.begin(), base_ids.end());
  g_all_vecs.insert(g_all_vecs.end(), base_vectors.begin(), base_vectors.end());

  // Phase 2: prepare writer and reader data
  const uint32_t write_base = base_id + static_cast<uint32_t>(base_count);
  std::vector<std::vector<uint32_t>> write_ids(writers);
  std::vector<std::vector<float>> write_vectors(writers);
  for (size_t w = 0; w < writers; ++w) {
    write_ids[w].resize(write_per_thread);
    for (size_t i = 0; i < write_per_thread; ++i) {
      write_ids[w][i] = write_base + static_cast<uint32_t>(w * write_per_thread + i);
    }
    write_vectors[w] = make_deterministic_dataset(write_ids[w], dim);
  }

  // Each reader queries a slice of base vectors
  std::vector<std::vector<size_t>> reader_offsets(readers);
  for (size_t r = 0; r < readers; ++r) {
    for (size_t i = 0; i < read_per_thread; ++i) {
      reader_offsets[r].push_back((r * read_per_thread + i) % base_count);
    }
  }

  // Phase 3: launch concurrent writers + readers
  std::barrier sync_barrier(static_cast<std::ptrdiff_t>(writers + readers));
  std::vector<ReadWriteThreadResult> writer_results(writers);
  std::vector<ReadWriteThreadResult> reader_results(readers);
  std::vector<std::thread> threads;
  threads.reserve(writers + readers);

  for (size_t w = 0; w < writers; ++w) {
    threads.emplace_back([&, w]() {
      sync_barrier.arrive_and_wait();
      auto t0 = std::chrono::steady_clock::now();
      try {
        writer_results[w].count = index.insert_count(write_vectors[w], write_ids[w]);
      } catch (const std::exception& e) {
        writer_results[w].error = e.what();
      }
      auto t1 = std::chrono::steady_clock::now();
      writer_results[w].elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    });
  }

  for (size_t r = 0; r < readers; ++r) {
    threads.emplace_back([&, r]() {
      sync_barrier.arrive_and_wait();
      auto t0 = std::chrono::steady_clock::now();
      try {
        for (size_t offset : reader_offsets[r]) {
          std::vector<float> query(
            base_vectors.begin() + static_cast<std::ptrdiff_t>(offset * dim),
            base_vectors.begin() + static_cast<std::ptrdiff_t>((offset + 1) * dim));
          std::vector<uint32_t> ids;
          std::vector<float> dists;
          index.search(query, top_k, ids, dists);
          ++reader_results[r].count;
          if (std::find(ids.begin(), ids.end(), base_ids[offset]) == ids.end()) {
            ++reader_results[r].misses;
          }
        }
      } catch (const std::exception& e) {
        reader_results[r].error = e.what();
      }
      auto t1 = std::chrono::steady_clock::now();
      reader_results[r].elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // Aggregate and report
  size_t total_written = 0;
  for (size_t w = 0; w < writers; ++w) {
    total_written += writer_results[w].count;
    std::cout << "[rw] writer " << w
              << " inserted=" << writer_results[w].count
              << " elapsed_ms=" << writer_results[w].elapsed_ms;
    if (!writer_results[w].error.empty()) {
      std::cout << " error=\"" << writer_results[w].error << '"';
    }
    std::cout << std::endl;
  }

  size_t total_queries = 0;
  size_t total_misses = 0;
  for (size_t r = 0; r < readers; ++r) {
    total_queries += reader_results[r].count;
    total_misses += reader_results[r].misses;
    std::cout << "[rw] reader " << r
              << " queries=" << reader_results[r].count
              << " misses=" << reader_results[r].misses
              << " elapsed_ms=" << reader_results[r].elapsed_ms;
    if (!reader_results[r].error.empty()) {
      std::cout << " error=\"" << reader_results[r].error << '"';
    }
    std::cout << std::endl;
  }

  const size_t expected_writes = writers * write_per_thread;
  if (total_written != expected_writes) {
    throw std::runtime_error("[rw] write count mismatch: " + std::to_string(total_written) +
                             " vs " + std::to_string(expected_writes));
  }

  for (size_t w = 0; w < writers; ++w) {
    g_all_ids.insert(g_all_ids.end(), write_ids[w].begin(), write_ids[w].end());
    g_all_vecs.insert(g_all_vecs.end(), write_vectors[w].begin(), write_vectors[w].end());
  }

  const double miss_rate = total_queries > 0
    ? static_cast<double>(total_misses) / static_cast<double>(total_queries)
    : 0.0;
  std::cout << "[rw] concurrent phase: queries=" << total_queries
            << " misses=" << total_misses
            << " miss_rate=" << miss_rate << std::endl;

  if (miss_rate > 0.25) {
    throw std::runtime_error("[rw] concurrent read miss rate too high: " +
                             std::to_string(miss_rate));
  }

  // Phase 4: post-write verification — sample 16 newly inserted vectors
  const size_t post_sample = 16;
  size_t post_misses = 0;
  for (size_t s = 0; s < post_sample; ++s) {
    const size_t w = s % writers;
    const size_t i = (s / writers) % write_per_thread;
    std::vector<float> query(
      write_vectors[w].begin() + static_cast<std::ptrdiff_t>(i * dim),
      write_vectors[w].begin() + static_cast<std::ptrdiff_t>((i + 1) * dim));
    std::vector<uint32_t> ids;
    std::vector<float> dists;
    index.search(query, top_k, ids, dists);
    if (std::find(ids.begin(), ids.end(), write_ids[w][i]) == ids.end()) {
      ++post_misses;
      std::cerr << "[rw] post-write missing id " << write_ids[w][i] << std::endl;
    }
  }

  std::cout << "[rw] post-write verification: sampled=" << post_sample
            << " misses=" << post_misses << std::endl;

  if (post_misses > 0) {
    throw std::runtime_error("[rw] post-write verification failed: " +
                             std::to_string(post_misses) + " misses");
  }

  std::cout << "[rw] passed" << std::endl;
}

void run_recall_test(ShineIndex& index, uint32_t base_id) {
  const size_t dim = index.dimension();
  const size_t num_vectors = 256;
  const size_t K = 10;
  const double min_recall = 0.70;

  // Insert vectors
  std::vector<uint32_t> ids(num_vectors);
  for (size_t i = 0; i < num_vectors; ++i) {
    ids[i] = base_id + static_cast<uint32_t>(i);
  }
  const auto vectors = make_deterministic_dataset(ids, dim);
  const size_t inserted = index.insert_count(vectors, ids);
  std::cout << "[recall] inserted=" << inserted << " expected=" << num_vectors << std::endl;
  if (inserted != num_vectors) {
    throw std::runtime_error("recall insert count mismatch");
  }

  g_all_ids.insert(g_all_ids.end(), ids.begin(), ids.end());
  g_all_vecs.insert(g_all_vecs.end(), vectors.begin(), vectors.end());

  const size_t total = g_all_ids.size();
  std::cout << "[recall] ground truth universe: " << total << " vectors" << std::endl;

  // Brute-force ground truth against ALL vectors in the index
  auto l2_dist = [&](const float* va, const float* vb) -> float {
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      float diff = va[d] - vb[d];
      sum += diff * diff;
    }
    return sum;
  };

  double total_recall = 0.0;
  size_t queries_run = 0;

  for (size_t q = 0; q < num_vectors; ++q) {
    const float* query_vec = vectors.data() + q * dim;

    // Compute distances to ALL vectors in the index
    std::vector<std::pair<float, uint32_t>> dists;
    dists.reserve(total);
    for (size_t j = 0; j < total; ++j) {
      dists.push_back({l2_dist(query_vec, g_all_vecs.data() + j * dim), g_all_ids[j]});
    }
    std::sort(dists.begin(), dists.end());

    // Ground truth top-K
    std::vector<uint32_t> gt;
    for (size_t i = 0; i < dists.size() && gt.size() < K; ++i) {
      gt.push_back(dists[i].second);
    }

    // Search
    std::vector<uint32_t> result_ids;
    std::vector<float> result_dists;
    std::vector<float> query(vectors.begin() + static_cast<std::ptrdiff_t>(q * dim),
                             vectors.begin() + static_cast<std::ptrdiff_t>((q + 1) * dim));
    index.search(query, K, result_ids, result_dists);

    // Compute recall
    size_t hits = 0;
    for (uint32_t rid : result_ids) {
      if (std::find(gt.begin(), gt.end(), rid) != gt.end()) {
        ++hits;
      }
    }
    total_recall += static_cast<double>(hits) / static_cast<double>(K);
    ++queries_run;
  }

  const double avg_recall = queries_run > 0 ? total_recall / static_cast<double>(queries_run) : 0.0;
  std::cout << "[recall] recall@" << K << " = " << avg_recall
            << " (threshold=" << min_recall << ", queries=" << queries_run << ")" << std::endl;

  if (avg_recall < min_recall) {
    throw std::runtime_error("[recall] recall@" + std::to_string(K) + " = " +
                             std::to_string(avg_recall) + " < " + std::to_string(min_recall));
  }

  std::cout << "[recall] passed" << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <service_config> [index_prefix] [concurrent_threads] [vectors_per_thread] [test_reload]" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    const std::string service_config = argv[1];
    const uint32_t unique_base =
      static_cast<uint32_t>((static_cast<uint64_t>(getpid()) * 4099ull) ^ static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::string index_prefix =
      argc >= 3 ? argv[2] : "/tmp/shine_index_smoke_" + std::to_string(static_cast<long long>(getpid()));
    const size_t concurrent_threads = argc >= 4 ? static_cast<size_t>(std::stoul(argv[3])) : 8;
    const size_t vectors_per_thread = argc >= 5 ? static_cast<size_t>(std::stoul(argv[4])) : 64;
    const bool test_reload = argc >= 6 ? std::stoi(argv[5]) != 0 : false;

    ShineIndex index(service_config);
    std::cout << "index type: " << index.getIndexType() << std::endl;
    std::cout << "dimension:  " << index.dimension() << std::endl;
    std::cout << "id base:    " << unique_base << std::endl;
    run_smoke_test(index, index_prefix + "_smoke", unique_base, test_reload);
    run_concurrent_insert_test(index, concurrent_threads, vectors_per_thread, unique_base + 100000u);
    run_concurrent_read_write_test(index, unique_base + 200000u);
    run_recall_test(index, unique_base + 300000u);

    std::cout << "ShineIndex tests passed" << std::endl;
    return EXIT_SUCCESS;

  } catch (const std::exception& e) {
    std::cerr << "ShineIndex test failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
