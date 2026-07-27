#include "service/compute_service/detail.hh"

using namespace compute_service_detail;

ComputeService::LocalMainSearchOutput
ComputeService::search_local_result(const vec<element_t>& query, u32 k) {
  if (query.size() != config_.dim) {
    throw std::invalid_argument("search dimension mismatch");
  }
  auto sample = std::make_shared<service::breakdown::Sample>(
    service::breakdown::Operation::query,
    breakdown_enabled_.load(std::memory_order_acquire));
  const auto started = std::chrono::steady_clock::now();
  sample->enqueued_at = started;
  sample->mark_started(started, started);
  service::QueryResult results = persistent_search_->search(
    span<const element_t>{query.data(), query.size()}, k);
  sample->mark_finished(std::chrono::steady_clock::now());
  return {.results = std::move(results), .sample = std::move(sample)};
}

ComputeService::LocalMainSearchOutput
ComputeService::search_local_raw_result(
    VectorDType query_dtype, const byte_t* query_data, u32 k) {
  if (query_data == nullptr) throw std::invalid_argument("raw query pointer is null");
  auto sample = std::make_shared<service::breakdown::Sample>(
    service::breakdown::Operation::query,
    breakdown_enabled_.load(std::memory_order_acquire));
  const auto started = std::chrono::steady_clock::now();
  sample->enqueued_at = started;
  sample->mark_started(started, started);
  service::QueryResult results = persistent_search_->search(query_dtype, query_data, k);
  sample->mark_finished(std::chrono::steady_clock::now());
  return {.results = std::move(results), .sample = std::move(sample)};
}

vec<node_t> ComputeService::search_local(const vec<element_t>& query, u32 k) {
  LocalMainSearchOutput output = search_local_result(query, k);
  vec<node_t> ids;
  ids.reserve(std::min<size_t>(k, output.results.size()));
  for (const service::QueryResultItem& result : output.results) {
    if (ids.size() == k) break;
    ids.push_back(result.id);
  }
  if (output.sample && output.sample->finished_flag) {
    service::breakdown::add_sample(
      completed_breakdown_report_.query, *output.sample);
  }
  return ids;
}

vec<node_t> ComputeService::search_local_raw(
    VectorDType query_dtype, const byte_t* query_data, u32 k) {
  LocalMainSearchOutput output = search_local_raw_result(query_dtype, query_data, k);
  vec<node_t> ids;
  ids.reserve(std::min<size_t>(k, output.results.size()));
  for (const service::QueryResultItem& result : output.results) {
    if (ids.size() == k) break;
    ids.push_back(result.id);
  }
  if (output.sample && output.sample->finished_flag) {
    service::breakdown::add_sample(
      completed_breakdown_report_.query, *output.sample);
  }
  return ids;
}

vec<node_t> ComputeService::search_raw(
    VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k) {
  if (dim != config_.dim) throw std::invalid_argument("raw search dimension mismatch");
  return search_local_raw(query_dtype, query_data, k);
}

vec<node_t> ComputeService::search(const vec<element_t>& query, u32 k) {
  return search_local(query, k);
}
