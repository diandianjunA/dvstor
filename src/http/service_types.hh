#pragma once

#include <chrono>
#include <future>
#include <memory>

#include "common/types.hh"
#include "service/breakdown.hh"

namespace service {

struct InsertRequest {
  node_t id;
  vec<element_t> components;
  std::promise<bool> result;
  std::chrono::steady_clock::time_point enqueued_at{std::chrono::steady_clock::now()};
  std::shared_ptr<breakdown::Sample> breakdown_sample{};
};

struct QueryRequest {
  vec<element_t> components;
  u32 k;
  std::promise<vec<node_t>> result;
  std::chrono::steady_clock::time_point enqueued_at{std::chrono::steady_clock::now()};
  std::shared_ptr<breakdown::Sample> breakdown_sample{};
};

using InsertQueue = concurrent_queue<InsertRequest*>;
using QueryQueue = concurrent_queue<QueryRequest*>;

}  // namespace service

// Expose commonly used types at global scope
using service::InsertQueue;
using service::QueryQueue;
