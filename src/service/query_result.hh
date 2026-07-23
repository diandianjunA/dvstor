#pragma once

#include "common/types.hh"

namespace service {

struct QueryResultItem {
  node_t id{};
  distance_t distance{};
};

using QueryResult = vec<QueryResultItem>;

}  // namespace service
