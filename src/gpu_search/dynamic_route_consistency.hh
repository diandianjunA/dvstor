#pragma once

#include "gpu_search/types.hh"

namespace gpu_search {

#if defined(__CUDACC__)
#define DVSTOR_ROUTE_HD __host__ __device__
#else
#define DVSTOR_ROUTE_HD
#endif

// A route read is usable only when the writer sequence stayed at the same
// even value across metadata validation *and* PQ scoring.  Keeping this tiny
// predicate host-testable makes it difficult to accidentally narrow the
// protected window back to metadata alone.
DVSTOR_ROUTE_HD inline constexpr bool dynamic_route_window_stable(
    u64 before, u64 after) {
  return (before & 1u) == 0 && before == after;
}

#undef DVSTOR_ROUTE_HD

}  // namespace gpu_search
