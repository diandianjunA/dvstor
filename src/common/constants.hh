#pragma once

#include <cstddef>

#include <library/types.hh>

inline constexpr size_t kCacheLineBytes = 64;
// Storage peer QPs are a bounded connection pool. QP0 carries ordered control
// RPCs; the remainder are independent one-sided data lanes. Keep a generous
// hard ceiling while selecting a smaller default in Configuration.
inline constexpr u32 kMaxPeerQps = 16;

// One degree limit is shared by the offline builder, on-disk metadata, CPU
// update path, and persistent GPU query engine.  Keeping the limit here avoids
// producing an index that only fails after its expensive build has completed.
inline constexpr u32 kMaxSupportedGraphDegree = 128;
