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

// Persistent GPU navigation encodes logical ordinals in 30 bits and uses
// fixed compile-time Beam/PQ workspaces. Share these limits with builders and
// configuration preflight so expensive incompatible indexes are never made.
inline constexpr u64 kMaxGpuNavigationNodes = (u64{1} << 30) - 1;
inline constexpr u32 kMaxPersistentTraversalBeam = 128;
inline constexpr u32 kMaxPersistentSubquantizers = 32;

// RemotePtr wire layout reserves six shard bits and a 34-bit 16-byte
// offset. Validate deployment topology and registered-memory sizing before
// opening RDMA devices or connecting peers.
inline constexpr u32 kMaxStorageShards = 64;
inline constexpr u32 kMaxStorageRegionGiB = 256;
