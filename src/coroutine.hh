#pragma once

#include <coroutine>

#include "remote_pointer.hh"

/**
 * Coroutines called by other coroutines.
 * Handle is destroyed by the destructor to prevent memory leaks.
 */
struct MinorCoroutine {
  struct promise_type {
    MinorCoroutine get_return_object() { return MinorCoroutine{Handle::from_promise(*this)}; }
    // std::suspend_never directly runs the coroutine (the object is created after first suspend)
    static std::suspend_never initial_suspend() { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    static void return_void() {}
    static void unhandled_exception() { throw; }
  };

  using Handle = std::coroutine_handle<promise_type>;

  explicit MinorCoroutine(Handle handle) : handle(handle) {}

  ~MinorCoroutine() {
    if (handle) {
      handle.destroy();
    }
  }

  MinorCoroutine(const MinorCoroutine&) = delete;
  MinorCoroutine(MinorCoroutine&&) = delete;
  MinorCoroutine& operator=(const MinorCoroutine&) = delete;
  MinorCoroutine& operator=(MinorCoroutine&&) noexcept = delete;

  Handle handle;
};

/**
 * VamanaCoroutine: replaces HNSWCoroutine for the GPU-based Vamana index.
 * Uses beam search state instead of HNSW-style heaps.
 */
struct VamanaCoroutine {
  struct promise_type {
    VamanaCoroutine get_return_object() { return VamanaCoroutine{Handle::from_promise(*this)}; }
    static std::suspend_always initial_suspend() { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    static void return_void() {}
    static void unhandled_exception() { throw; }
  };

  using Handle = std::coroutine_handle<promise_type>;
  Handle handle;

  // Beam search state
  struct BeamEntry {
    RemotePtr rptr;
    distance_t distance;
    bool expanded{false};
  };

  vec<BeamEntry> beam{};
  hashset_t<RemotePtr> visited_nodes{};
  vec<RemotePtr> scratch_unvisited{};
  vec<RemotePtr> reserved_ptrs_a{};
  vec<RemotePtr> reserved_ptrs_b{};
  vec<RemotePtr> indirect_candidate_ptrs{};
  vec<u32> indirect_candidate_indices{};
  vec<float> scratch_distances{};
  vec<u32> scratch_indices_a{};
  vec<u32> scratch_indices_b{};
  vec<u8> scratch_flags{};

  // GPU state
  bool gpu_pending{false};
};

struct StorageOwnerInsertCoroutine {
  struct promise_type {
    StorageOwnerInsertCoroutine get_return_object() {
      return StorageOwnerInsertCoroutine{Handle::from_promise(*this)};
    }
    static std::suspend_always initial_suspend() { return {}; }
    static std::suspend_always final_suspend() noexcept { return {}; }
    static void return_void() {}
    static void unhandled_exception() { throw; }
  };

  using Handle = std::coroutine_handle<promise_type>;
  Handle handle;
};
