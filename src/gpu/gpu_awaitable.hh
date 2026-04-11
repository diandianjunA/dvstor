#pragma once

/**
 * GpuAwaitable: coroutine awaitable that suspends until a GPU kernel completes.
 *
 * Usage pattern:
 *   1. Stage data to GPU pinned buffer → cudaMemcpyAsync → launch kernel → cudaEventRecord
 *   2. co_await GpuAwaitable{thread}
 *   3. Scheduler polls cudaEventQuery(event) → resumes coroutine when done
 *
 * Before returning GpuAwaitable, the caller must have already:
 *   - Launched the kernel on the coroutine's stream
 *   - Called cudaEventRecord(event, stream)
 */

#include <coroutine>

// Forward declare
class ComputeThread;

namespace gpu {

struct GpuAwaitable {
    ComputeThread* thread;

    // Never ready immediately — always suspend to let GPU work overlap with RDMA
    static bool await_ready() { return false; }

    // Track that we have an outstanding GPU operation
    void await_suspend(std::coroutine_handle<>) const;

    // Nothing to return
    static void await_resume() {}
};

}  // namespace gpu
