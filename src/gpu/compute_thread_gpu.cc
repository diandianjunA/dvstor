#include "compute_thread.hh"

#include <cuda_runtime.h>

void ComputeThread::poll_gpu_events() {
    const u32 num_coros = gpu_post_balances.size();
    for (u32 coro_id = 0; coro_id < num_coros; ++coro_id) {
        if (gpu_post_balances[coro_id].load(std::memory_order_relaxed) <= 0) continue;

        // Check if this coroutine's GPU event has completed
        cudaError_t status = cudaEventQuery(gpu_buffers.event(coro_id));
        if (status == cudaSuccess) {
            gpu_post_balances[coro_id].store(0, std::memory_order_relaxed);
        }
        // cudaErrorNotReady means still in progress — do nothing
    }
}
