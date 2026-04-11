#include "gpu/gpu_awaitable.hh"
#include "compute_thread.hh"

namespace gpu {

void GpuAwaitable::await_suspend(std::coroutine_handle<>) const {
    thread->track_gpu_post();
}

}  // namespace gpu
