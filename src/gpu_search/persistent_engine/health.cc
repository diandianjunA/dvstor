#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

using namespace persistent_engine_detail;

std::string PersistentSearchEngine::Impl::unhealthy_message() {
  std::lock_guard<std::mutex> lock(health_mutex);
  return health_error.empty()
    ? "persistent GPU query engine is unhealthy" : health_error;
}

void PersistentSearchEngine::Impl::reject_query_slot(u32 slot) {
  if (slot >= query_slots || query_slot_states == nullptr) return;
  QuerySlotState& state = query_slot_states[slot];
  u32 phase = state.phase.load(std::memory_order_acquire);
  while (phase == static_cast<u32>(QuerySlotPhase::preparing) ||
         phase == static_cast<u32>(QuerySlotPhase::pending)) {
    if (state.phase.compare_exchange_weak(
          phase, static_cast<u32>(QuerySlotPhase::rejected),
          std::memory_order_release, std::memory_order_acquire)) {
      state.phase.notify_all();
      return;
    }
  }
}

void PersistentSearchEngine::Impl::release_query_slot(u32 slot) {
  QuerySlotState& state = query_slot_states[slot];
  // Publish reusable state before publishing the slot into the free queue.
  // The queue cell's release/acquire hand-off guarantees a successful pop
  // observes this phase before attempting free -> preparing.
  state.phase.store(static_cast<u32>(QuerySlotPhase::free),
                    std::memory_order_release);
  if (!free_slots->try_push(slot)) {
    // A full free-slot queue means the same slot was released twice. Continuing
    // would permit concurrent reuse of query/result scratch and corrupt recall.
    std::cerr << "[gpu-search] fatal duplicate bounded query-slot release: "
              << slot << '\n';
    std::terminate();
  }
}

void PersistentSearchEngine::Impl::mark_unhealthy(const std::string& message) {
  {
    std::lock_guard<std::mutex> lock(health_mutex);
    if (!healthy.load(std::memory_order_relaxed)) return;
    health_error = message;
    healthy.store(false, std::memory_order_release);
  }
  query_stop.store(true, std::memory_order_release);
  if (free_slots != nullptr) free_slots->notify_all();
  if (admission_queue != nullptr) admission_queue->notify_all();
  reject_all_pending(message);
  std::cerr << "[gpu-search] query engine entered fail-stop mode: "
            << message << '\n';
}

void PersistentSearchEngine::Impl::reject_all_pending(
    const std::string& message) {
  (void)message;
  if (query_slot_states == nullptr) return;
  for (u32 slot = 0; slot < query_slots; ++slot) {
    reject_query_slot(slot);
  }
}

void PersistentSearchEngine::Impl::bind_cuda_device(
    const char* operation) const {
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(GPU navigation)");
  if (current_device != static_cast<int>(config.gpu_device)) {
    check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)), operation);
  }
}

}  // namespace gpu_search
