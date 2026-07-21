#include "gpu_search/persistent_engine/impl.hh"

namespace gpu_search {

void PersistentSearchEngine::Impl::submit_centroid_route_publication(
    const CentroidRoutePublishDescriptor& descriptor) {
  const auto timeout = std::chrono::milliseconds(std::clamp<u32>(
    config.storage_owner_rpc_timeout_ms, 1000, 5000));
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!route_submissions.try_push(descriptor)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
        "persistent GPU centroid-route command queue is not making progress");
    }
    std::this_thread::yield();
  }

  CentroidRoutePublishCompletion completion{};
  while (!route_completions.try_pop(completion)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
        "persistent GPU centroid-route publication timed out");
    }
    std::this_thread::yield();
  }
  if (completion.command_id != descriptor.command_id ||
      completion.status != 0 ||
      completion.update_count != descriptor.update_count) {
    throw std::runtime_error(
      "persistent GPU centroid-route publication failed: command=" +
      std::to_string(completion.command_id) + " status=" +
      std::to_string(completion.status) + " updates=" +
      std::to_string(completion.update_count));
  }
}

}  // namespace gpu_search
