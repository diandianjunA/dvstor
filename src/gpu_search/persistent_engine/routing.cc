#include "gpu_search/persistent_engine/impl.hh"
#include "gpu_search/persistent_engine/cuda_helpers.hh"

namespace gpu_search {

using namespace persistent_engine_detail;

std::optional<u32> PersistentSearchEngine::Impl::select_centroid_home(
    std::span<const f32> vector) const {
  const std::shared_ptr<const centroid_home::Snapshot> snapshot =
    std::atomic_load_explicit(&centroid_home_snapshot,
                              std::memory_order_acquire);
  if (snapshot == nullptr) return std::nullopt;
  // Publication parsing already validated every centroid. Rechecking the
  // immutable S*D body on every insert would double the routing hot loop.
  return centroid_home::select_published_snapshot(vector, *snapshot);
}

}  // namespace gpu_search
