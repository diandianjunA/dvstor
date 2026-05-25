#include "tools/vamana_offline/progress.hh"

#include <iomanip>
#include <sstream>

namespace tools::vamana_offline {

size_t effective_thread_count(u32 configured_threads) {
  const size_t detected = std::thread::hardware_concurrency();
  return configured_threads == 0 ? std::max<size_t>(detected, 1) : configured_threads;
}

str format_duration(std::chrono::steady_clock::duration duration) {
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  const auto hours = seconds / 3600;
  const auto minutes = (seconds % 3600) / 60;
  const auto secs = seconds % 60;
  std::ostringstream os;
  if (hours > 0) os << hours << "h" << minutes << "m" << secs << "s";
  else if (minutes > 0) os << minutes << "m" << secs << "s";
  else os << secs << "s";
  return os.str();
}


}  // namespace tools::vamana_offline
