#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/types.hh"

namespace tools::breakdown_benchmark {

class DynamicIdPool {
public:
  explicit DynamicIdPool(std::filesystem::path path) : path_(std::move(path)) {}

  bool state_file_exists() const {
    std::error_code error;
    const bool exists = std::filesystem::exists(path_, error);
    if (error) {
      throw std::runtime_error("failed to inspect dynamic ID state: " + path_.string());
    }
    return exists;
  }

  void load(u32 id_namespace_size, u32 minimum_id = 0) {
    std::ifstream input(path_, std::ios::binary | std::ios::ate);
    if (!input) {
      throw std::runtime_error("failed to open dynamic ID state: " + path_.string());
    }
    const auto end = input.tellg();
    if (end < static_cast<std::streamoff>(header_bytes())) {
      throw std::runtime_error("truncated dynamic ID state: " + path_.string());
    }
    const auto file_size = static_cast<u64>(end);
    input.seekg(0);
    std::array<char, 8> magic{};
    u64 count = 0;
    input.read(magic.data(), magic.size());
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    const std::array<char, 8> expected{'D', 'V', 'D', 'Y', 'N', 'I', 'D', '1'};
    if (!input || magic != expected || count > id_namespace_size ||
        file_size != header_bytes() + count * sizeof(u32)) {
      throw std::runtime_error("invalid dynamic ID state: " + path_.string());
    }
    std::vector<u32> loaded(static_cast<size_t>(count));
    if (count != 0) {
      input.read(reinterpret_cast<char*>(loaded.data()), count * sizeof(u32));
    }
    if (!input) {
      throw std::runtime_error("truncated dynamic ID state entries: " + path_.string());
    }
    std::unordered_set<u32> unique;
    unique.reserve(loaded.size());
    for (const u32 id : loaded) {
      if (id < minimum_id || id >= id_namespace_size || !unique.insert(id).second) {
        throw std::runtime_error("dynamic ID state has an invalid or duplicate ID");
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ids_.assign(loaded.begin(), loaded.end());
  }

  void save() const {
    std::vector<u32> snapshot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot.assign(ids_.begin(), ids_.end());
    }
    const auto temporary = std::filesystem::path(path_.string() + ".tmp");
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      const std::array<char, 8> magic{'D', 'V', 'D', 'Y', 'N', 'I', 'D', '1'};
      const u64 count = snapshot.size();
      output.write(magic.data(), magic.size());
      output.write(reinterpret_cast<const char*>(&count), sizeof(count));
      if (!snapshot.empty()) {
        output.write(reinterpret_cast<const char*>(snapshot.data()),
                     snapshot.size() * sizeof(u32));
      }
      output.flush();
      if (!output) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("failed to write dynamic ID state: " + temporary.string());
      }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path_, error);
    if (error) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      throw std::runtime_error(
        "failed to publish dynamic ID state " + path_.string() + ": " + error.message());
    }
  }

  void push(u32 id) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ids_.push_back(id);
    }
    available_.notify_one();
  }

  std::optional<u32> pop(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!available_.wait_until(lock, deadline, [this] { return !ids_.empty(); })) {
      return std::nullopt;
    }
    const u32 id = ids_.front();
    ids_.pop_front();
    return id;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_.size();
  }

private:
  static constexpr u64 header_bytes() { return 16; }

  std::filesystem::path path_;
  mutable std::mutex mutex_;
  std::condition_variable available_;
  std::deque<u32> ids_;
};

}  // namespace tools::breakdown_benchmark
