  std::string unhealthy_message() {
    std::lock_guard<std::mutex> lock(admission_mutex);
    return health_error.empty() ? "persistent GPU query engine is unhealthy" : health_error;
  }

  void reject_submission(const PendingSubmission& submission,
                         const std::string& message) {
    std::shared_ptr<PendingQuery> pending;
    {
      std::lock_guard<std::mutex> lock(pending_mutex);
      const auto iterator = pending_queries.find(submission.descriptor.request_id);
      if (iterator != pending_queries.end()) {
        pending = std::move(iterator->second);
        pending_queries.erase(iterator);
      }
    }
    if (!pending) return;
    if (active_query_tickets != nullptr) {
      active_query_tickets[pending->slot].store(0, std::memory_order_release);
    }
    if (active_query_snapshots != nullptr) {
      active_query_snapshots[pending->slot].store(0, std::memory_order_release);
    }
    pending->promise.set_exception(
      std::make_exception_ptr(std::runtime_error(message)));
    {
      std::lock_guard<std::mutex> lock(slot_mutex);
      free_slots.push_back(pending->slot);
    }
    slot_cv.notify_one();
    pending_count.fetch_sub(1, std::memory_order_release);
    maintenance_cv.notify_all();
  }

  void mark_unhealthy(const std::string& message) {
    std::deque<PendingSubmission> rejected;
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      if (!healthy.load(std::memory_order_relaxed)) return;
      health_error = message;
      healthy.store(false, std::memory_order_release);
      rejected.swap(admission_queue);
    }
    admission_cv.notify_all();
    slot_cv.notify_all();
    for (const PendingSubmission& submission : rejected) {
      reject_submission(submission, message);
    }
    std::cerr << "[gpu-search] query engine entered fail-stop mode: "
              << message << '\n';
  }

  void reject_queued_submissions(const std::string& message) {
    std::deque<PendingSubmission> rejected;
    {
      std::lock_guard<std::mutex> lock(admission_mutex);
      rejected.swap(admission_queue);
    }
    for (const PendingSubmission& submission : rejected) {
      reject_submission(submission, message);
    }
  }

  void reject_all_pending(const std::string& message) {
    std::vector<std::shared_ptr<PendingQuery>> rejected;
    {
      std::lock_guard<std::mutex> lock(pending_mutex);
      rejected.reserve(pending_queries.size());
      for (auto& [request_id, pending] : pending_queries) {
        (void)request_id;
        rejected.push_back(std::move(pending));
      }
      pending_queries.clear();
    }
    if (rejected.empty()) return;
    {
      std::lock_guard<std::mutex> lock(slot_mutex);
      for (const auto& pending : rejected) {
        if (active_query_tickets != nullptr) {
          active_query_tickets[pending->slot].store(0, std::memory_order_release);
        }
        if (active_query_snapshots != nullptr) {
          active_query_snapshots[pending->slot].store(0, std::memory_order_release);
        }
        free_slots.push_back(pending->slot);
      }
    }
    for (const auto& pending : rejected) {
      try {
        pending->promise.set_exception(
          std::make_exception_ptr(std::runtime_error(message)));
      } catch (const std::future_error&) {
      }
    }
    pending_count.fetch_sub(rejected.size(), std::memory_order_release);
    slot_cv.notify_all();
    maintenance_cv.notify_all();
  }

  void bind_cuda_device(const char* operation) const {
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(GPU navigation)");
    if (current_device != static_cast<int>(config.gpu_device)) {
      check_cuda(cudaSetDevice(static_cast<int>(config.gpu_device)), operation);
    }
  }

