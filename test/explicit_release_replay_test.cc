#include <array>
#include <cassert>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "service/storage_owner_protocol.hh"

namespace {

enum class ReceiptPhase : std::uint8_t {
  prepared,
  armed,
  aborted,
  cleanup_activated,
};

struct Receipt {
  ReceiptPhase phase{ReceiptPhase::prepared};
  std::uint64_t result{};
};

// Small-capacity executable model of the production receipt lifecycle. There
// is deliberately no clock and no eviction scan: admission depends only on
// true in-flight entries, and release is an idempotent postcondition.
class ExplicitReleaseTableModel {
public:
  explicit ExplicitReleaseTableModel(std::size_t capacity)
      : capacity_(capacity) {
    assert(capacity_ != 0);
  }

  bool claim(std::uint64_t token, Receipt receipt) {
    if (records_.contains(token)) return true;
    if (records_.size() == capacity_) return false;
    return records_.emplace(token, receipt).second;
  }

  std::optional<Receipt> find(std::uint64_t token) const {
    const auto position = records_.find(token);
    if (position == records_.end()) return std::nullopt;
    return position->second;
  }

  bool transition(std::uint64_t token, ReceiptPhase phase,
                  std::uint64_t result) {
    const auto position = records_.find(token);
    if (position == records_.end()) return false;
    position->second = Receipt{phase, result};
    return true;
  }

  bool release(std::uint64_t token) {
    records_.erase(token);
    return true;
  }

  std::size_t size() const { return records_.size(); }

private:
  std::size_t capacity_{};
  std::unordered_map<std::uint64_t, Receipt> records_;
};

// Models the conservative per-authority completion prefix retained by the
// older release design. RC receive order assigns sequences; workers may
// complete them out of order. Production Stage1 release below deliberately
// uses the narrower same-token quiescence condition instead.
class OrderedCompletionModel {
public:
  void complete(std::uint64_t sequence) {
    assert(sequence != 0);
    if (sequence == prefix_ + 1) {
      ++prefix_;
      while (out_of_order_.erase(prefix_ + 1) != 0) ++prefix_;
      return;
    }
    assert(sequence > prefix_ + 1);
    assert(out_of_order_.insert(sequence).second);
  }

  bool may_release(std::uint64_t release_sequence) const {
    return release_sequence != 0 && prefix_ + 1 == release_sequence;
  }

private:
  std::uint64_t prefix_{};
  std::unordered_set<std::uint64_t> out_of_order_;
};

enum class ReleaseAttempt : std::uint8_t {
  retry,
  resolved,
};

// Models the receiver-side token quiescence probe.  A release never waits on
// an older Execute while occupying a worker: it reports retry, preserving the
// receipt, and the authority reposts the same request ID after backoff.
class NonblockingStage1ReleaseModel {
public:
  void begin_execute(std::uint64_t token) {
    ++tokens_[token].inflight_execute;
  }

  void finish_execute(std::uint64_t token) {
    TokenState& state = tokens_[token];
    assert(state.inflight_execute != 0);
    --state.inflight_execute;
  }

  ReleaseAttempt release(std::uint64_t token, std::uint64_t request_id) {
    assert(token != 0 && request_id != 0);
    ++attempts_;
    TokenState& state = tokens_[token];
    if (state.release_request_id == 0) {
      state.release_request_id = request_id;
    } else {
      // Transport retry and lost-ACK replay must retain the registry identity
      // as well as the semantic token.
      assert(state.release_request_id == request_id);
    }
    if (state.inflight_execute != 0) return ReleaseAttempt::retry;
    state.receipt_present = false;
    return ReleaseAttempt::resolved;
  }

  bool receipt_present(std::uint64_t token) const {
    const auto position = tokens_.find(token);
    return position == tokens_.end() || position->second.receipt_present;
  }
  std::size_t attempts() const { return attempts_; }

private:
  struct TokenState {
    std::size_t inflight_execute{};
    bool receipt_present{true};
    std::uint64_t release_request_id{};
  };

  std::unordered_map<std::uint64_t, TokenState> tokens_;
  std::size_t attempts_{};
};

struct OperationKey {
  std::uint32_t authority_shard{};
  std::uint32_t source_client{};
  std::uint32_t item_index{};
  std::uint64_t client_batch_id{};

  bool operator==(const OperationKey&) const = default;
};

struct OperationKeyHash {
  std::size_t operator()(const OperationKey& key) const {
    std::size_t value = std::hash<std::uint64_t>{}(key.client_batch_id);
    value ^= std::hash<std::uint64_t>{}(
      (static_cast<std::uint64_t>(key.authority_shard) << 32) |
      key.source_client) + 0x9e3779b97f4a7c15ull +
      (value << 6) + (value >> 2);
    value ^= std::hash<std::uint32_t>{}(key.item_index) +
      0x9e3779b97f4a7c15ull + (value << 6) + (value >> 2);
    return value;
  }
};

// Mirrors the production Stage1 receipt-table topology: the key selects one
// of 64 independent mutex/map pairs, and duplicate semantics are resolved
// while holding only that key's shard.
class ShardedReceiptTableModel {
public:
  static constexpr std::size_t kShardCount = 64;

  bool claim(const OperationKey& key, Receipt receipt) {
    Shard& shard = shard_for(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    if (shard.records.contains(key)) return true;
    if (shard.records.size() == per_shard_capacity_) return false;
    return shard.records.emplace(key, receipt).second;
  }

  std::size_t shard_index(const OperationKey& key) const {
    return OperationKeyHash{}(key) & (kShardCount - 1);
  }

  const void* lock_identity(const OperationKey& key) {
    return &shard_for(key).mutex;
  }

  std::size_t records_in_shard(const OperationKey& key) {
    Shard& shard = shard_for(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    return shard.records.size();
  }

  template <typename Callback>
  void with_shard_lock(const OperationKey& key, Callback&& callback) {
    Shard& shard = shard_for(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    callback();
  }

private:
  struct Shard {
    std::mutex mutex;
    std::unordered_map<OperationKey, Receipt, OperationKeyHash> records;
  };

  Shard& shard_for(const OperationKey& key) {
    return shards_[shard_index(key)];
  }

  static constexpr std::size_t per_shard_capacity_ = 16;
  std::array<Shard, kShardCount> shards_;
};

void test_sustained_stage1_release_has_no_rate_window() {
  ExplicitReleaseTableModel table(2);
  for (std::uint64_t token = 1; token <= 100'000; ++token) {
    assert(table.claim(token, {ReceiptPhase::prepared, 0}));
    // Same-token execute retry observes the same in-flight artifact.
    assert(table.claim(token, {ReceiptPhase::prepared, 0}));
    assert(table.transition(token, ReceiptPhase::armed, token + 1000));
    const auto replay = table.find(token);
    assert(replay.has_value());
    assert(replay->phase == ReceiptPhase::armed);
    assert(replay->result == token + 1000);

    // First release reaches the receiver but its response is lost. Retrying
    // release must ACK the already-missing postcondition.
    assert(table.release(token));
    assert(table.release(token));
    assert(table.size() == 0);
  }
}

void test_capacity_tracks_only_true_inflight_operations() {
  ExplicitReleaseTableModel table(2);
  assert(table.claim(1, {ReceiptPhase::prepared, 0}));
  assert(table.claim(2, {ReceiptPhase::prepared, 0}));
  assert(!table.claim(3, {ReceiptPhase::prepared, 0}));
  assert(table.transition(1, ReceiptPhase::armed, 11));
  // Terminal without release is still genuinely uncertain/in flight.
  assert(!table.claim(3, {ReceiptPhase::prepared, 0}));
  assert(table.release(1));
  assert(table.claim(3, {ReceiptPhase::prepared, 0}));
}

void test_abort_fence_waits_for_parallel_worker_prefix() {
  ExplicitReleaseTableModel table(2);
  OrderedCompletionModel completion;

  // Sequence 1 is an execute already delivered on the RC QP. Sequence 2 is
  // abort, which may finish first and installs a compact missing-token fence.
  assert(table.claim(77, {ReceiptPhase::aborted, 0}));
  completion.complete(2);
  assert(!completion.may_release(3));
  const auto delayed_execute = table.find(77);
  assert(delayed_execute.has_value());
  assert(delayed_execute->phase == ReceiptPhase::aborted);

  completion.complete(1);
  assert(completion.may_release(3));
  assert(table.release(77));
  completion.complete(3);

  // Only after release ACK may an aborted public token acquire a fresh
  // authority lease and execute Stage1 again.
  assert(table.claim(77, {ReceiptPhase::prepared, 0}));
}

void test_stage1_release_defers_without_blocking_a_worker() {
  NonblockingStage1ReleaseModel receipt;
  constexpr std::uint64_t request_id = 91;
  constexpr std::uint64_t blocked_token = 7;
  constexpr std::uint64_t independent_token = 8;

  // Keep an Execute live on a separate worker. The release must return retry
  // before that worker is allowed to finish, rather than waiting on it.
  std::barrier execute_started(2);
  std::barrier allow_execute_finish(2);
  std::thread execute([&]() {
    receipt.begin_execute(blocked_token);
    execute_started.arrive_and_wait();
    allow_execute_finish.arrive_and_wait();
    receipt.finish_execute(blocked_token);
  });
  execute_started.arrive_and_wait();
  assert(receipt.release(blocked_token, request_id) == ReleaseAttempt::retry);
  assert(receipt.receipt_present(blocked_token));

  // Quiescence is per semantic token, not a global source-shard prefix.
  assert(receipt.release(independent_token, request_id + 1) ==
         ReleaseAttempt::resolved);
  assert(!receipt.receipt_present(independent_token));

  // Once the older handler quiesces, the identical release request resolves.
  allow_execute_finish.arrive_and_wait();
  execute.join();
  assert(receipt.release(blocked_token, request_id) ==
         ReleaseAttempt::resolved);
  assert(!receipt.receipt_present(blocked_token));

  // A lost successful ACK is harmless: replay sees the already-missing
  // postcondition and resolves again without recreating the receipt.
  assert(receipt.release(blocked_token, request_id) ==
         ReleaseAttempt::resolved);
  assert(!receipt.receipt_present(blocked_token));
  assert(receipt.attempts() == 4);
}

void test_cleanup_duplicate_and_lost_release_response() {
  ExplicitReleaseTableModel table(1);
  for (std::uint64_t token = 1; token <= 100'000; ++token) {
    assert(table.claim(token, {ReceiptPhase::cleanup_activated, token + 9}));
    const auto duplicate = table.find(token);
    assert(duplicate.has_value());
    assert(duplicate->phase == ReceiptPhase::cleanup_activated);
    assert(duplicate->result == token + 9);
    assert(table.release(token));
    assert(table.release(token));
    assert(table.size() == 0);
  }
}

void test_stage1_receipts_use_independent_key_shards() {
  ShardedReceiptTableModel table;
  OperationKey first{1, 7, 3, 1};
  OperationKey second = first;
  do {
    ++second.client_batch_id;
  } while (table.shard_index(first) == table.shard_index(second));

  assert(table.lock_identity(first) != table.lock_identity(second));
  // Both callbacks rendezvous while holding their selected shard mutex. This
  // can complete only when different-token shards do not share a global lock.
  std::barrier rendezvous(3);
  std::thread a([&]() {
    table.with_shard_lock(first, [&]() { rendezvous.arrive_and_wait(); });
  });
  std::thread b([&]() {
    table.with_shard_lock(second, [&]() { rendezvous.arrive_and_wait(); });
  });
  rendezvous.arrive_and_wait();
  a.join();
  b.join();
}

void test_stage1_same_token_claim_is_idempotent_under_concurrency() {
  ShardedReceiptTableModel table;
  const OperationKey key{2, 9, 11, 101};
  constexpr std::size_t kThreads = 16;
  std::barrier start(static_cast<std::ptrdiff_t>(kThreads + 1));
  std::array<bool, kThreads> claimed{};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (std::size_t i = 0; i < kThreads; ++i) {
    workers.emplace_back([&, i]() {
      start.arrive_and_wait();
      claimed[i] = table.claim(
        key, {ReceiptPhase::prepared, static_cast<std::uint64_t>(i)});
    });
  }
  start.arrive_and_wait();
  for (std::thread& worker : workers) worker.join();
  for (bool result : claimed) assert(result);
  assert(table.records_in_shard(key) == 1);
}

}  // namespace

int main() {
  namespace protocol = service::storage_owner;
  static_assert(static_cast<u32>(protocol::Stage1ArmAction::release) == 3);
  static_assert(static_cast<u32>(
                  protocol::CleanupActivateAction::release) == 2);
  test_sustained_stage1_release_has_no_rate_window();
  test_capacity_tracks_only_true_inflight_operations();
  test_abort_fence_waits_for_parallel_worker_prefix();
  test_stage1_release_defers_without_blocking_a_worker();
  test_cleanup_duplicate_and_lost_release_response();
  test_stage1_receipts_use_independent_key_shards();
  test_stage1_same_token_claim_is_idempotent_under_concurrency();
  return 0;
}
