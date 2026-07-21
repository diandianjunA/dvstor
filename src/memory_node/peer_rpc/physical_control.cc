#include "memory_node/peer_rpc/detail.hh"

#include "vamana/storage_layout_resolver.hh"

namespace protocol = service::storage_owner;
namespace authority = memory_node_storage_owner_index_detail;

namespace {

bool same_token(const protocol::AuthorityOperationToken& lhs,
                const protocol::AuthorityOperationToken& rhs) {
  return lhs.source_client == rhs.source_client &&
    lhs.item_index == rhs.item_index &&
    lhs.client_batch_id == rhs.client_batch_id;
}

bool same_cleanup_identity(const protocol::CleanupActivateItem& lhs,
                           const protocol::CleanupActivateItem& rhs) {
  return same_token(lhs.token, rhs.token) && lhs.old_raw == rhs.old_raw &&
    lhs.id == rhs.id && lhs.old_generation == rhs.old_generation &&
    lhs.authority_shard == rhs.authority_shard;
}

}  // namespace

bool MemoryNode::activate_local_cleanup_items(
    u32 authority_shard,
    span<const protocol::CleanupActivateItem> items,
    vec<protocol::CleanupActivateResult>& results,
    const Configuration& config) {
  results.assign(items.size(), {});
  if (authority_shard >= num_storage_nodes_ || items.empty()) return false;

  bool success = true;
  for (size_t index = 0; index < items.size(); ++index) {
    const protocol::CleanupActivateItem& item = items[index];
    protocol::CleanupActivateResult& output = results[index];
    output.target_raw = item.old_raw;
    output.token = item.token;
    output.status = static_cast<u32>(protocol::MutationStatus::failed);

    const RemotePtr target{item.old_raw};
    const auto action = static_cast<protocol::CleanupActivateAction>(
      item.action);
    if (!authority::valid_authority_operation(item.token) ||
        item.authority_shard != authority_shard ||
        (action != protocol::CleanupActivateAction::activate &&
         action != protocol::CleanupActivateAction::release) ||
        !valid_local_storage_node_pointer(target)) {
      success = false;
      continue;
    }

    const Stage1OperationKey key{
      .authority_shard = authority_shard,
      .source_client = item.token.source_client,
      .item_index = item.token.item_index,
      .client_batch_id = item.token.client_batch_id,
    };
    CleanupActivationDedupeShard& dedupe = cleanup_activation_dedupe_[
      Stage1OperationKeyHash{}(key) &
      (kCleanupActivationShardCount - 1)];
    if (action == protocol::CleanupActivateAction::release) {
      std::lock_guard<std::mutex> lock(dedupe.mutex);
      const auto existing = dedupe.records.find(key);
      if (existing == dedupe.records.end()) {
        // The per-source completion barrier in the cleanup worker preserves
        // RC receive order across the parallel pool. A retry after a lost
        // release response therefore observes the already reached
        // postcondition and cannot race an older activation.
        output.status = static_cast<u32>(protocol::MutationStatus::ok);
        continue;
      }
      if (existing->second.in_progress ||
          !same_cleanup_identity(existing->second.item, item)) {
        success = false;
        continue;
      }
      output.maintenance_sequence =
        existing->second.result.maintenance_sequence;
      dedupe.records.erase(existing);
      output.status = static_cast<u32>(protocol::MutationStatus::ok);
      continue;
    }

    bool execute = false;
    {
      std::unique_lock<std::mutex> lock(dedupe.mutex);
      auto existing = dedupe.records.find(key);
      if (existing != dedupe.records.end() &&
          existing->second.in_progress) {
        const auto deadline = std::chrono::steady_clock::now() +
          std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
        dedupe.changed.wait_until(lock, deadline, [&]() {
          const auto current = dedupe.records.find(key);
          return current == dedupe.records.end() ||
            !current->second.in_progress;
        });
        existing = dedupe.records.find(key);
      }
      if (existing != dedupe.records.end()) {
        if (!same_cleanup_identity(existing->second.item, item) ||
            existing->second.in_progress) {
          success = false;
        } else {
          output = existing->second.result;
          success &= output.status ==
            static_cast<u32>(protocol::MutationStatus::ok);
        }
        continue;
      }

      // O(1) admission: completed activation receipts are true in-flight state
      // until their authority sends release; no timeout scan participates in
      // the update-rate fast path.
      if (dedupe.records.size() >=
          cleanup_activation_dedupe_limit_per_shard_) {
        success = false;
        continue;
      }
      const auto [position, inserted] = dedupe.records.emplace(
        key, CleanupActivationRecord{
          .item = item,
          .result = output,
          .in_progress = true,
        });
      lib_assert(inserted && position->second.in_progress,
                 "cleanup activation dedupe claim failed");
      execute = true;
    }
    lib_assert(execute, "cleanup activation did not own its dedupe claim");

    bool activated = false;
    u64 maintenance_sequence = 0;
    protocol::MutationStatus item_status =
      protocol::MutationStatus::failed;
    NodeSnapshot snapshot;
    const bool snapshot_read = read_node_snapshot(target, snapshot);
    if (!snapshot_read || snapshot.id != item.id ||
        snapshot.generation != item.old_generation) {
      item_status = protocol::MutationStatus::not_found;
    } else if (snapshot.deleted) {
      // A fresh activation may observe cleanup already completed. The identity
      // proves that the requested postcondition has already been reached. Use
      // the current completion tail as a conservative fence: the original
      // cleanup reserved an earlier sequence before publishing DELETED, so
      // waiting for this tail can only delay reclamation, never expose it
      // early.
      if (storage_owner_maintenance_completion_ring_ != nullptr) {
        const u64 next =
          storage_owner_maintenance_completion_ring_->next_sequence();
        if (next > 1) {
          maintenance_sequence = next - 1;
          item_status = protocol::MutationStatus::ok;
        } else {
          item_status = protocol::MutationStatus::already_deleted;
        }
      }
    } else if (storage_owner_maintenance_enabled(config) &&
               storage_owner_maintenance_completion_ring_ != nullptr &&
               !storage_owner_maintenance_shutdown_.load(
                 std::memory_order_acquire)) {
      StorageOwnerMaintenanceTask task;
      task.kind = StorageOwnerMaintenanceKind::cleanup_deleted_node;
      task.id = item.id;
      task.generation = item.old_generation;
      task.target = target;
      task.authority_shard = item.authority_shard;
      task.source_client = item.token.source_client;
      task.operation_item_index = item.token.item_index;
      task.operation_batch_id = item.token.client_batch_id;
      maintenance_sequence = activate_storage_owner_cleanup(
        std::move(task), config);
      activated = maintenance_sequence != 0;
      if (activated) {
        item_status = protocol::MutationStatus::ok;
      } else {
        NodeSnapshot raced_snapshot;
        if (read_node_snapshot(target, raced_snapshot) &&
            raced_snapshot.id == item.id &&
            raced_snapshot.generation == item.old_generation &&
            raced_snapshot.deleted) {
          const u64 next =
            storage_owner_maintenance_completion_ring_->next_sequence();
          lib_assert(next > 1,
                     "cleanup published DELETED without reserving a sequence");
          maintenance_sequence = next - 1;
          item_status = protocol::MutationStatus::ok;
        }
      }
    }

    output.maintenance_sequence = maintenance_sequence;
    output.status = static_cast<u32>(item_status);
    {
      std::lock_guard<std::mutex> lock(dedupe.mutex);
      const auto position = dedupe.records.find(key);
      lib_assert(position != dedupe.records.end() &&
                   position->second.in_progress,
                 "cleanup activation lost its dedupe record");
      if (item_status != protocol::MutationStatus::failed) {
        position->second.result = output;
        position->second.in_progress = false;
      } else {
        dedupe.records.erase(position);
      }
    }
    dedupe.changed.notify_all();
    success &= item_status == protocol::MutationStatus::ok;
  }
  return success;
}

bool MemoryNode::apply_local_authority_placement_items(
    span<const protocol::AuthorityPlacementItem> items,
    vec<protocol::AuthorityPlacementResult>& results) {
  results.assign(items.size(), {});
  if (items.empty()) return false;
  bool structurally_valid = true;
  for (size_t index = 0; index < items.size(); ++index) {
    const protocol::AuthorityPlacementItem& item = items[index];
    protocol::AuthorityPlacementResult& output = results[index];
    const RemotePtr expected{item.expected_raw};
    const RemotePtr desired{item.desired_raw};
    if (!authority::valid_authority_operation(item.token) ||
        (item.generation == 0 && !desired.is_null()) ||
        expected.is_null() ||
        expected.memory_node() >= num_storage_nodes_ ||
        (!desired.is_null() &&
         desired.memory_node() >= num_storage_nodes_)) {
      output.status = static_cast<u32>(
        protocol::AuthorityPlacementStatus::conflict);
      structurally_valid = false;
      continue;
    }

    u64 resulting_version = 0;
    const AuthorityRelocateState state = relocate_authority_if_current(
      item.id, item.token, item.generation, expected, desired,
      item.expected_placement_version, &resulting_version);
    output.resulting_placement_version = resulting_version;
    switch (state) {
      case AuthorityRelocateState::committed:
        output.status = static_cast<u32>(
          protocol::AuthorityPlacementStatus::committed);
        break;
      case AuthorityRelocateState::replay:
        output.status = static_cast<u32>(
          protocol::AuthorityPlacementStatus::replay);
        break;
      case AuthorityRelocateState::busy:
        output.status = static_cast<u32>(
          protocol::AuthorityPlacementStatus::busy);
        break;
      case AuthorityRelocateState::stale:
        output.status = static_cast<u32>(
          protocol::AuthorityPlacementStatus::stale);
        break;
      case AuthorityRelocateState::conflict:
        output.status = static_cast<u32>(
          protocol::AuthorityPlacementStatus::conflict);
        output.resulting_placement_version = 0;
        break;
    }
  }
  return structurally_valid;
}

bool MemoryNode::apply_local_dynamic_node_control_items(
    u32 source_shard,
    span<const protocol::DynamicNodeControlItem> items,
    vec<protocol::DynamicNodeControlResult>& results,
    const Configuration& config) {
  results.assign(items.size(), {});
  if (source_shard >= num_storage_nodes_ || items.empty()) return false;
  (void)config;

  bool structurally_valid = true;
  enum class IdentityObservation : u8 {
    live,
    deleted,
    stale,
    indeterminate,
  };
  const auto inspect_identity = [&](RemotePtr pointer, node_t id,
                                    u32 generation) {
    if (pointer.is_null() || !pointer.is_well_formed() ||
        pointer.memory_node() >= num_storage_nodes_ ||
        !VamanaNode::hot_graph_entry_available(pointer)) {
      return IdentityObservation::stale;
    }
    const auto header_address =
      vamana::StorageLayoutResolver::header(pointer);
    constexpr size_t identity_bytes =
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE;
    if (header_address.offset > mn_memory_bytes_ ||
        identity_bytes > mn_memory_bytes_ - header_address.offset) {
      return IdentityObservation::stale;
    }

    byte_t identity[identity_bytes]{};
    constexpr u32 kIdentityAttempts = 3;
    for (u32 attempt = 0; attempt < kIdentityAttempts; ++attempt) {
      u64 after = 0;
      if (local_shard(pointer.memory_node())) {
        const u64 before = load_local_node_header_acquire(pointer);
        std::memcpy(identity, &before, sizeof(before));
        std::memcpy(identity + VamanaNode::HEADER_SIZE,
                    index_buffer_.get_full_buffer() +
                      pointer.byte_offset() + VamanaNode::HEADER_SIZE,
                    VamanaNode::COMPACT_META_SIZE);
        std::atomic_thread_fence(std::memory_order_acquire);
        after = load_local_node_header_acquire(pointer);
      } else {
        remote_read_bytes(pointer.memory_node(), header_address.offset,
                          identity, sizeof(identity), 0);
        remote_read_bytes(pointer.memory_node(), header_address.offset,
                          &after, sizeof(after), 0);
      }
      u64 before = 0;
      std::memcpy(&before, identity, sizeof(before));
      if (before != after) {
        std::this_thread::yield();
        continue;
      }
      if (VamanaNode::header_incarnation(after) !=
          pointer.incarnation()) {
        return IdentityObservation::stale;
      }
      if ((after & VamanaNode::HEADER_NODE_LOCK) != 0) {
        std::this_thread::yield();
        continue;
      }
      node_t observed_id = 0;
      u32 observed_generation = 0;
      u32 observed_incarnation = 0;
      std::memcpy(&observed_id,
                  identity + VamanaNode::offset_id(),
                  sizeof(observed_id));
      std::memcpy(&observed_generation,
                  identity + VamanaNode::offset_generation(),
                  sizeof(observed_generation));
      std::memcpy(&observed_incarnation,
                  identity + VamanaNode::offset_slot_incarnation(),
                  sizeof(observed_incarnation));
      if (observed_incarnation != pointer.incarnation() ||
          observed_id != id || observed_generation != generation) {
        return IdentityObservation::stale;
      }
      return (after & VamanaNode::HEADER_DELETED) != 0
        ? IdentityObservation::deleted : IdentityObservation::live;
    }
    return IdentityObservation::indeterminate;
  };

  for (size_t index = 0; index < items.size(); ++index) {
    const protocol::DynamicNodeControlItem& item = items[index];
    protocol::DynamicNodeControlResult& output = results[index];
    const auto action = static_cast<protocol::DynamicNodeControlAction>(
      item.action);
    const RemotePtr node{item.node_raw};
    const RemotePtr allocated{item.allocated_raw};
    if (!authority::valid_authority_operation(item.token) ||
        item.generation == 0 ||
        item.authority_shard >= num_storage_nodes_ || node.is_null() ||
        !node.is_well_formed() ||
        node.memory_node() >= num_storage_nodes_ ||
        (action != protocol::DynamicNodeControlAction::allocate &&
         action != protocol::DynamicNodeControlAction::retire &&
         action !=
           protocol::DynamicNodeControlAction::settle_allocation) ||
        (action !=
           protocol::DynamicNodeControlAction::settle_allocation &&
         !allocated.is_null()) ||
        (action ==
           protocol::DynamicNodeControlAction::settle_allocation &&
         (allocated.is_null() || !allocated.is_well_formed() ||
          allocated.memory_node() != storage_id_))) {
      structurally_valid = false;
      continue;
    }

    if (action == protocol::DynamicNodeControlAction::allocate) {
      using Ledger = authority::DynamicAllocationReceiptLedger;
      const IdentityObservation source =
        inspect_identity(node, item.id, item.generation);
      const auto ledger_source_state = [](IdentityObservation observation) {
        return observation == IdentityObservation::live
          ? Ledger::SourceState::live
          : (observation == IdentityObservation::deleted ||
             observation == IdentityObservation::stale)
              ? Ledger::SourceState::terminal
              : Ledger::SourceState::indeterminate;
      };
      const Ledger::SourceState source_state =
        ledger_source_state(source);
      const Ledger::BeginResult claim =
        dynamic_allocation_receipts_.begin(item, source_state);
      switch (claim.state) {
        case Ledger::BeginState::replay:
          output = claim.result;
          break;
        case Ledger::BeginState::claimed: {
          // The first observation can precede settlement of an older receipt
          // for this semantic operation.  Settlement proves the source is
          // terminal before erasing that receipt, so a post-claim identity
          // read closes the stale-observation TOCTOU without retaining
          // unbounded replay tombstones.
          const IdentityObservation revalidated_source =
            inspect_identity(node, item.id, item.generation);
          const Ledger::ClaimValidationState validation =
            dynamic_allocation_receipts_.validate_claim_source(
              item, ledger_source_state(revalidated_source));
          if (validation ==
              Ledger::ClaimValidationState::stale_source) {
            output.node_raw = node.raw_address;
            output.status = static_cast<u32>(
              protocol::DynamicNodeControlStatus::stale);
            break;
          }
          if (validation ==
              Ledger::ClaimValidationState::indeterminate_source) {
            // The claim was canceled atomically. Leave a failed result so the
            // caller retries instead of guessing that the source is live.
            break;
          }
          if (validation !=
              Ledger::ClaimValidationState::validated) {
            structurally_valid = false;
            break;
          }

          try {
            output.node_raw = allocate_local_node().raw_address;
            output.status = static_cast<u32>(
              protocol::DynamicNodeControlStatus::ok);
          } catch (...) {
            // allocate_local_node() currently either returns a fully reserved
            // slot or terminates on capacity exhaustion. Keep the receipt
            // exception-safe if that implementation later gains a throwing
            // pre-reservation path.
            lib_assert(dynamic_allocation_receipts_.cancel_claim(item),
                       "dynamic allocation exception lost its claim");
            throw;
          }
          // Once a slot exists, losing its receipt would make safe replay and
          // reclamation impossible. publish() cannot legitimately fail: a
          // settlement observes an unready claim as pending and never erases
          // it.
          lib_assert(dynamic_allocation_receipts_.publish(item, output),
                     "dynamic allocation lost its claimed receipt");
          break;
        }
        case Ledger::BeginState::stale_source:
          output.node_raw = node.raw_address;
          output.status = static_cast<u32>(
            protocol::DynamicNodeControlStatus::stale);
          break;
        case Ledger::BeginState::conflict:
          structurally_valid = false;
          break;
        case Ledger::BeginState::pending:
        case Ledger::BeginState::indeterminate_source:
        case Ledger::BeginState::pressure:
          // The caller retries the same semantic token. Never guess that an
          // in-flight reservation has expired and allocate a second slot.
          break;
      }
      continue;
    }

    if (action ==
        protocol::DynamicNodeControlAction::settle_allocation) {
      using Ledger = authority::DynamicAllocationReceiptLedger;
      const IdentityObservation source =
        inspect_identity(node, item.id, item.generation);
      const IdentityObservation destination =
        inspect_identity(allocated, item.id, item.generation);
      const bool source_terminal =
        source == IdentityObservation::deleted ||
        source == IdentityObservation::stale;
      const bool destination_terminal =
        destination == IdentityObservation::live ||
        destination == IdentityObservation::deleted ||
        destination == IdentityObservation::stale;
      const Ledger::SettleState settled =
        dynamic_allocation_receipts_.settle(
          item, source_terminal, destination_terminal);
      output.node_raw = allocated.raw_address;
      if (settled == Ledger::SettleState::settled ||
          settled == Ledger::SettleState::replay) {
        output.status = static_cast<u32>(
          protocol::DynamicNodeControlStatus::ok);
      } else if (settled == Ledger::SettleState::conflict) {
        structurally_valid = false;
      }
      continue;
    }

    if (node.memory_node() != storage_id_ ||
        node.byte_offset() < gpu_dynamic_node_base_ ||
        !valid_local_storage_node_pointer(node) ||
        storage_owner_maintenance_completion_ring_ == nullptr) {
      structurally_valid = false;
      continue;
    }
    NodeSnapshot snapshot;
    if (!read_node_snapshot(node, snapshot) || snapshot.id != item.id ||
        snapshot.generation != item.generation) {
      // The desired postcondition already holds for this logical record. A
      // delayed retirement must not tombstone or enqueue its address after it
      // has been reused by a different generation.
      output.node_raw = node.raw_address;
      output.status = static_cast<u32>(
        protocol::DynamicNodeControlStatus::stale);
      continue;
    }

    if ((snapshot.header & VamanaNode::HEADER_CENTROID_ACCOUNTED) != 0) {
      // Retirement is not a membership operation. Its caller must first
      // withdraw and publish this exact tagged identity, otherwise a route
      // snapshot could advertise the tombstone until a later retry.
      output.node_raw = node.raw_address;
      output.status = static_cast<u32>(
        protocol::DynamicNodeControlStatus::failed);
      continue;
    }

    const u64 sequence = begin_storage_owner_maintenance_sequence(1);
    if (!snapshot.deleted) {
      (void)mark_node_deleted(node, item.generation);
    }
    retire_local_dynamic_node(node, sequence);
    complete_storage_owner_maintenance_sequence(sequence);
    output.node_raw = node.raw_address;
    output.maintenance_sequence = sequence;
    output.status = static_cast<u32>(
      protocol::DynamicNodeControlStatus::ok);
  }
  return structurally_valid;
}

bool MemoryNode::handle_peer_cleanup_activate_request(
    u32 source_shard,
    const protocol::PeerRpcHeader& header,
    const protocol::CleanupActivateItem* items,
    const Configuration& config) {
  const u32 item_count = header.item_count;
  vec<protocol::CleanupActivateResult> results;
  const bool processed = items != nullptr && item_count != 0 &&
    activate_local_cleanup_items(
      source_shard,
      span<const protocol::CleanupActivateItem>{items, item_count},
      results, config);
  const size_t bytes = protocol::cleanup_activate_response_bytes(item_count);
  vec<byte_t> response(bytes, 0);
  auto* output_header =
    reinterpret_cast<protocol::PeerRpcHeader*>(response.data());
  output_header->magic = protocol::kPeerRpcMagic;
  output_header->version = protocol::kPeerRpcVersion;
  output_header->type = static_cast<u32>(
    protocol::PeerRpcType::cleanup_activate_response);
  output_header->source_shard = storage_id_;
  output_header->item_count = item_count;
  output_header->request_id = header.request_id;
  // Per-item statuses carry terminal/retryable semantics. A structurally
  // valid envelope remains consumable even when one item is stale.
  output_header->status = static_cast<u32>(protocol::InsertStatus::ok);
  if (results.size() == item_count) {
    std::memcpy(protocol::cleanup_activate_results(response.data()),
                results.data(), results.size() * sizeof(results[0]));
  }
  send_peer_rpc_message(source_shard, response.data(), response.size());
  return processed;
}

bool MemoryNode::handle_peer_authority_placement_request(
    u32 source_shard,
    const protocol::PeerRpcHeader& header,
    const protocol::AuthorityPlacementItem* items,
    const Configuration&) {
  const u32 item_count = header.item_count;
  vec<protocol::AuthorityPlacementResult> results;
  const bool processed = source_shard < num_storage_nodes_ &&
    items != nullptr && item_count != 0 &&
    apply_local_authority_placement_items(
      span<const protocol::AuthorityPlacementItem>{items, item_count},
      results);
  const size_t bytes = protocol::authority_placement_response_bytes(
    item_count);
  vec<byte_t> response(bytes, 0);
  auto* output_header =
    reinterpret_cast<protocol::PeerRpcHeader*>(response.data());
  output_header->magic = protocol::kPeerRpcMagic;
  output_header->version = protocol::kPeerRpcVersion;
  output_header->type = static_cast<u32>(
    protocol::PeerRpcType::authority_placement_response);
  output_header->source_shard = storage_id_;
  output_header->item_count = item_count;
  output_header->request_id = header.request_id;
  output_header->status = static_cast<u32>(protocol::InsertStatus::ok);
  if (results.size() == item_count) {
    std::memcpy(protocol::authority_placement_results(response.data()),
                results.data(), results.size() * sizeof(results[0]));
  }
  send_peer_rpc_message(source_shard, response.data(), response.size());
  return processed;
}

bool MemoryNode::handle_peer_dynamic_node_control_request(
    u32 source_shard,
    const protocol::PeerRpcHeader& header,
    const protocol::DynamicNodeControlItem* items,
    const Configuration& config) {
  const u32 item_count = header.item_count;
  vec<protocol::DynamicNodeControlResult> results;
  const bool processed = items != nullptr && item_count != 0 &&
    apply_local_dynamic_node_control_items(
      source_shard,
      span<const protocol::DynamicNodeControlItem>{items, item_count},
      results, config);
  const size_t bytes = protocol::dynamic_node_control_response_bytes(
    item_count);
  vec<byte_t> response(bytes, 0);
  auto* output_header = reinterpret_cast<protocol::PeerRpcHeader*>(
    response.data());
  output_header->magic = protocol::kPeerRpcMagic;
  output_header->version = protocol::kPeerRpcVersion;
  output_header->type = static_cast<u32>(
    protocol::PeerRpcType::dynamic_node_control_response);
  output_header->source_shard = storage_id_;
  output_header->item_count = item_count;
  output_header->request_id = header.request_id;
  output_header->status = static_cast<u32>(protocol::InsertStatus::ok);
  if (results.size() == item_count) {
    std::memcpy(protocol::dynamic_node_control_results(response.data()),
                results.data(), results.size() * sizeof(results[0]));
  }
  send_peer_rpc_message(source_shard, response.data(), response.size());
  return processed;
}

bool MemoryNode::post_peer_control_request_attempt(
    u32 target_shard,
    protocol::PeerRpcType request_type,
    protocol::PeerRpcType response_type,
    u64 request_id,
    u32 item_count,
    const void* items,
    size_t item_bytes,
    size_t request_bytes,
    const Configuration& config) {
  if (target_shard >= num_storage_nodes_ || target_shard == storage_id_ ||
      request_id == 0 || item_count == 0 || items == nullptr ||
      request_bytes < sizeof(protocol::PeerRpcHeader) ||
      request_bytes > peer_rpc_runtime_.message_bytes ||
      item_bytes != request_bytes - sizeof(protocol::PeerRpcHeader) ||
      peer_async_responses_ == nullptr) {
    return false;
  }
  const auto registration = peer_async_responses_->register_send_attempt(
    request_id, target_shard, response_type, item_count);
  if (registration ==
      memory_node_detail::PeerResponseRegistration::already_complete) {
    return true;
  }
  if (registration != memory_node_detail::PeerResponseRegistration::registered &&
      registration != memory_node_detail::PeerResponseRegistration::retry) {
    return false;
  }

  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  u32 slot_id = 0;
  while (!try_acquire_peer_rpc_send_slot(
           target_shard, PeerRpcSendClass::control, slot_id)) {
    if (peer_reverse_shutdown_.load(std::memory_order_acquire) ||
        std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::unique_lock<std::mutex> lock(peer_completion_mutex_);
    peer_completion_cv_.wait_for(lock, std::chrono::microseconds(100));
  }

  const size_t offset = peer_rpc_async_send_offset(target_shard, slot_id);
  byte_t* message = peer_rpc_runtime_.buffer.get_full_buffer() + offset;
  std::memset(message, 0, request_bytes);
  auto* header = reinterpret_cast<protocol::PeerRpcHeader*>(message);
  header->magic = protocol::kPeerRpcMagic;
  header->version = protocol::kPeerRpcVersion;
  header->type = static_cast<u32>(request_type);
  header->source_shard = storage_id_;
  header->item_count = item_count;
  header->request_id = request_id;
  std::memcpy(message + sizeof(*header), items, item_bytes);
  post_peer_rpc_send_slot(target_shard, slot_id, request_bytes);
  return true;
}

MemoryNode::TryPeerResponse MemoryNode::wait_peer_control_response(
    u64 request_id,
    u32 target_shard,
    protocol::PeerRpcType response_type,
    u32 item_count,
    protocol::PeerRpcHeader& header,
    vec<byte_t>& payload,
    PeerResponseLease& lease,
    const Configuration& config) {
  lease = {};
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(config.storage_owner_rpc_timeout_ms);
  for (;;) {
    const TryPeerResponse state = try_consume_peer_rpc_response(
      request_id, target_shard, response_type, item_count,
      header, payload, lease);
    if (state != TryPeerResponse::pending) return state;
    if (std::chrono::steady_clock::now() >= deadline) return state;
    std::unique_lock<std::mutex> lock(peer_completion_mutex_);
    peer_completion_cv_.wait_for(lock, std::chrono::microseconds(100));
  }
}

bool MemoryNode::activate_cleanup_fanout_and_wait(
    span<const protocol::CleanupActivateItem> items,
    vec<protocol::CleanupActivateResult>& results,
    const Configuration& config) {
  results.assign(items.size(), {});
  if (items.empty()) return true;

  struct IndexedItem {
    protocol::CleanupActivateItem item{};
    size_t input_index{};
  };
  struct Pending {
    u32 target_shard{};
    u64 request_id{};
    vec<protocol::CleanupActivateItem> items;
    vec<size_t> input_indices;
    bool posted{};
  };
  std::map<u32, vec<IndexedItem>> grouped;
  bool success = true;
  for (size_t index = 0; index < items.size(); ++index) {
    const RemotePtr target{items[index].old_raw};
    const auto action = static_cast<protocol::CleanupActivateAction>(
      items[index].action);
    if (target.is_null() || target.memory_node() >= num_storage_nodes_ ||
        items[index].authority_shard != storage_id_ ||
        (action != protocol::CleanupActivateAction::activate &&
         action != protocol::CleanupActivateAction::release)) {
      results[index].target_raw = items[index].old_raw;
      results[index].token = items[index].token;
      success = false;
      continue;
    }
    grouped[target.memory_node()].push_back({items[index], index});
  }

  const size_t payload_capacity = peer_rpc_runtime_.message_bytes -
    sizeof(protocol::PeerRpcHeader);
  const u32 wire_capacity = static_cast<u32>(std::min<size_t>(
    config.storage_owner_batch_max,
    payload_capacity / sizeof(protocol::CleanupActivateItem)));
  lib_assert(wire_capacity != 0,
             "peer control slot cannot hold one cleanup activation");
  vec<Pending> pending;

  // Post every remote chunk before performing local work or waiting for ACKs.
  for (const auto& [target_shard, shard_items] : grouped) {
    if (target_shard == storage_id_) continue;
    for (size_t begin = 0; begin < shard_items.size();
         begin += wire_capacity) {
      const size_t count = std::min<size_t>(
        wire_capacity, shard_items.size() - begin);
      Pending request;
      request.target_shard = target_shard;
      request.request_id = allocate_peer_request_id();
      request.items.reserve(count);
      request.input_indices.reserve(count);
      for (size_t offset = 0; offset < count; ++offset) {
        request.items.push_back(shard_items[begin + offset].item);
        request.input_indices.push_back(
          shard_items[begin + offset].input_index);
      }
      const u32 item_count = static_cast<u32>(request.items.size());
      request.posted = post_peer_control_request_attempt(
        target_shard, protocol::PeerRpcType::cleanup_activate_request,
        protocol::PeerRpcType::cleanup_activate_response,
        request.request_id, item_count, request.items.data(),
        request.items.size() * sizeof(request.items[0]),
        protocol::cleanup_activate_request_bytes(item_count), config);
      pending.push_back(std::move(request));
    }
  }

  const auto local = grouped.find(storage_id_);
  if (local != grouped.end()) {
    vec<protocol::CleanupActivateItem> local_items;
    local_items.reserve(local->second.size());
    for (const IndexedItem& indexed : local->second) {
      local_items.push_back(indexed.item);
    }
    vec<protocol::CleanupActivateResult> local_results;
    (void)activate_local_cleanup_items(
      storage_id_, span<const protocol::CleanupActivateItem>{local_items},
      local_results, config);
    for (size_t index = 0; index < local_results.size(); ++index) {
      results[local->second[index].input_index] = local_results[index];
      success &= local_results[index].status != static_cast<u32>(
        protocol::MutationStatus::failed);
    }
  }

  constexpr u32 kTransportAttempts = 3;
  for (Pending& request : pending) {
    bool complete = false;
    for (u32 attempt = 0; attempt < kTransportAttempts && !complete;
         ++attempt) {
      if (!request.posted) {
        request.posted = post_peer_control_request_attempt(
          request.target_shard,
          protocol::PeerRpcType::cleanup_activate_request,
          protocol::PeerRpcType::cleanup_activate_response,
          request.request_id, static_cast<u32>(request.items.size()),
          request.items.data(),
          request.items.size() * sizeof(request.items[0]),
          protocol::cleanup_activate_request_bytes(
            static_cast<u32>(request.items.size())), config);
        if (!request.posted) continue;
      }
      protocol::PeerRpcHeader header;
      vec<byte_t> payload;
      PeerResponseLease response_lease{};
      const TryPeerResponse state = wait_peer_control_response(
        request.request_id, request.target_shard,
        protocol::PeerRpcType::cleanup_activate_response,
        static_cast<u32>(request.items.size()), header, payload,
        response_lease, config);
      const size_t expected_bytes = protocol::cleanup_activate_response_bytes(
        static_cast<u32>(request.items.size()));
      if (state == TryPeerResponse::success &&
          payload.size() == expected_bytes) {
        const auto* wire_results =
          protocol::cleanup_activate_results(payload.data());
        bool retry_failed_item = false;
        bool valid = true;
        for (size_t index = 0; index < request.items.size(); ++index) {
          const auto& input = request.items[index];
          const auto& output = wire_results[index];
          valid &= same_token(input.token, output.token) &&
            output.target_raw == input.old_raw && output.reserved == 0;
          if (output.status ==
              static_cast<u32>(protocol::MutationStatus::ok)) {
            const auto action = static_cast<
              protocol::CleanupActivateAction>(input.action);
            valid &= action == protocol::CleanupActivateAction::release ||
              output.maintenance_sequence != 0;
          } else if (output.status ==
                     static_cast<u32>(protocol::MutationStatus::failed)) {
            retry_failed_item = true;
          } else {
            valid &= output.status == static_cast<u32>(
                       protocol::MutationStatus::not_found) ||
              output.status == static_cast<u32>(
                protocol::MutationStatus::already_deleted);
          }
        }
        if (valid && !retry_failed_item) {
          for (size_t index = 0; index < request.items.size(); ++index) {
            results[request.input_indices[index]] = wire_results[index];
          }
          complete = acknowledge_peer_rpc_response(response_lease);
          if (complete) break;
        }
      }
      request.posted = false;
      if (response_lease.valid()) {
        (void)rearm_peer_rpc_response(response_lease);
      }
    }
    if (!complete) {
      success = false;
      cancel_peer_rpc_response(request.request_id);
    }
  }
  return success;
}

bool MemoryNode::relocate_via_authority(
    u32 authority_shard,
    const protocol::AuthorityPlacementItem& item,
    protocol::AuthorityPlacementResult& result,
    const Configuration& config) {
  result = {};
  if (authority_shard >= num_storage_nodes_) return false;
  if (authority_shard == storage_id_) {
    vec<protocol::AuthorityPlacementResult> local_results;
    const bool valid = apply_local_authority_placement_items(
      span<const protocol::AuthorityPlacementItem>{&item, 1},
      local_results);
    if (!local_results.empty()) result = local_results.front();
    return valid;
  }

  const u64 request_id = allocate_peer_request_id();
  constexpr u32 kTransportAttempts = 3;
  bool posted = false;
  for (u32 attempt = 0; attempt < kTransportAttempts; ++attempt) {
    if (!posted) {
      posted = post_peer_control_request_attempt(
        authority_shard,
        protocol::PeerRpcType::authority_placement_request,
        protocol::PeerRpcType::authority_placement_response,
        request_id, 1, &item, sizeof(item),
        protocol::authority_placement_request_bytes(1), config);
      if (!posted) continue;
    }
    protocol::PeerRpcHeader header;
    vec<byte_t> payload;
    PeerResponseLease response_lease{};
    const TryPeerResponse state = wait_peer_control_response(
      request_id, authority_shard,
      protocol::PeerRpcType::authority_placement_response,
      1, header, payload, response_lease, config);
    if (state == TryPeerResponse::success &&
        payload.size() == protocol::authority_placement_response_bytes(1)) {
      const protocol::AuthorityPlacementResult candidate =
        protocol::authority_placement_results(payload.data())[0];
      if (candidate.reserved == 0 &&
          candidate.status <= static_cast<u32>(
            protocol::AuthorityPlacementStatus::conflict)) {
        result = candidate;
        if (acknowledge_peer_rpc_response(response_lease)) return true;
      }
    }
    posted = false;
    if (response_lease.valid()) {
      (void)rearm_peer_rpc_response(response_lease);
    }
  }
  cancel_peer_rpc_response(request_id);
  return false;
}

bool MemoryNode::control_dynamic_node_on_shard(
    u32 physical_shard,
    const protocol::DynamicNodeControlItem& item,
    protocol::DynamicNodeControlResult& result,
    const Configuration& config) {
  result = {};
  if (physical_shard >= num_storage_nodes_) return false;
  if (physical_shard == storage_id_) {
    vec<protocol::DynamicNodeControlResult> local_results;
    const bool valid = apply_local_dynamic_node_control_items(
      storage_id_, span<const protocol::DynamicNodeControlItem>{&item, 1},
      local_results, config);
    if (!local_results.empty()) result = local_results.front();
    return valid;
  }

  const u64 request_id = allocate_peer_request_id();
  constexpr u32 kTransportAttempts = 3;
  bool posted = false;
  for (u32 attempt = 0; attempt < kTransportAttempts; ++attempt) {
    if (!posted) {
      posted = post_peer_control_request_attempt(
        physical_shard,
        protocol::PeerRpcType::dynamic_node_control_request,
        protocol::PeerRpcType::dynamic_node_control_response,
        request_id, 1, &item, sizeof(item),
        protocol::dynamic_node_control_request_bytes(1), config);
      if (!posted) continue;
    }
    protocol::PeerRpcHeader header;
    vec<byte_t> payload;
    PeerResponseLease response_lease{};
    const TryPeerResponse state = wait_peer_control_response(
      request_id, physical_shard,
      protocol::PeerRpcType::dynamic_node_control_response,
      1, header, payload, response_lease, config);
    if (state == TryPeerResponse::success &&
        payload.size() ==
          protocol::dynamic_node_control_response_bytes(1)) {
      const protocol::DynamicNodeControlResult candidate =
        protocol::dynamic_node_control_results(payload.data())[0];
      if (candidate.reserved == 0 &&
          candidate.status <= static_cast<u32>(
            protocol::DynamicNodeControlStatus::failed)) {
        result = candidate;
        if (acknowledge_peer_rpc_response(response_lease)) return true;
      }
    }
    posted = false;
    if (response_lease.valid()) {
      (void)rearm_peer_rpc_response(response_lease);
    }
  }
  cancel_peer_rpc_response(request_id);
  return false;
}

bool MemoryNode::enqueue_peer_physical_control_task(
    PeerPhysicalControlTask&& task) {
  const auto type = static_cast<protocol::PeerRpcType>(task.header.type);
  if (type == protocol::PeerRpcType::cleanup_activate_request) {
    std::lock_guard<std::mutex> lock(peer_cleanup_control_tasks_mutex_);
    if (peer_reverse_shutdown_.load(std::memory_order_acquire) ||
        peer_cleanup_control_tasks_.size() >=
          peer_physical_control_task_queue_limit_ ||
        task.source_shard >= peer_cleanup_next_source_sequences_.size() ||
        peer_cleanup_next_source_sequences_[task.source_shard] ==
          std::numeric_limits<u64>::max()) {
      return false;
    }
    task.source_sequence =
      ++peer_cleanup_next_source_sequences_[task.source_shard];
    peer_cleanup_control_tasks_.push_back(std::move(task));
    peer_cleanup_control_tasks_cv_.notify_one();
    return true;
  }
  if (type == protocol::PeerRpcType::authority_placement_request ||
      type == protocol::PeerRpcType::dynamic_node_control_request) {
    std::lock_guard<std::mutex> lock(peer_placement_control_tasks_mutex_);
    if (peer_reverse_shutdown_.load(std::memory_order_acquire) ||
        peer_placement_control_tasks_.size() >=
          peer_physical_control_task_queue_limit_) {
      return false;
    }
    peer_placement_control_tasks_.push_back(std::move(task));
    peer_placement_control_tasks_cv_.notify_one();
    return true;
  }
  return false;
}

void MemoryNode::peer_cleanup_control_worker_loop() {
  const Configuration& config = *storage_worker_config_;
  for (;;) {
    PeerPhysicalControlTask task;
    {
      std::unique_lock<std::mutex> lock(peer_cleanup_control_tasks_mutex_);
      peer_cleanup_control_tasks_cv_.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
          !peer_cleanup_control_tasks_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) &&
          peer_cleanup_control_tasks_.empty()) {
        return;
      }
      task = std::move(peer_cleanup_control_tasks_.front());
      peer_cleanup_control_tasks_.pop_front();
    }
    peer_cleanup_control_tasks_cv_.notify_one();
    const auto* items = protocol::cleanup_activate_items(
      task.payload.data());
    bool release_barrier = false;
    for (u32 item = 0; item < task.header.item_count; ++item) {
      release_barrier |= items[item].action == static_cast<u32>(
        protocol::CleanupActivateAction::release);
    }
    lib_assert(task.source_shard < peer_cleanup_completion_states_.size() &&
                 peer_cleanup_completion_states_[task.source_shard] !=
                   nullptr &&
                 task.source_sequence != 0,
               "peer cleanup task omitted its RC receive-order sequence");
    PeerOrderedCompletionState& completion =
      *peer_cleanup_completion_states_[task.source_shard];
    if (release_barrier) {
      std::unique_lock<std::mutex> lock(completion.mutex);
      // A release is allowed to erase a semantic receipt only after all
      // earlier requests from this RC source have completed, even though
      // unrelated sources/tokens execute concurrently on other workers.
      completion.changed.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
          completion.completed_prefix + 1 == task.source_sequence;
      });
    }
    (void)handle_peer_cleanup_activate_request(
      task.source_shard, task.header,
      items, config);
    lib_assert(peer_request_deduplicator_->abandon(
                 task.dedup_lease, task.source_shard, task.header),
               "cleanup-control completion lost its dedup lease");
    {
      std::lock_guard<std::mutex> lock(completion.mutex);
      if (task.source_sequence == completion.completed_prefix + 1) {
        ++completion.completed_prefix;
        while (completion.completed_out_of_order.erase(
                 completion.completed_prefix + 1) != 0) {
          ++completion.completed_prefix;
        }
      } else {
        lib_assert(task.source_sequence > completion.completed_prefix + 1,
                   "peer cleanup task completed its sequence twice");
        const bool inserted = completion.completed_out_of_order.insert(
          task.source_sequence).second;
        lib_assert(inserted,
                   "peer cleanup out-of-order sequence completed twice");
      }
    }
    completion.changed.notify_all();
  }
}

void MemoryNode::peer_placement_control_worker_loop() {
  const Configuration& config = *storage_worker_config_;
  for (;;) {
    PeerPhysicalControlTask task;
    {
      std::unique_lock<std::mutex> lock(
        peer_placement_control_tasks_mutex_);
      peer_placement_control_tasks_cv_.wait(lock, [&]() {
        return peer_reverse_shutdown_.load(std::memory_order_acquire) ||
          !peer_placement_control_tasks_.empty();
      });
      if (peer_reverse_shutdown_.load(std::memory_order_acquire) &&
          peer_placement_control_tasks_.empty()) {
        return;
      }
      task = std::move(peer_placement_control_tasks_.front());
      peer_placement_control_tasks_.pop_front();
    }
    peer_placement_control_tasks_cv_.notify_one();
    const auto type = static_cast<protocol::PeerRpcType>(task.header.type);
    if (type == protocol::PeerRpcType::authority_placement_request) {
      (void)handle_peer_authority_placement_request(
        task.source_shard, task.header,
        protocol::authority_placement_items(task.payload.data()), config);
    } else {
      lib_assert(type == protocol::PeerRpcType::dynamic_node_control_request,
                 "placement control queue received an invalid request");
      (void)handle_peer_dynamic_node_control_request(
        task.source_shard, task.header,
        protocol::dynamic_node_control_items(task.payload.data()), config);
    }
    lib_assert(peer_request_deduplicator_->abandon(
                 task.dedup_lease, task.source_shard, task.header),
               "placement-control completion lost its dedup lease");
  }
}
