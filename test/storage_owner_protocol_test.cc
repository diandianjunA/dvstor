#include <cassert>
#include <vector>

#include "service/storage_owner_protocol.hh"

namespace {

void test_authority_extension_rpc_layouts() {
  namespace protocol = service::storage_owner;

  static_assert(protocol::kMutationProtocolVersion == 3);
  static_assert(protocol::kPeerRpcVersion == 13);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::stage1_arm_response) == 14);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::cleanup_activate_request) == 15);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::cleanup_activate_response) == 16);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::authority_placement_request) == 17);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::authority_placement_response) == 18);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::dynamic_node_control_request) == 19);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::dynamic_node_control_response) == 20);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::stage2_expand_score_request) == 21);
  static_assert(static_cast<u32>(
                  protocol::PeerRpcType::stage2_expand_score_response) == 22);
  static_assert(sizeof(protocol::Stage2ExpandScoreItem) == 24);
  static_assert(sizeof(protocol::Stage2ExpandScoreResult) == 32);
  static_assert(sizeof(protocol::Stage2ExpandScoreNeighbor) == 16);
  static_assert(static_cast<u32>(
                  protocol::DynamicNodeControlAction::settle_allocation) == 3);
  static_assert(sizeof(protocol::DynamicNodeControlItem) == 48);
  static_assert(sizeof(protocol::ReverseUpdateOp) == 32);
  static_assert(sizeof(protocol::Stage1ExecuteItem) == 48);
  static_assert(sizeof(protocol::Stage1ExecuteResult) == 40);
  static_assert(offsetof(protocol::Stage1ExecuteItem,
                         initial_placement_version) == 16);
  static_assert(offsetof(protocol::Stage1ExecuteResult,
                         maintenance_sequence) == 16);
  static_assert(static_cast<u32>(protocol::Stage1ArmAction::arm) == 1);
  static_assert(static_cast<u32>(protocol::Stage1ArmAction::abort) == 2);
  static_assert(static_cast<u32>(protocol::Stage1ArmAction::release) == 3);
  static_assert(static_cast<u32>(
                  protocol::CleanupActivateAction::activate) == 1);
  static_assert(static_cast<u32>(
                  protocol::CleanupActivateAction::release) == 2);
  static_assert(static_cast<u32>(
                  protocol::AuthorityPlacementStatus::committed) == 0);
  static_assert(static_cast<u32>(
                  protocol::AuthorityPlacementStatus::replay) == 1);
  static_assert(static_cast<u32>(
                  protocol::AuthorityPlacementStatus::busy) == 2);
  static_assert(static_cast<u32>(
                  protocol::AuthorityPlacementStatus::stale) == 3);
  static_assert(static_cast<u32>(
                  protocol::AuthorityPlacementStatus::conflict) == 4);

  constexpr u32 item_count = 2;

  std::vector<byte_t> stage1_arm(
    protocol::stage1_arm_request_bytes(item_count), 0);
  auto* arm_header =
    reinterpret_cast<protocol::PeerRpcHeader*>(stage1_arm.data());
  arm_header->type = static_cast<u32>(
    protocol::PeerRpcType::stage1_arm_request);
  arm_header->item_count = item_count;
  auto* arm_items = protocol::stage1_arm_items(stage1_arm.data());
  arm_items[0] = {
    .token = {.source_client = 7, .item_index = 3,
              .client_batch_id = 101},
    .target_raw = 0x111,
    .initial_placement_version = 11,
    .id = 13,
    .generation = 2,
    .action = static_cast<u32>(protocol::Stage1ArmAction::arm),
  };
  arm_items[1] = {
    .token = {.source_client = 7, .item_index = 4,
              .client_batch_id = 102},
    .target_raw = 0x222,
    .initial_placement_version = 0,
    .id = 14,
    .generation = 3,
    .action = static_cast<u32>(protocol::Stage1ArmAction::abort),
  };
  const auto* const_arm_items = protocol::stage1_arm_items(
    static_cast<const void*>(stage1_arm.data()));
  assert(const_arm_items[0].initial_placement_version == 11);
  assert(const_arm_items[0].token.client_batch_id == 101);
  assert(const_arm_items[0].target_raw == 0x111);
  assert(const_arm_items[0].id == 13);
  assert(const_arm_items[0].generation == 2);
  assert(const_arm_items[0].action ==
         static_cast<u32>(protocol::Stage1ArmAction::arm));
  assert(const_arm_items[1].initial_placement_version == 0);
  assert(const_arm_items[1].action ==
         static_cast<u32>(protocol::Stage1ArmAction::abort));
  assert(reinterpret_cast<const byte_t*>(const_arm_items + item_count) ==
         stage1_arm.data() + stage1_arm.size());

  std::vector<byte_t> stage1_arm_response(
    protocol::stage1_arm_response_bytes(item_count), 0);
  auto* arm_results = protocol::stage1_arm_results(
    stage1_arm_response.data());
  arm_results[0] = {
    .token = arm_items[0].token,
    .target_raw = arm_items[0].target_raw,
    .maintenance_sequence = 41,
    .status = static_cast<u32>(protocol::MutationStatus::ok),
  };
  const auto* const_arm_results = protocol::stage1_arm_results(
    static_cast<const void*>(stage1_arm_response.data()));
  assert(const_arm_results[0].token.client_batch_id == 101);
  assert(const_arm_results[0].target_raw == 0x111);
  assert(const_arm_results[0].maintenance_sequence == 41);
  assert(const_arm_results[0].status ==
         static_cast<u32>(protocol::MutationStatus::ok));
  assert(reinterpret_cast<const byte_t*>(const_arm_results + item_count) ==
         stage1_arm_response.data() + stage1_arm_response.size());

  std::vector<byte_t> cleanup_request(
    protocol::cleanup_activate_request_bytes(item_count), 0);
  auto* cleanup_items = protocol::cleanup_activate_items(
    cleanup_request.data());
  cleanup_items[0] = {
    .token = {.source_client = 7, .item_index = 3,
              .client_batch_id = 101},
    .old_raw = 0x1000,
    .id = 41,
    .old_generation = 8,
    .authority_shard = 2,
    .action = static_cast<u32>(
      protocol::CleanupActivateAction::activate),
  };
  cleanup_items[1] = {
    .token = {.source_client = 8, .item_index = 4,
              .client_batch_id = 202},
    .old_raw = 0x2000,
    .id = 42,
    .old_generation = 9,
    .authority_shard = 3,
    .action = static_cast<u32>(
      protocol::CleanupActivateAction::release),
  };
  const auto* const_cleanup_items = protocol::cleanup_activate_items(
    static_cast<const void*>(cleanup_request.data()));
  assert(const_cleanup_items[0].token.source_client == 7);
  assert(const_cleanup_items[0].token.item_index == 3);
  assert(const_cleanup_items[0].token.client_batch_id == 101);
  assert(const_cleanup_items[0].old_raw == 0x1000);
  assert(const_cleanup_items[0].id == 41);
  assert(const_cleanup_items[0].old_generation == 8);
  assert(const_cleanup_items[0].authority_shard == 2);
  assert(const_cleanup_items[0].action == static_cast<u32>(
    protocol::CleanupActivateAction::activate));
  assert(const_cleanup_items[1].action == static_cast<u32>(
    protocol::CleanupActivateAction::release));
  assert(reinterpret_cast<const byte_t*>(const_cleanup_items + item_count) ==
         cleanup_request.data() + cleanup_request.size());

  std::vector<byte_t> cleanup_response(
    protocol::cleanup_activate_response_bytes(item_count), 0);
  auto* cleanup_results = protocol::cleanup_activate_results(
    cleanup_response.data());
  cleanup_results[0] = {
    .target_raw = 0x3000,
    .maintenance_sequence = 901,
    .token = {.source_client = 7, .item_index = 3,
              .client_batch_id = 101},
    .status = static_cast<u32>(protocol::MutationStatus::ok),
  };
  const auto* const_cleanup_results = protocol::cleanup_activate_results(
    static_cast<const void*>(cleanup_response.data()));
  assert(const_cleanup_results[0].target_raw == 0x3000);
  assert(const_cleanup_results[0].maintenance_sequence == 901);
  assert(const_cleanup_results[0].token.source_client == 7);
  assert(const_cleanup_results[0].token.item_index == 3);
  assert(const_cleanup_results[0].token.client_batch_id == 101);
  assert(const_cleanup_results[0].status ==
         static_cast<u32>(protocol::MutationStatus::ok));
  assert(reinterpret_cast<const byte_t*>(const_cleanup_results + item_count) ==
         cleanup_response.data() + cleanup_response.size());

  std::vector<byte_t> placement_request(
    protocol::authority_placement_request_bytes(item_count), 0);
  auto* placement_items = protocol::authority_placement_items(
    placement_request.data());
  placement_items[0] = {
    .token = {.source_client = 9, .item_index = 5,
              .client_batch_id = 303},
    .id = 77,
    .generation = 10,
    .expected_raw = 0x4000,
    .desired_raw = 0x5000,
    .expected_placement_version = 13,
  };
  const auto* const_placement_items = protocol::authority_placement_items(
    static_cast<const void*>(placement_request.data()));
  assert(const_placement_items[0].token.source_client == 9);
  assert(const_placement_items[0].token.item_index == 5);
  assert(const_placement_items[0].token.client_batch_id == 303);
  assert(const_placement_items[0].id == 77);
  assert(const_placement_items[0].generation == 10);
  assert(const_placement_items[0].expected_raw == 0x4000);
  assert(const_placement_items[0].desired_raw == 0x5000);
  assert(const_placement_items[0].expected_placement_version == 13);
  assert(reinterpret_cast<const byte_t*>(const_placement_items + item_count) ==
         placement_request.data() + placement_request.size());

  std::vector<byte_t> placement_response(
    protocol::authority_placement_response_bytes(item_count), 0);
  auto* placement_results = protocol::authority_placement_results(
    placement_response.data());
  placement_results[0] = {
    .resulting_placement_version = 14,
    .status = static_cast<u32>(
      protocol::AuthorityPlacementStatus::committed),
  };
  placement_results[1] = {
    .resulting_placement_version = 14,
    .status = static_cast<u32>(
      protocol::AuthorityPlacementStatus::replay),
  };
  const auto* const_placement_results =
    protocol::authority_placement_results(
      static_cast<const void*>(placement_response.data()));
  assert(const_placement_results[0].resulting_placement_version == 14);
  assert(const_placement_results[0].status == static_cast<u32>(
    protocol::AuthorityPlacementStatus::committed));
  assert(const_placement_results[1].resulting_placement_version == 14);
  assert(const_placement_results[1].status == static_cast<u32>(
    protocol::AuthorityPlacementStatus::replay));
  assert(reinterpret_cast<const byte_t*>(
           const_placement_results + item_count) ==
         placement_response.data() + placement_response.size());

  std::vector<byte_t> dynamic_request(
    protocol::dynamic_node_control_request_bytes(item_count), 0);
  auto* dynamic_items = protocol::dynamic_node_control_items(
    dynamic_request.data());
  dynamic_items[0] = {
    .token = {.source_client = 9, .item_index = 5,
              .client_batch_id = 303},
    .node_raw = 0x6000,
    .allocated_raw = 0,
    .id = 77,
    .generation = 10,
    .authority_shard = 2,
    .action = static_cast<u32>(
      protocol::DynamicNodeControlAction::allocate),
  };
  dynamic_items[1] = {
    .token = {.source_client = 9, .item_index = 5,
              .client_batch_id = 303},
    .node_raw = 0x7000,
    .allocated_raw = 0,
    .id = 77,
    .generation = 10,
    .authority_shard = 2,
    .action = static_cast<u32>(
      protocol::DynamicNodeControlAction::retire),
  };
  const auto* const_dynamic_items =
    protocol::dynamic_node_control_items(
      static_cast<const void*>(dynamic_request.data()));
  assert(const_dynamic_items[0].node_raw == 0x6000);
  assert(const_dynamic_items[0].action == static_cast<u32>(
    protocol::DynamicNodeControlAction::allocate));
  assert(const_dynamic_items[1].node_raw == 0x7000);
  assert(const_dynamic_items[1].action == static_cast<u32>(
    protocol::DynamicNodeControlAction::retire));
  dynamic_items[1].allocated_raw = 0x8000;
  dynamic_items[1].action = static_cast<u32>(
    protocol::DynamicNodeControlAction::settle_allocation);
  assert(const_dynamic_items[1].allocated_raw == 0x8000);
  assert(const_dynamic_items[1].action == static_cast<u32>(
    protocol::DynamicNodeControlAction::settle_allocation));
  assert(reinterpret_cast<const byte_t*>(
           const_dynamic_items + item_count) ==
         dynamic_request.data() + dynamic_request.size());

  std::vector<byte_t> dynamic_response(
    protocol::dynamic_node_control_response_bytes(item_count), 0);
  auto* dynamic_results = protocol::dynamic_node_control_results(
    dynamic_response.data());
  dynamic_results[0] = {
    .node_raw = 0x8000,
    .status = static_cast<u32>(
      protocol::DynamicNodeControlStatus::ok),
  };
  dynamic_results[1] = {
    .node_raw = 0x7000,
    .maintenance_sequence = 44,
    .status = static_cast<u32>(
      protocol::DynamicNodeControlStatus::stale),
  };
  const auto* const_dynamic_results =
    protocol::dynamic_node_control_results(
      static_cast<const void*>(dynamic_response.data()));
  assert(const_dynamic_results[0].node_raw == 0x8000);
  assert(const_dynamic_results[1].maintenance_sequence == 44);
  assert(reinterpret_cast<const byte_t*>(
           const_dynamic_results + item_count) ==
         dynamic_response.data() + dynamic_response.size());
}

}  // namespace

int main() {
  VamanaNode::init_static_storage(128, 96, VectorDType::uint8);
  test_authority_extension_rpc_layouts();
  return 0;
}
