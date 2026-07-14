#pragma once

#include "memory_node/memory_node.hh"

#include <algorithm>
#include <iostream>
#include <limits>

namespace memory_node_storage_owner_runtime_detail {

inline bool storage_owner_local_stitch_mode(
    const configuration::IndexConfiguration& config) {
  return config.storage_owner_update_mode == "local_stitch";
}

inline bool storage_owner_batch_is_local_stage1(
    const configuration::IndexConfiguration& config) {
  return storage_owner_local_stitch_mode(config);
}

}  // namespace memory_node_storage_owner_runtime_detail
