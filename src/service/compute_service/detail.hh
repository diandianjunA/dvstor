#pragma once

#include "service/compute_service.hh"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

#include "common/index_path.hh"
#include "gpu_search/index_format.hh"
#include "service/storage_owner_client_helpers.hh"
#include "vamana/idmap.hh"

namespace compute_service_detail {

inline constexpr u32 kRpcMagic = 0x53484e57;
inline constexpr u32 kRpcVersion = 1;
inline constexpr u32 kInitialRpcRecvsPerPeer = 8;
inline constexpr u32 kMaxRpcResults = 512;

using service::storage_owner_client::add_storage_owner_breakdown;
using service::storage_owner_client::add_storage_owner_sender_breakdown;
using service::storage_owner_client::duration_ns;
using service::storage_owner_client::duration_ns_clamped;
using service::storage_owner_client::per_item_ns;
using service::storage_owner_client::storage_owner_wr_id;
using service::storage_owner_client::storage_owner_completion_wr_id;
using service::storage_owner_client::storage_owner_is_completion_wr;
using service::storage_owner_client::storage_owner_wr_owner;

}  // namespace compute_service_detail
