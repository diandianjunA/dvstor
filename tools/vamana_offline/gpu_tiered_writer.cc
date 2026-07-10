#include "tools/vamana_offline/gpu_tiered_writer.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <queue>

#include <library/utils.hh>

#include "common/index_path.hh"
#include "vamana/hot_graph.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_offline {
namespace {

struct EntrySample {
  u64 priority{};
  u32 ordinal{};
  bool operator<(const EntrySample& other) const { return priority < other.priority; }
};

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

}  // namespace

GpuTieredWriteResult write_gpu_tiered_index(
    const VamanaGraph& graph,
    const Dataset& dataset,
    const VamanaBuildConfig& config,
    const vec<NodePlacement>& placements,
    const vec<u64>& shard_file_bytes,
    const filepath_t& output_prefix) {
  lib_assert(config.build_gpu_tiered_index, "GPU tiered writer called while disabled");
  lib_assert(config.use_rabitq, "GPU V4 requires RaBitQ entries");
  lib_assert(VamanaNode::compact_storage() && VamanaNode::HAS_HOT_GRAPH &&
               VamanaNode::HOT_GRAPH_FORMAT_VERSION >= 2,
             "GPU V4 requires the compact V2 graph plane");
  lib_assert(graph.num_nodes == dataset.size() && placements.size() == dataset.size(),
             "GPU V4 writer input cardinality mismatch");
  lib_assert(shard_file_bytes.size() == config.num_memory_nodes,
             "GPU V4 writer shard cardinality mismatch");
  lib_assert(dataset.size() > 0 && dataset.size() < (1ull << 30),
             "GPU V4 supports 1..2^30-1 base nodes");
  lib_assert(VamanaNode::hot_graph_entry_size() <= gpu_search::format::kGraphCacheLineBytes,
             "compact graph entry exceeds the GPU V4 cache line");

  vec<u64> counts(config.num_memory_nodes, 0);
  for (const NodePlacement& placement : placements) ++counts[placement.memory_node];
  vec<u64> ordinal_bases(config.num_memory_nodes, 0);
  for (u32 shard = 1; shard < config.num_memory_nodes; ++shard) {
    ordinal_bases[shard] = ordinal_bases[shard - 1] + counts[shard - 1];
  }

  gpu_search::format::View manifest;
  manifest.header.dim = dataset.dim;
  manifest.header.graph_degree = config.R;
  manifest.header.vector_dtype = static_cast<u32>(dataset.dtype);
  manifest.header.rabitq_code_bits = VamanaNode::rabitq_code_bits();
  manifest.header.rabitq_entry_bytes = static_cast<u32>(VamanaNode::rabitq_entry_size());
  manifest.header.num_shards = config.num_memory_nodes;
  manifest.header.graph_entry_bytes = static_cast<u32>(VamanaNode::hot_graph_entry_size());
  manifest.header.graph_pointer_bytes = vamana::hot_graph::kCompactPointerBytes;
  manifest.header.graph_shard_bits = VamanaNode::HOT_GRAPH_SHARD_BITS;
  manifest.header.num_nodes = dataset.size();
  manifest.header.base_generation = 1;
  manifest.centroid = VamanaNode::rabitq_centroid;
  manifest.shards.resize(config.num_memory_nodes);

  const NodePlacement& medoid = placements[graph.medoid];
  const u64 medoid_slot = (medoid.offset - gpu_search::format::kNodeBaseOffset) /
    VamanaNode::total_size();
  manifest.header.medoid_ordinal = static_cast<u32>(ordinal_bases[medoid.memory_node] + medoid_slot);

  GpuTieredWriteResult result;
  result.index_file = index_path::gpu_tiered_file(output_prefix);
  result.code_files.resize(config.num_memory_nodes);
  result.code_remote_offsets.resize(config.num_memory_nodes);
  result.code_bytes.resize(config.num_memory_nodes);
  vec<std::ofstream> outputs(config.num_memory_nodes);
  vec<gpu_search::format::CodeHeader> headers(config.num_memory_nodes);
  vec<u64> checksums(config.num_memory_nodes, gpu_search::format::checksum64_initial());
  vec<u64> written(config.num_memory_nodes, 0);
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    result.code_files[shard] = index_path::gpu_code_file(
      output_prefix, shard + 1, config.num_memory_nodes);
    outputs[shard].open(result.code_files[shard], std::ios::binary | std::ios::trunc);
    lib_assert(outputs[shard].good(), "failed to create GPU V4 code sidecar");
    gpu_search::format::CodeHeader placeholder;
    outputs[shard].write(reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));
    headers[shard].memory_node = shard;
    headers[shard].code_bits = VamanaNode::rabitq_code_bits();
    headers[shard].entry_bytes = static_cast<u32>(VamanaNode::rabitq_entry_size());
    headers[shard].node_size = static_cast<u32>(VamanaNode::total_size());
    headers[shard].entry_count = counts[shard];
    lib_assert(shard_file_bytes[shard] >= VamanaNode::HOT_GRAPH_DYNAMIC_BASE_OFFSETS[shard],
               "GPU V4 shard image ends before its dynamic base");
    headers[shard].remote_offset = gpu_search::format::align_up(
      VamanaNode::HOT_GRAPH_DYNAMIC_BASE_OFFSETS[shard], 64);
    headers[shard].payload_bytes = counts[shard] * headers[shard].entry_bytes;
    result.code_remote_offsets[shard] = headers[shard].remote_offset;
    result.code_bytes[shard] = headers[shard].payload_bytes;
    manifest.shards[shard] = {
      .ordinal_base = ordinal_bases[shard],
      .node_count = counts[shard],
      .node_base_offset = gpu_search::format::kNodeBaseOffset,
      .node_stride = VamanaNode::total_size(),
      .graph_base_offset = VamanaNode::HOT_GRAPH_ENTRY_OFFSETS[shard],
      .dynamic_base_offset = VamanaNode::HOT_GRAPH_DYNAMIC_BASE_OFFSETS[shard],
      .code_remote_offset = headers[shard].remote_offset,
      .code_bytes = headers[shard].payload_bytes,
      .memory_node = shard,
      .dynamic_record_bytes = VamanaNode::HOT_GRAPH_DYNAMIC_RECORD_BYTES,
      .dynamic_hot_offset = VamanaNode::HOT_GRAPH_DYNAMIC_HOT_OFFSET,
    };
  }

  const u32 target_entries = static_cast<u32>(
    std::min<size_t>(config.gpu_entry_points, dataset.size()));
  const u32 quota = (target_entries + config.num_memory_nodes - 1) /
    config.num_memory_nodes;
  vec<std::priority_queue<EntrySample>> entry_samples(config.num_memory_nodes);
  vec<byte_t> entry(VamanaNode::rabitq_entry_size(), 0);
  for (size_t index = 0; index < dataset.size(); ++index) {
    const NodePlacement& placement = placements[index];
    const u64 slot = (placement.offset - gpu_search::format::kNodeBaseOffset) /
      VamanaNode::total_size();
    lib_assert(slot == written[placement.memory_node],
               "GPU V4 code stream requires monotonically assigned shard slots");
    ++written[placement.memory_node];
    VamanaNode::RabitqCode code;
    f32 norm = 0.0f;
    f32 error = 0.0f;
    VamanaNode::compute_rabitq_entry(dataset.raw_vector(index), dataset.dtype,
                                     code, norm, error);
    std::fill(entry.begin(), entry.end(), 0);
    std::memcpy(entry.data(), code.data(), code.size());
    std::memcpy(entry.data() + gpu_search::format::rabitq_norm_offset(
                  VamanaNode::rabitq_code_bits()), &norm, sizeof(norm));
    std::memcpy(entry.data() + gpu_search::format::rabitq_error_offset(
                  VamanaNode::rabitq_code_bits()), &error, sizeof(error));
    outputs[placement.memory_node].write(
      reinterpret_cast<const char*>(entry.data()), entry.size());
    checksums[placement.memory_node] = gpu_search::format::checksum64_update(
      checksums[placement.memory_node], entry.data(), entry.size());

    const u32 ordinal = static_cast<u32>(ordinal_bases[placement.memory_node] + slot);
    EntrySample sample{
      mix64(static_cast<u64>(ordinal) ^ static_cast<u32>(config.seed)), ordinal};
    auto& heap = entry_samples[placement.memory_node];
    if (heap.size() < quota) heap.push(sample);
    else if (sample.priority < heap.top().priority) {
      heap.pop();
      heap.push(sample);
    }
  }

  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    lib_assert(written[shard] == counts[shard], "GPU V4 code stream count mismatch");
    headers[shard].payload_checksum = checksums[shard];
    str error;
    lib_assert(gpu_search::format::write_code_header(outputs[shard], headers[shard], &error),
               error);
    outputs[shard].flush();
    lib_assert(outputs[shard].good(), "failed to flush GPU V4 code sidecar");
  }

  vec<EntrySample> selected;
  for (auto& heap : entry_samples) {
    while (!heap.empty()) {
      selected.push_back(heap.top());
      heap.pop();
    }
  }
  std::sort(selected.begin(), selected.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.priority < rhs.priority;
  });
  manifest.entry_points.push_back(manifest.header.medoid_ordinal);
  for (const EntrySample& sample : selected) {
    if (manifest.entry_points.size() >= target_entries) break;
    if (sample.ordinal != manifest.header.medoid_ordinal) {
      manifest.entry_points.push_back(sample.ordinal);
    }
  }
  result.entry_points = static_cast<u32>(manifest.entry_points.size());
  str error;
  lib_assert(gpu_search::format::write_file(result.index_file, manifest, &error), error);
  return result;
}

}  // namespace tools::vamana_offline
