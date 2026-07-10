#include "tools/vamana_offline/gpu_tiered_writer.hh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>

#include <library/utils.hh>

#include "common/index_path.hh"
#include "remote_pointer.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_offline {
namespace {

struct EntrySample {
  u64 priority{};
  u32 id{};

  bool operator<(const EntrySample& other) const { return priority < other.priority; }
};

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

class ShardPageWriter {
public:
  ShardPageWriter(const filepath_t& path, u32 shard, u32 page_bytes, u64 remote_offset)
      : path_(path), shard_(shard), page_bytes_(page_bytes), remote_offset_(remote_offset),
        page_(page_bytes, 0), cursor_(sizeof(gpu_search::format::PageHeader)) {
    output_.open(path_, std::ios::binary | std::ios::trunc);
    lib_assert(output_.good(), "failed to create GPU graph pages: " + path_.string());
    gpu_search::format::ShardPageFileHeader header;
    header.page_bytes = page_bytes_;
    header.memory_node = shard_;
    header.remote_offset = remote_offset_;
    output_.write(reinterpret_cast<const char*>(&header), sizeof(header));
  }

  void append(u32 node_id, const vec<u32>& neighbors,
              gpu_search::format::IdEncoding encoding,
              gpu_search::format::NodeRecord& node_record) {
    const size_t id_bytes = static_cast<size_t>(encoding);
    const size_t degree = std::min<size_t>(neighbors.size(), std::numeric_limits<u16>::max());
    const size_t record_bytes = sizeof(gpu_search::format::PageNodeHeader) + degree * id_bytes;
    const size_t padded_record_bytes = gpu_search::format::align_up(
      record_bytes, alignof(gpu_search::format::PageNodeHeader));
    lib_assert(padded_record_bytes + sizeof(gpu_search::format::PageHeader) <= page_bytes_,
               "one graph adjacency record does not fit in a GPU graph page");
    if (cursor_ + padded_record_bytes > page_bytes_ ||
        page_node_count_ == std::numeric_limits<u16>::max()) {
      flush_page();
    }

    node_record.cold_record_offset = static_cast<u32>(cursor_);
    node_record.cold_page_offset = remote_offset_ + static_cast<u64>(page_index_) * page_bytes_;
    gpu_search::format::PageNodeHeader node_header;
    node_header.node_id = node_id;
    node_header.degree = static_cast<u16>(degree);
    node_header.flags = 0;
    std::memcpy(page_.data() + cursor_, &node_header, sizeof(node_header));
    byte_t* encoded = page_.data() + cursor_ + sizeof(node_header);
    for (size_t i = 0; i < degree; ++i) {
      gpu_search::format::encode_id(encoded + i * id_bytes, neighbors[i], encoding);
    }
    cursor_ += padded_record_bytes;
    ++page_node_count_;
  }

  u64 finish() {
    if (page_node_count_ > 0) flush_page();
    gpu_search::format::ShardPageFileHeader header;
    header.page_bytes = page_bytes_;
    header.memory_node = shard_;
    header.remote_offset = remote_offset_;
    header.data_bytes = static_cast<u64>(page_index_) * page_bytes_;
    header.checksum = gpu_search::format::checksum64(
      reinterpret_cast<const byte_t*>(&header), offsetof(gpu_search::format::ShardPageFileHeader, checksum));
    output_.seekp(0);
    output_.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output_.flush();
    lib_assert(output_.good(), "failed to flush GPU graph pages: " + path_.string());
    return header.data_bytes;
  }

private:
  void flush_page() {
    gpu_search::format::PageHeader header;
    header.node_count = page_node_count_;
    header.payload_bytes = static_cast<u32>(cursor_ - sizeof(header));
    std::memcpy(page_.data(), &header, sizeof(header));
    output_.seekp(static_cast<std::streamoff>(
      sizeof(gpu_search::format::ShardPageFileHeader) + static_cast<u64>(page_index_) * page_bytes_));
    output_.write(reinterpret_cast<const char*>(page_.data()), page_.size());
    lib_assert(output_.good(), "failed to write GPU graph page: " + path_.string());
    ++page_index_;
    std::fill(page_.begin(), page_.end(), 0);
    cursor_ = sizeof(gpu_search::format::PageHeader);
    page_node_count_ = 0;
  }

  filepath_t path_;
  u32 shard_{};
  u32 page_bytes_{};
  u64 remote_offset_{};
  std::ofstream output_;
  vec<byte_t> page_;
  size_t cursor_{};
  u16 page_node_count_{};
  u32 page_index_{};
};

}  // namespace

GpuTieredWriteResult write_gpu_tiered_index(
    const VamanaGraph& graph,
    const Dataset& dataset,
    const VamanaBuildConfig& config,
    const vec<NodePlacement>& placements,
    const vec<u64>& shard_file_bytes,
    const filepath_t& output_prefix) {
  lib_assert(config.build_gpu_tiered_index, "GPU tiered writer called while disabled");
  lib_assert(config.use_rabitq, "GPU tiered index requires RaBitQ entries");
  lib_assert(graph.num_nodes == dataset.size() && placements.size() == dataset.size(),
             "GPU tiered writer input cardinality mismatch");
  lib_assert(shard_file_bytes.size() == config.num_memory_nodes,
             "GPU tiered writer shard cardinality mismatch");

  namespace format = gpu_search::format;
  format::View view;
  view.header.page_bytes = config.gpu_graph_page_bytes;
  view.header.dim = dataset.dim;
  view.header.graph_degree = config.R;
  view.header.hot_degree = config.gpu_hot_degree;
  view.header.vector_dtype = static_cast<u32>(dataset.dtype);
  view.header.rabitq_code_bits = VamanaNode::rabitq_code_bits();
  view.header.rabitq_entry_bytes = format::rabitq_entry_bytes(view.header.rabitq_code_bits);
  lib_assert(view.header.rabitq_entry_bytes == VamanaNode::rabitq_entry_size() &&
             format::rabitq_code_storage_bytes(view.header.rabitq_code_bits) ==
               VamanaNode::rabitq_code_storage_size(),
             "GPU tiered RaBitQ layout diverges from the storage-node layout");
  view.header.id_encoding_bytes = dataset.size() <= 0x00ffffffULL ? 3 : 4;
  view.header.num_shards = config.num_memory_nodes;
  view.header.medoid_id = dataset.id(graph.medoid);
  view.header.num_nodes = dataset.size();
  view.header.base_generation = 1;
  view.nodes.resize(dataset.size());
  view.hot_neighbors.reserve(dataset.size() * std::min<u32>(config.gpu_hot_degree, config.R));
  view.rabitq_entries.resize(dataset.size() * VamanaNode::rabitq_entry_size(), 0);
  view.shards.resize(config.num_memory_nodes);
  view.centroid = VamanaNode::rabitq_centroid;

  const auto encoding = static_cast<format::IdEncoding>(view.header.id_encoding_bytes);
  vec<std::unique_ptr<ShardPageWriter>> page_writers;
  page_writers.reserve(config.num_memory_nodes);
  GpuTieredWriteResult result;
  result.index_file = index_path::gpu_tiered_file(output_prefix);
  result.graph_page_offsets.resize(config.num_memory_nodes);
  result.graph_page_bytes.resize(config.num_memory_nodes);
  result.hot_degree = config.gpu_hot_degree;
  result.page_bytes = config.gpu_graph_page_bytes;
  const u32 target_entry_points = static_cast<u32>(
    std::min<size_t>(config.gpu_entry_points, dataset.size()));
  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    const u64 remote_offset = format::align_up(shard_file_bytes[shard], config.gpu_graph_page_bytes);
    result.graph_page_offsets[shard] = remote_offset;
    page_writers.push_back(std::make_unique<ShardPageWriter>(
      index_path::gpu_graph_pages_file(output_prefix, shard + 1, config.num_memory_nodes),
      shard, config.gpu_graph_page_bytes, remote_offset));
    view.shards[shard].graph_pages_offset = remote_offset;
    view.shards[shard].vector_region_offset = 16 + VamanaNode::offset_vector();
    view.shards[shard].vector_stride = (VamanaNode::total_size() + 7) & ~7ULL;
    view.shards[shard].memory_node = shard;
  }

  const u32 entry_quota = (target_entry_points + config.num_memory_nodes - 1) /
    config.num_memory_nodes;
  vec<std::priority_queue<EntrySample>> entry_samples(config.num_memory_nodes);
  vec<u32> neighbors;
  for (size_t i = 0; i < dataset.size(); ++i) {
    lib_assert(dataset.id(i) == i,
               "GPU tiered index requires dense IDs in [0, num_vectors)");
    graph.copy_neighbors(i, neighbors);
    auto& record = view.nodes[i];
    const NodePlacement& placement = placements[i];
    auto& samples = entry_samples[placement.memory_node];
    const EntrySample sample{
      .priority = mix64(static_cast<u64>(dataset.id(i)) ^
                        (static_cast<u64>(static_cast<u32>(config.seed)) << 32)),
      .id = dataset.id(i),
    };
    if (samples.size() < entry_quota) {
      samples.push(sample);
    } else if (sample.priority < samples.top().priority) {
      samples.pop();
      samples.push(sample);
    }
    record.remote_node = RemotePtr{placement.memory_node, placement.offset}.raw_address;
    record.generation = 1;
    record.shard = static_cast<u16>(placement.memory_node);
    record.hot_neighbor_begin = static_cast<u32>(view.hot_neighbors.size());
    record.hot_neighbor_count = static_cast<u16>(
      std::min<size_t>(neighbors.size(), config.gpu_hot_degree));
    for (u32 j = 0; j < record.hot_neighbor_count; ++j) {
      view.hot_neighbors.push_back(dataset.id(neighbors[j]));
    }
    page_writers[placement.memory_node]->append(dataset.id(i), neighbors, encoding, record);
    ++view.shards[placement.memory_node].node_count;

    VamanaNode::RabitqCode code;
    f32 norm = 0.0f;
    f32 error = 0.0f;
    VamanaNode::compute_rabitq_entry(dataset.raw_vector(i), dataset.dtype, code, norm, error);
    byte_t* entry = view.rabitq_entries.data() + i * VamanaNode::rabitq_entry_size();
    std::memcpy(entry, code.data(), code.size());
    std::memcpy(entry + format::rabitq_norm_offset(view.header.rabitq_code_bits),
                &norm, sizeof(norm));
    std::memcpy(entry + format::rabitq_error_offset(view.header.rabitq_code_bits),
                &error, sizeof(error));
  }

  vec<EntrySample> selected_entries;
  for (auto& samples : entry_samples) {
    while (!samples.empty()) {
      selected_entries.push_back(samples.top());
      samples.pop();
    }
  }
  std::sort(selected_entries.begin(), selected_entries.end(),
            [](const EntrySample& lhs, const EntrySample& rhs) {
              return lhs.priority < rhs.priority;
            });
  if (selected_entries.size() > target_entry_points) {
    selected_entries.resize(target_entry_points);
  }
  view.entry_points.reserve(target_entry_points);
  view.entry_points.push_back(view.header.medoid_id);
  for (const EntrySample& sample : selected_entries) {
    if (sample.id != view.header.medoid_id &&
        view.entry_points.size() < target_entry_points) {
      view.entry_points.push_back(sample.id);
    }
  }
  result.entry_points = static_cast<u32>(view.entry_points.size());

  for (u32 shard = 0; shard < config.num_memory_nodes; ++shard) {
    result.graph_page_bytes[shard] = page_writers[shard]->finish();
    view.shards[shard].graph_pages_bytes = result.graph_page_bytes[shard];
  }
  str error;
  lib_assert(format::write_file(result.index_file, view, &error), error);
  return result;
}

}  // namespace tools::vamana_offline
