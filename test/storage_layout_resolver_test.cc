#include <iostream>

#include "vamana/storage_layout_resolver.hh"

namespace {

bool check_format(vamana::StorageFormat format, u32 dim, u32 R, VectorDType dtype) {
  VamanaNode::disable_rabitq();
  VamanaNode::disable_hot_graph();
  VamanaNode::set_storage_format(format);
  VamanaNode::init_static_storage(dim, R, dtype);

  const RemotePtr ptr{0, 4096};
  if (vamana::StorageLayoutResolver::header(ptr).offset != ptr.byte_offset()) return false;
  if (vamana::StorageLayoutResolver::id(ptr).offset != ptr.byte_offset() + 8) return false;
  if (vamana::StorageLayoutResolver::vector(ptr).offset !=
      ptr.byte_offset() + VamanaNode::offset_vector()) return false;

  if (format == vamana::StorageFormat::aos_v1) {
    const auto read = vamana::StorageLayoutResolver::neighbor_read(ptr);
    return !read.compact &&
      read.address.offset == ptr.byte_offset() + VamanaNode::offset_id() &&
      VamanaNode::total_size() >= VamanaNode::offset_neighbors() + VamanaNode::NEIGHBORS_SIZE;
  }

  const u32 entry_bytes = static_cast<u32>(VamanaNode::hot_graph_entry_size());
  const u64 graph_offset = 1ull << 20;
  const u64 dynamic_offset = 2ull << 20;
  VamanaNode::configure_hot_graph({graph_offset}, {8}, entry_bytes, 0, 2,
                                  {dynamic_offset},
                                  static_cast<u32>(VamanaNode::dynamic_record_size()),
                                  static_cast<u32>(VamanaNode::total_size()));
  const RemotePtr base_ptr{0, vamana::hot_graph::kNodeBaseOffset + 3 * VamanaNode::total_size()};
  const auto read = vamana::StorageLayoutResolver::neighbor_read(base_ptr);
  if (!read.compact || read.address.offset != graph_offset + 3 * entry_bytes) return false;
  if (VamanaNode::offset_vector() !=
      VamanaNode::HEADER_SIZE + VamanaNode::COMPACT_META_SIZE) return false;
  if (VamanaNode::offset_vector() != 16) return false;
  if (VamanaNode::total_size() % VamanaNode::COMPACT_ALIGNMENT != 0) return false;
  if (VamanaNode::total_size() >=
      VamanaNode::align_storage(VamanaNode::NODE_PREFIX_SIZE + VamanaNode::NEIGHBORS_SIZE) +
        VamanaNode::vector_storage_bytes()) return false;

  vec<RemotePtr> neighbors{RemotePtr{0, 1024}, RemotePtr{0, 2048}};
  vec<byte_t> compact(entry_bytes);
  vec<byte_t> decoded(VamanaNode::neighbor_read_size());
  VamanaNode::encode_hot_graph_entry(compact.data(), 1, 2, neighbors.data(), 2, 0, 7, 2);
  if (!VamanaNode::decode_hot_graph_entry(compact.data(), decoded.data())) return false;
  const auto* slots = reinterpret_cast<const RemotePtr*>(
    decoded.data() + VamanaNode::neighbor_payload_offset_in_read());
  if (slots[0] != neighbors[0] || slots[1] != neighbors[1]) return false;
  compact.back() ^= 1;
  return !VamanaNode::decode_hot_graph_entry(compact.data(), decoded.data());
}

}  // namespace

int main() {
  bool ok = true;
  ok = ok && !vamana::parse_storage_format("hybrid_split_v1").has_value();
  ok = ok && !vamana::parse_storage_format("hybrid_dart_delta_v3").has_value();
  ok = ok && !vamana::parse_storage_format("compact_neighbors_delta_v3").has_value();
  for (VectorDType dtype : {VectorDType::float32, VectorDType::uint8, VectorDType::int8}) {
    for (u32 dim : {16u, 128u, 300u}) {
      ok = ok && check_format(vamana::StorageFormat::aos_v1, dim, 48, dtype);
      ok = ok && check_format(vamana::StorageFormat::compact_v1, dim, 48, dtype);
    }
  }
  if (!ok) {
    std::cerr << "storage layout resolver test failed\n";
    return 1;
  }
  return 0;
}
