#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gpu_search/index_format.hh"

int main() {
  namespace format = gpu_search::format;
  format::View view;
  view.layout.dim = 16;
  view.layout.graph_degree = 3;
  view.layout.vector_dtype = static_cast<u32>(VectorDType::uint8);
  view.layout.pq_subquantizers = 16;
  view.layout.pq_bits = 8;
  view.layout.code_bytes = 16;
  view.layout.model_checksum = 0x12345678ULL;
  view.layout.num_shards = 2;
  view.layout.graph_entry_bytes = 24;
  view.layout.graph_pointer_bytes = 5;
  view.layout.graph_shard_bits = 1;
  view.layout.medoid_ordinal = 2;
  view.layout.num_nodes = 4;
  view.shards = {
    {.ordinal_base = 0, .node_count = 2, .node_base_offset = 16,
     .node_stride = 64, .graph_base_offset = 4096,
     .dynamic_base_offset = 8192, .code_remote_offset = 8192,
     .code_bytes = 32, .memory_node = 0, .dynamic_record_bytes = 96,
     .dynamic_hot_offset = 64},
    {.ordinal_base = 2, .node_count = 2, .node_base_offset = 16,
     .node_stride = 64, .graph_base_offset = 4096,
     .dynamic_base_offset = 8192, .code_remote_offset = 8192,
     .code_bytes = 32, .memory_node = 1, .dynamic_record_bytes = 96,
     .dynamic_hot_offset = 64},
  };
  view.entry_points = {2, 0, 3};

  std::string error;
  assert(format::validate_view(view, &error));

  RemotePtr pointer;
  assert(format::ordinal_to_remote(view, 0, pointer));
  assert(pointer == RemotePtr(0, 16));
  assert(format::ordinal_to_remote(view, 3, pointer));
  assert(pointer == RemotePtr(1, 80));
  u32 ordinal = 0;
  assert(format::remote_to_ordinal(view, RemotePtr(1, 16), ordinal));
  assert(ordinal == 2);
  assert(!format::remote_to_ordinal(view, RemotePtr(1, 8192), ordinal));

  format::View malformed = view;
  malformed.shards[1].ordinal_base = 3;
  assert(!format::validate_view(malformed, &error));
  malformed = view;
  malformed.shards[0].code_remote_offset += 64;
  assert(!format::validate_view(malformed, &error));

  const auto code_path =
    std::filesystem::temp_directory_path() / "dvstor-pq16-codes.bin";
  std::vector<byte_t> payload(32, 0x5a);
  format::CodeHeader code_header;
  code_header.memory_node = 0;
  code_header.code_bytes = 16;
  code_header.node_size = 64;
  code_header.entry_count = 2;
  code_header.remote_offset = 8192;
  code_header.payload_bytes = payload.size();
  code_header.model_checksum = view.layout.model_checksum;
  code_header.payload_checksum = format::checksum64(payload.data(), payload.size());
  {
    std::ofstream output(code_path, std::ios::binary | std::ios::trunc);
    format::CodeHeader placeholder;
    output.write(reinterpret_cast<const char*>(&placeholder), sizeof(placeholder));
    output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    assert(format::write_code_header(output, code_header, &error));
  }
  format::CodeHeader loaded_code;
  assert(format::read_code_header(code_path, loaded_code, &error));
  assert(loaded_code.payload_checksum == code_header.payload_checksum);

  {
    std::fstream codes(code_path, std::ios::binary | std::ios::in | std::ios::out);
    codes.seekg(static_cast<std::streamoff>(sizeof(format::CodeHeader)));
    char byte = 0;
    codes.read(&byte, 1);
    byte ^= 1;
    codes.seekp(static_cast<std::streamoff>(sizeof(format::CodeHeader)));
    codes.write(&byte, 1);
    assert(codes.good());
  }
  format::CodeHeader unchanged_header;
  assert(format::read_code_header(code_path, unchanged_header, &error));
  assert(unchanged_header.payload_checksum == code_header.payload_checksum);

  std::filesystem::remove(code_path);
  return 0;
}
