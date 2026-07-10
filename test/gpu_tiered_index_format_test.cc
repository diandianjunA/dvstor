#include <cassert>
#include <filesystem>
#include <string>

#include "gpu_search/index_format.hh"

int main() {
  namespace format = gpu_search::format;
  format::View view;
  view.header.dim = 4;
  view.header.graph_degree = 4;
  view.header.hot_degree = 2;
  view.header.vector_dtype = static_cast<u32>(VectorDType::uint8);
  view.header.rabitq_code_bits = 8;
  view.header.rabitq_entry_bytes = format::rabitq_entry_bytes(8);
  view.header.id_encoding_bytes = 3;
  view.header.num_shards = 1;
  view.header.num_nodes = 2;
  view.nodes = {
    {.remote_node = 16, .cold_page_offset = 4096, .cold_record_offset = 16,
     .generation = 1, .hot_neighbor_begin = 0, .hot_neighbor_count = 1,
     .shard = 0, .flags = 0},
    {.remote_node = 32, .cold_page_offset = 4096, .cold_record_offset = 32,
     .generation = 1, .hot_neighbor_begin = 1, .hot_neighbor_count = 1,
     .shard = 0, .flags = 0},
  };
  view.hot_neighbors = {1, 0};
  view.rabitq_entries.resize(2 * view.header.rabitq_entry_bytes, 0);
  view.rabitq_entries[0] = 0xaa;
  view.rabitq_entries[view.header.rabitq_entry_bytes] = 0x55;
  view.shards = {{.graph_pages_offset = 4096, .graph_pages_bytes = 4096,
                  .vector_region_offset = 16, .vector_stride = 64,
                  .node_count = 2, .memory_node = 0}};
  view.centroid = {1.0f, 2.0f, 3.0f, 4.0f};
  view.entry_points = {0, 1};

  std::string error;
  assert(format::validate_view(view, &error));
  const auto path = std::filesystem::temp_directory_path() / "dvstor_gpu_tiered_test.bin";
  assert(format::write_file(path, view, &error));

  format::View loaded;
  assert(format::read_file(path, loaded, &error));
  assert(loaded.header.num_nodes == 2);
  assert(loaded.hot_neighbors == view.hot_neighbors);
  assert(loaded.rabitq_entries == view.rabitq_entries);
  assert(loaded.centroid == view.centroid);
  assert(loaded.entry_points == view.entry_points);

  format::View malformed = view;
  malformed.header.rabitq_entry_bytes -= sizeof(f32);
  assert(!format::validate_view(malformed, &error));

  byte_t encoded[4]{};
  format::encode_id(encoded, 0x00a1b2c3u, format::IdEncoding::u24);
  assert(format::decode_id(encoded, format::IdEncoding::u24) == 0x00a1b2c3u);
  format::encode_id(encoded, 0xf1a1b2c3u, format::IdEncoding::u32);
  assert(format::decode_id(encoded, format::IdEncoding::u32) == 0xf1a1b2c3u);
  std::filesystem::remove(path);
  return 0;
}
