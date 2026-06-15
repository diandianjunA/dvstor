#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "service/storage_owner_protocol.hh"
#include "vamana/anchor_index.hh"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void write_test_sidecar(const filepath_t& prefix) {
  std::ofstream output(prefix.string() + ".anchors", std::ios::binary | std::ios::trunc);
  require(output.good(), "failed to create test sidecar");

  vamana::anchor::Header header;
  header.dim = 3;
  header.shard_count = 2;
  header.vector_dtype = static_cast<u32>(VectorDType::int8);
  header.vector_bytes = 3;
  header.anchors_per_shard = 2;
  header.total_anchors = 4;
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));

  const std::array<std::array<float, 3>, 2> centroids{{{0.5f, 0.5f, 0.5f},
                                                       {100.5f, 100.5f, 100.5f}}};
  const std::array<std::array<std::array<i8, 3>, 2>, 2> vectors{{
    {{{0, 0, 0}, {1, 1, 1}}},
    {{{100, 100, 100}, {101, 101, 101}}}
  }};
  for (u32 shard = 0; shard < 2; ++shard) {
    const vamana::anchor::ShardHeader shard_header{shard, 2};
    output.write(reinterpret_cast<const char*>(&shard_header), sizeof(shard_header));
    output.write(reinterpret_cast<const char*>(centroids[shard].data()),
                 sizeof(centroids[shard]));
    for (u32 i = 0; i < 2; ++i) {
      vamana::anchor::EntryHeader entry;
      entry.rptr_raw = RemotePtr{shard, 16 + static_cast<u64>(i) * 64}.raw_address;
      entry.id = shard * 2 + i;
      output.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
      output.write(reinterpret_cast<const char*>(vectors[shard][i].data()), 3);
    }
  }
  require(output.good(), "failed to write test sidecar");
}

void test_protocol_alignment() {
  VamanaNode::init_static_storage(7, 4, VectorDType::uint8);
  constexpr u32 item_count = 3;
  constexpr u32 hint_count = 4;
  vec<byte_t> request(service::storage_owner::insert_batch_request_bytes(
    item_count, 7, hint_count));
  auto* header = reinterpret_cast<service::storage_owner::InsertBatchRequestHeader*>(request.data());
  header->item_count = item_count;
  header->anchor_hint_count = hint_count;
  const u64* hints = service::storage_owner::request_anchor_hints(request.data(), item_count);
  require(reinterpret_cast<std::uintptr_t>(hints) % alignof(u64) == 0,
          "insert anchor hints are not aligned");
  require(reinterpret_cast<const byte_t*>(hints + item_count * hint_count) <=
            request.data() + request.size(),
          "insert anchor hints exceed request size");

  vec<byte_t> exact(service::storage_owner::insert_batch_request_bytes(item_count, 7));
  auto* exact_header = reinterpret_cast<service::storage_owner::InsertBatchRequestHeader*>(exact.data());
  exact_header->anchor_hint_count = 0;
  require(service::storage_owner::request_anchor_hints(exact.data(), item_count) == nullptr,
          "exact request unexpectedly exposes anchor hints");
}

void test_anchor_routing() {
  const filepath_t prefix{"/tmp/dvstor_anchor_update_protocol_test"};
  std::filesystem::remove(prefix.string() + ".anchors");
  write_test_sidecar(prefix);

  vamana::anchor::Index index;
  str error;
  require(index.load(prefix, 3, 2, &error), error.c_str());
  const vec<float> query{100.0f, 100.0f, 100.0f};
  const auto route = index.route(query, 2);
  require(route.owner == 1 && route.hints.size() == 2,
          "semantic anchor route is incorrect");
  require(route.hints.front().memory_node() == 1,
          "semantic anchor hint came from the wrong shard");

  const auto override_route = index.route(query, 4, 0);
  require(override_route.owner == 0 && override_route.hints.size() == 4,
          "owner override route is incorrect");
  require(override_route.hints[0].memory_node() == 0 &&
            override_route.hints[2].memory_node() == 1,
          "owner override did not mix local and semantic anchors");
  std::filesystem::remove(prefix.string() + ".anchors");
}

}  // namespace

int main() {
  test_protocol_alignment();
  test_anchor_routing();
  return EXIT_SUCCESS;
}
