#include <cassert>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

#include "common/index_path.hh"
#include "service/base_owner_map.hh"
#include "vamana/idmap.hh"

namespace {

using Entries = std::vector<vamana::idmap::Entry>;

vamana::idmap::Entry entry(node_t id, u32 physical_shard) {
  return vamana::idmap::Entry{
    id,
    (static_cast<u64>(physical_shard) << 48) |
      (static_cast<u64>(id) + 64),
    0,
    0};
}

void write_sidecar(const filepath_t& prefix,
                   u32 owner_count,
                   u32 owner,
                   const Entries& entries,
                   u64 declared_count = std::numeric_limits<u64>::max(),
                   u32 header_owner = std::numeric_limits<u32>::max()) {
  const filepath_t path = index_path::owner_idmap_file(
    prefix, static_cast<size_t>(owner) + 1, owner_count);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  assert(output.good());
  vamana::idmap::Header header;
  header.owner_shard = header_owner == std::numeric_limits<u32>::max()
    ? owner : header_owner;
  header.shard_count = owner_count;
  header.entry_count = declared_count == std::numeric_limits<u64>::max()
    ? entries.size() : declared_count;
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!entries.empty()) {
    output.write(reinterpret_cast<const char*>(entries.data()),
                 static_cast<std::streamsize>(entries.size() * sizeof(entries[0])));
  }
  assert(output.good());
}

filepath_t prefix_in(const filepath_t& directory, const char* name) {
  const filepath_t subdirectory = directory / name;
  std::filesystem::create_directories(subdirectory);
  return subdirectory / "index";
}

void test_metis_owner_is_not_id_modulo(const filepath_t& directory) {
  const filepath_t prefix = prefix_in(directory, "metis");
  write_sidecar(prefix, 3, 0, {entry(2, 0), entry(100, 0)});
  write_sidecar(prefix, 3, 1, {entry(3, 1)});
  write_sidecar(prefix, 3, 2, {entry(4, 2)});

  service::BaseOwnerMap owners;
  str error;
  assert(owners.load(prefix, 3, "owner_sharded_v1", &error));
  assert(owners.entry_count() == 4);
  assert(owners.owner_for(2) == 0);  // 2 % 3 would be owner 2.
  assert(owners.owner_for(3) == 1);  // 3 % 3 would be owner 0.
  assert(owners.owner_for(4) == 2);  // 4 % 3 would be owner 1.
  assert(!owners.owner_for(99).has_value());
}

void test_conflicting_duplicate_fails(const filepath_t& directory) {
  const filepath_t prefix = prefix_in(directory, "conflict");
  write_sidecar(prefix, 2, 0, {entry(7, 0)});
  write_sidecar(prefix, 2, 1, {entry(7, 1)});

  service::BaseOwnerMap owners;
  str error;
  assert(!owners.load(prefix, 2, "owner_sharded_v1", &error));
  assert(error.find("conflicting owner") != str::npos);
}

void test_truncated_file_fails(const filepath_t& directory) {
  const filepath_t prefix = prefix_in(directory, "truncated");
  write_sidecar(prefix, 2, 0, {entry(1, 0)}, 2);
  write_sidecar(prefix, 2, 1, {});

  service::BaseOwnerMap owners;
  str error;
  assert(!owners.load(prefix, 2, "owner_sharded_v1", &error));
  assert(error.find("file size mismatch") != str::npos);
}

void test_header_and_metadata_format_are_strict(const filepath_t& directory) {
  const filepath_t prefix = prefix_in(directory, "header");
  write_sidecar(prefix, 1, 0, {entry(1, 0)},
                std::numeric_limits<u64>::max(), 1);

  service::BaseOwnerMap owners;
  str error;
  assert(!owners.load(prefix, 1, "owner_sharded_v1", &error));
  assert(error.find("invalid owner idmap header") != str::npos);
  assert(!owners.load(prefix, 1, "legacy", &error));
  assert(error.find("idmap_format=owner_sharded_v1") != str::npos);
}

void test_missing_owner_sidecar_fails(const filepath_t& directory) {
  const filepath_t prefix = prefix_in(directory, "missing");
  write_sidecar(prefix, 2, 0, {entry(1, 0)});

  service::BaseOwnerMap owners;
  str error;
  assert(!owners.load(prefix, 2, "owner_sharded_v1", &error));
  assert(error.find("missing owner idmap sidecar") != str::npos);
}

}  // namespace

int main() {
  const filepath_t directory = std::filesystem::temp_directory_path() /
    ("dvstor_base_owner_map_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);

  test_metis_owner_is_not_id_modulo(directory);
  test_conflicting_duplicate_fails(directory);
  test_truncated_file_fails(directory);
  test_header_and_metadata_format_are_strict(directory);
  test_missing_owner_sidecar_fails(directory);

  std::filesystem::remove_all(directory);
  return 0;
}
