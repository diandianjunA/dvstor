#pragma once

#include <library/utils.hh>

#include "types.hh"

namespace index_path {

inline filepath_t base_directory(const filepath_t& data_path) {
  lib_assert(!data_path.empty(), "data path must not be empty");
  return data_path.has_extension() ? data_path.parent_path() : data_path;
}

inline filepath_t default_prefix(const filepath_t& data_path, u32 m, u32 ef_construction) {
  return base_directory(data_path) / "dump" /
         ("index_m" + std::to_string(m) + "_efc" + std::to_string(ef_construction));
}

inline filepath_t resolve_prefix(const filepath_t& data_path,
                                 const filepath_t& explicit_prefix,
                                 u32 m,
                                 u32 ef_construction) {
  return explicit_prefix.empty() ? default_prefix(data_path, m, ef_construction) : explicit_prefix;
}

inline filepath_t shard_file(const filepath_t& prefix, size_t node_ordinal, size_t num_nodes) {
  return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" + std::to_string(num_nodes) +
                    ".dat");
}

inline filepath_t owner_idmap_file(const filepath_t& prefix, size_t node_ordinal, size_t num_nodes) {
  return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" +
                    std::to_string(num_nodes) + ".idmap");
}

inline filepath_t anchor_file(const filepath_t& prefix) {
  return filepath_t(prefix.string() + ".anchors");
}

inline filepath_t navigation_model_file(const filepath_t& prefix, u32 subquantizers) {
  return filepath_t(prefix.string() + ".pq" + std::to_string(subquantizers));
}

inline filepath_t navigation_code_file(const filepath_t& prefix,
                                       size_t node_ordinal,
                                       size_t num_nodes,
                                       u32 subquantizers) {
  return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" +
                    std::to_string(num_nodes) + ".pq" +
                    std::to_string(subquantizers) + ".codes");
}

inline filepath_t navigation_code_for_shard(const filepath_t& shard_file_path,
                                             u32 subquantizers) {
  filepath_t result = shard_file_path;
  result.replace_extension(".pq" + std::to_string(subquantizers) + ".codes");
  return result;
}

}  // namespace index_path
