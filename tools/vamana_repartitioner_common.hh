#pragma once

#include <memory>

#include "common/types.hh"
#include "common/vector_dtype.hh"
#include "nlohmann/json.hh"
#include "tools/vamana_offline/partitioning.hh"

namespace tools::vamana_repartition {

struct Options {
  filepath_t input_prefix;
  filepath_t output_prefix;
  u32 memory_nodes{};
  u32 dim{};
  u32 R{};
  VectorDType vector_dtype{VectorDType::float32};
  bool vector_dtype_set{false};
  str storage_format{"auto"};
  u32 anchors_per_shard{};
  bool anchors_per_shard_set{false};
  u64 anchor_seed{1234};
  bool overwrite{false};
};

struct CrossShardStats {
  size_t total_edges{};
  size_t cross_edges{};

  double ratio() const {
    return total_edges == 0
      ? 0.0
      : static_cast<double>(cross_edges) / static_cast<double>(total_edges);
  }
};

struct WriteResult {
  CrossShardStats after_stats;
  size_t node_count{};
};

class Index {
public:
  explicit Index(Options options);
  ~Index();

  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;
  Index(Index&&) noexcept;
  Index& operator=(Index&&) noexcept;

  const Options& options() const;
  size_t node_count() const;
  u32 medoid_vertex() const;
  const str& input_storage_format() const;
  const str& output_storage_format() const;

  vec<vec<u32>> read_neighbor_lists(CrossShardStats* stats) const;
  vec<u64> read_partition_edges(u32 max_degree, CrossShardStats* stats) const;

  WriteResult write(const vec<u32>& parts,
                    const str& partition_strategy,
                    const tools::vamana_offline::PartitionStats& partition_stats,
                    const CrossShardStats& before_stats,
                    const nlohmann::json& partition_metadata = {}) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tools::vamana_repartition
