#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/distance.hh"
#include "service/compute_service.hh"

class ShineIndex {
public:
  explicit ShineIndex(const std::string& service_config_path);
  ~ShineIndex();

  void build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids);
  void insert(const std::vector<float>& vectors, const std::vector<uint32_t>& ids);
  size_t insert_count(const std::vector<float>& vectors, const std::vector<uint32_t>& ids);
  void search(const std::vector<float>& query,
              size_t top_k,
              std::vector<uint32_t>& ids,
              std::vector<float>& distances) const;
  void load(const std::string& index_path);
  void save(const std::string& index_path);
  std::string getIndexType() const;
  size_t dimension() const;

private:
  static std::vector<std::string> build_service_argv(const std::string& service_config_path);

private:
  bool ip_distance_{false};
  mutable std::unique_ptr<ComputeService<L2Distance>> l2_service_;
  mutable std::unique_ptr<ComputeService<IPDistance>> ip_service_;
};
