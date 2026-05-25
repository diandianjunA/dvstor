#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.hh"

namespace tools::breakdown_benchmark {

using ConfigMap = std::unordered_map<std::string, std::string>;

struct Args {
  std::string service_config_path;
  std::string workload{"both"};

ConfigMap read_config(const std::string& path);
std::vector<std::string> build_service_argv(const std::string& service_config_path);
std::vector<char*> make_argv(std::vector<std::string>& args);
Args parse_args(int argc, char** argv);

}  // namespace tools::breakdown_benchmark
