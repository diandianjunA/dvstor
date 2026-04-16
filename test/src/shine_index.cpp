#include "shine_index.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

using ConfigMap = std::unordered_map<std::string, std::string>;

std::string trim(std::string value) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

ConfigMap read_config(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open config: " + path);
  }

  ConfigMap config;
  std::string line;
  while (std::getline(input, line)) {
    const auto comment_pos = line.find_first_of("#;");
    if (comment_pos != std::string::npos) {
      line.erase(comment_pos);
    }

    line = trim(line);
    if (line.empty()) {
      continue;
    }

    const auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      continue;
    }

    auto key = trim(line.substr(0, eq_pos));
    auto value = trim(line.substr(eq_pos + 1));
    if (!key.empty()) {
      config[std::move(key)] = std::move(value);
    }
  }

  return config;
}

bool is_truthy(const std::string& value) {
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

std::vector<std::string> split_tokens(const std::string& value) {
  std::string normalized = value;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');

  std::stringstream ss(normalized);
  std::vector<std::string> tokens;
  std::string token;
  while (ss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<char*> make_argv(std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  return argv;
}

}  // namespace

ShineIndex::ShineIndex(const std::string& service_config_path) {
  auto args = build_service_argv(service_config_path);
  auto argv = make_argv(args);
  configuration::IndexConfiguration config(static_cast<int>(argv.size()), argv.data());
  ip_distance_ = config.ip_distance;

  if (ip_distance_) {
    ip_service_ = std::make_unique<ComputeService<IPDistance>>(config, false);
  } else {
    l2_service_ = std::make_unique<ComputeService<L2Distance>>(config, false);
  }
}

ShineIndex::~ShineIndex() = default;

void ShineIndex::build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) {
  insert_count(vecs, ids);
}

void ShineIndex::insert(const std::vector<float>& vectors, const std::vector<uint32_t>& ids) {
  insert_count(vectors, ids);
}

size_t ShineIndex::insert_count(const std::vector<float>& vectors, const std::vector<uint32_t>& ids) {
  if (ids.empty()) {
    return 0;
  }
  if (vectors.size() != ids.size() * dimension()) {
    throw std::runtime_error("insert vector count/dimension mismatch");
  }

  size_t inserted_total = 0;
  constexpr size_t batch_size = 100;
  for (size_t offset = 0; offset < ids.size(); offset += batch_size) {
    const size_t end = std::min(offset + batch_size, ids.size());

    if (ip_distance_) {
      vec<ComputeService<IPDistance>::InsertItem> batch;
      batch.reserve(end - offset);
      for (size_t i = offset; i < end; ++i) {
        const auto begin = vectors.begin() + static_cast<std::ptrdiff_t>(i * dimension());
        const auto finish = begin + static_cast<std::ptrdiff_t>(dimension());
        batch.push_back({ids[i], vec<element_t>(begin, finish)});
      }
      inserted_total += ip_service_->insert(batch);

    } else {
      vec<ComputeService<L2Distance>::InsertItem> batch;
      batch.reserve(end - offset);
      for (size_t i = offset; i < end; ++i) {
        const auto begin = vectors.begin() + static_cast<std::ptrdiff_t>(i * dimension());
        const auto finish = begin + static_cast<std::ptrdiff_t>(dimension());
        batch.push_back({ids[i], vec<element_t>(begin, finish)});
      }
      inserted_total += l2_service_->insert(batch);
    }
  }

  return inserted_total;
}

void ShineIndex::search(const std::vector<float>& query,
                        size_t top_k,
                        std::vector<uint32_t>& ids,
                        std::vector<float>& distances) const {
  if (query.size() != dimension()) {
    throw std::runtime_error("search dimension mismatch");
  }

  vec<node_t> results;
  if (ip_distance_) {
    results = ip_service_->search(query, static_cast<u32>(top_k));
  } else {
    results = l2_service_->search(query, static_cast<u32>(top_k));
  }

  ids.assign(results.begin(), results.end());
  distances.assign(ids.size(), 0.0f);
}

void ShineIndex::load(const std::string& index_path) {
  str error_message;
  const bool ok = ip_distance_ ? ip_service_->load_index(index_path, &error_message)
                               : l2_service_->load_index(index_path, &error_message);
  if (!ok) {
    throw std::runtime_error("failed to load index: " + error_message);
  }
}

void ShineIndex::save(const std::string& index_path) {
  str error_message;
  const bool ok = ip_distance_ ? ip_service_->store_index(index_path, &error_message)
                               : l2_service_->store_index(index_path, &error_message);
  if (!ok) {
    throw std::runtime_error("failed to save index: " + error_message);
  }
}

std::string ShineIndex::getIndexType() const {
  return "Shine";
}

size_t ShineIndex::dimension() const {
  return ip_distance_ ? ip_service_->config().dim : l2_service_->config().dim;
}

std::vector<std::string> ShineIndex::build_service_argv(const std::string& service_config_path) {
  const auto config = read_config(service_config_path);
  std::vector<std::string> args;
  args.emplace_back("shine_index_smoke_test");

  static const std::vector<std::string> multi_keys = {"servers", "clients"};
  static const std::vector<std::string> flag_keys = {
    "initiator",
    "cache",
    "routing",
    "load-index",
    "store-index",
    "disable-thread-pinning",
    "no-recall",
    "ip-dist",
  };

  for (const auto& [key, value] : config) {
    const std::string option = "--" + key;
    if (std::find(flag_keys.begin(), flag_keys.end(), key) != flag_keys.end()) {
      if (is_truthy(value)) {
        args.push_back(option);
      }
      continue;
    }

    if (std::find(multi_keys.begin(), multi_keys.end(), key) != multi_keys.end()) {
      const auto tokens = split_tokens(value);
      if (!tokens.empty()) {
        args.push_back(option);
        args.insert(args.end(), tokens.begin(), tokens.end());
      }
      continue;
    }

    args.push_back(option);
    args.push_back(value);
  }

  return args;
}
