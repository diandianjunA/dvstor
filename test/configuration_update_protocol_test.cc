#include <cassert>
#include <string>
#include <vector>

#include "common/configuration.hh"

namespace {

configuration::IndexConfiguration make_config(
    bool explicit_disable, bool explicit_namespace = false,
    bool bypass_dynamic_cache = false) {
  std::vector<std::string> arguments{
    "configuration_update_protocol_test",
    "--servers", "127.0.0.1:1234",
    "--storage-peers", "127.0.0.1:1234",
    "--index-prefix", "/tmp/index",
    "--threads", "1",
    "--dim", "128",
    "--k", "10",
  };
  if (explicit_disable) {
    arguments.emplace_back("--enable-updates");
    arguments.emplace_back("false");
  }
  if (explicit_namespace) {
    arguments.emplace_back("--vector-id-namespace-size");
    arguments.emplace_back("2000000");
  }
  if (bypass_dynamic_cache) {
    arguments.emplace_back("--gpu-dynamic-code-cache-entries");
    arguments.emplace_back("0");
  }
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (auto& argument : arguments) argv.push_back(argument.data());
  return configuration::IndexConfiguration(
    static_cast<int>(argv.size()), argv.data());
}

}  // namespace

int main() {
  const auto default_config = make_config(false);
  assert(default_config.enable_updates);
  assert(default_config.vector_id_namespace_size == default_config.max_vectors);
  // Mixed CPU/GPU traffic may transiently delay a CQE; the default must not
  // retain the old 20 ms false-failure threshold.
  assert(default_config.gpu_direct_timeout_ms == 250);

  const auto expanded_namespace = make_config(false, true);
  assert(expanded_namespace.max_vectors == 1'000'000);
  assert(expanded_namespace.vector_id_namespace_size == 2'000'000);

  const auto query_only_config = make_config(true);
  assert(!query_only_config.enable_updates);

  const auto cache_bypass_config = make_config(false, false, true);
  assert(cache_bypass_config.gpu_dynamic_code_cache_entries == 0);

  // Stage2 is mandatory for the sole supported update pipeline.
  assert(default_config.storage_owner_maintenance_workers > 0);
  return 0;
}
