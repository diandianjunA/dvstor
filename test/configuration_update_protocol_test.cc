#include <cassert>
#include <string>
#include <vector>

#include "common/configuration.hh"

namespace {

configuration::IndexConfiguration make_config(bool explicit_disable) {
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

  const auto query_only_config = make_config(true);
  assert(!query_only_config.enable_updates);

  // Stage2 is mandatory for the sole supported update pipeline.
  assert(default_config.storage_owner_maintenance_workers > 0);
  return 0;
}
