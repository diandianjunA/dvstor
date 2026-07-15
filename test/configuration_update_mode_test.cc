#include <cassert>
#include <string>
#include <vector>

#include "common/configuration.hh"

namespace {

configuration::IndexConfiguration make_config(bool explicit_disable,
                                               bool local_stitch = false) {
  std::vector<std::string> arguments{
    "configuration_update_mode_test",
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
  if (local_stitch) {
    arguments.emplace_back("--storage-owner-update-mode");
    arguments.emplace_back("local_stitch");
    arguments.emplace_back("--storage-owner-maintenance-mode");
    arguments.emplace_back("finalize");
    arguments.emplace_back("--storage-owner-maintenance-workers");
    arguments.emplace_back("1");
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

  // local_stitch has no static-anchor tuning dependency.  Finalization is its
  // only additional configuration requirement.
  const auto local_stitch_config = make_config(false, true);
  assert(local_stitch_config.storage_owner_update_mode == "local_stitch");
  assert(local_stitch_config.storage_owner_maintenance_mode == "finalize");
  return 0;
}
