#include <cassert>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "common/configuration.hh"

namespace {

configuration::IndexConfiguration make_config(
    bool explicit_disable, bool explicit_namespace = false,
    std::string beam_merge_policy = {},
    std::string graph_read_policy = {},
    std::string dynamic_graph_extent = {},
    std::string home_rpc_combining = {},
    std::string dynamic_graph_access_mode = {},
    std::string search_progression_mode = {},
    std::string graph_commit_width = {},
    std::string graph_issue_width = {},
    std::string update_completion_mode = {}) {
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
  if (!beam_merge_policy.empty()) {
    arguments.emplace_back("--gpu-query-beam-merge-policy");
    arguments.emplace_back(std::move(beam_merge_policy));
  }
  if (!graph_read_policy.empty()) {
    arguments.emplace_back("--gpu-query-graph-read-policy");
    arguments.emplace_back(std::move(graph_read_policy));
  }
  if (!dynamic_graph_extent.empty()) {
    arguments.emplace_back("--gpu-dynamic-graph-extent");
    arguments.emplace_back(std::move(dynamic_graph_extent));
  }
  if (!home_rpc_combining.empty()) {
    arguments.emplace_back("--storage-owner-stage2-home-rpc-combining");
    arguments.emplace_back(std::move(home_rpc_combining));
  }
  if (!dynamic_graph_access_mode.empty()) {
    arguments.emplace_back("--gpu-dynamic-graph-access-mode");
    arguments.emplace_back(std::move(dynamic_graph_access_mode));
  }
  if (!search_progression_mode.empty()) {
    arguments.emplace_back("--gpu-rdma-search-progression-mode");
    arguments.emplace_back(std::move(search_progression_mode));
  }
  if (!graph_commit_width.empty()) {
    arguments.emplace_back("--gpu-graph-commit-width");
    arguments.emplace_back(std::move(graph_commit_width));
  }
  if (!graph_issue_width.empty()) {
    arguments.emplace_back("--gpu-graph-issue-width");
    arguments.emplace_back(std::move(graph_issue_width));
  }
  if (!update_completion_mode.empty()) {
    arguments.emplace_back("--storage-owner-update-completion-mode");
    arguments.emplace_back(std::move(update_completion_mode));
  }
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (auto& argument : arguments) argv.push_back(argument.data());
  return configuration::IndexConfiguration(
    static_cast<int>(argv.size()), argv.data());
}

template <typename Function>
void expect_configuration_rejected(Function&& function) {
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    function();
    _exit(EXIT_SUCCESS);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFEXITED(status));
  assert(WEXITSTATUS(status) != EXIT_SUCCESS);
}

}  // namespace

int main() {
  const auto default_config = make_config(false);
  assert(default_config.enable_updates);
  assert(default_config.vector_id_namespace_size == default_config.max_vectors);
  // Mixed CPU/GPU traffic may transiently delay a CQE; the default must not
  // retain the old 20 ms false-failure threshold.
  assert(default_config.gpu_direct_timeout_ms == 250);
  assert(default_config.gpu_query_beam_merge_policy == "legacy");
  assert(default_config.dynamic_graph_access_mode == "manual");
  assert(default_config.gpu_query_graph_read_policy == "fixed");
  assert(default_config.gpu_dynamic_graph_extent);
  assert(default_config.gpu_rdma_search_progression_mode == "manual");
  assert(default_config.storage_owner_update_completion_mode ==
         "decoupled");
  assert(!default_config.synchronous_exact_updates_enabled());

  const auto coupled_updates = make_config(
    false, false, {}, {}, {}, {}, {}, {}, {}, {}, "COUPLED");
  assert(coupled_updates.storage_owner_update_completion_mode == "coupled");
  assert(coupled_updates.synchronous_exact_updates_enabled());

  const auto stable_run_config =
    make_config(false, false, "STABLE-RUN");
  assert(stable_run_config.gpu_query_beam_merge_policy == "stable-run");
  const auto live_extent_config =
    make_config(false, false, {}, "LIVE-EXTENT");
  assert(live_extent_config.gpu_query_graph_read_policy == "live-extent");
  const auto static_only_extent_config =
    make_config(false, false, {}, "LIVE-EXTENT", "false");
  assert(static_only_extent_config.gpu_query_graph_read_policy ==
         "live-extent");
  assert(!static_only_extent_config.gpu_dynamic_graph_extent);

  const auto fixed_graph_access = make_config(
    false, false, {}, {}, {}, {}, "FIXED");
  assert(fixed_graph_access.dynamic_graph_access_mode == "fixed");
  assert(fixed_graph_access.gpu_query_graph_read_policy == "fixed");
  assert(!fixed_graph_access.gpu_dynamic_graph_extent);
  assert(!fixed_graph_access.adaptive_dynamic_graph_access_enabled());

  const auto adaptive_graph_access = make_config(
    false, false, {}, {}, {}, {}, "ADAPTIVE");
  assert(adaptive_graph_access.dynamic_graph_access_mode == "adaptive");
  assert(adaptive_graph_access.gpu_query_graph_read_policy == "live-extent");
  assert(adaptive_graph_access.gpu_dynamic_graph_extent);
  assert(adaptive_graph_access.adaptive_dynamic_graph_access_enabled());

  const auto coupled_progression = make_config(
    false, false, {}, {}, {}, {}, {}, "COUPLED", "16", "0");
  assert(coupled_progression.gpu_rdma_search_progression_mode == "coupled");
  assert(coupled_progression.gpu_graph_commit_width == 16);
  assert(coupled_progression.gpu_graph_issue_width == 16);
  assert(coupled_progression.gpu_query_beam_merge_policy == "legacy");
  assert(!coupled_progression.decoupled_gpu_rdma_search_progression_enabled());

  const auto decoupled_progression = make_config(
    false, false, {}, {}, {}, {}, {}, "DECOUPLED", "16", "32");
  assert(decoupled_progression.gpu_rdma_search_progression_mode ==
         "decoupled");
  assert(decoupled_progression.gpu_graph_commit_width == 16);
  assert(decoupled_progression.gpu_graph_issue_width == 32);
  assert(decoupled_progression.gpu_query_beam_merge_policy == "stable-run");
  assert(decoupled_progression.decoupled_gpu_rdma_search_progression_enabled());

  // Formal contribution modes reject an explicitly requested half-enabled
  // lower-level configuration. Manual mode above remains available for the
  // static-only and other engineering ablations.
  expect_configuration_rejected([] {
    (void)make_config(
      false, false, {}, "live-extent", {}, {}, "fixed");
  });
  expect_configuration_rejected([] {
    (void)make_config(
      false, false, {}, {}, {}, {}, {}, "decoupled", "16", "16");
  });

  const auto expanded_namespace = make_config(false, true);
  assert(expanded_namespace.max_vectors == 1'000'000);
  assert(expanded_namespace.vector_id_namespace_size == 2'000'000);

  const auto query_only_config = make_config(true);
  assert(!query_only_config.enable_updates);

  assert(default_config.storage_owner_stage2_home_rpc_combining);
  const auto direct_home_rpc_config =
    make_config(false, false, {}, {}, {}, "false");
  assert(!direct_home_rpc_config.storage_owner_stage2_home_rpc_combining);

  // Stage2 is mandatory for the sole supported update pipeline.
  assert(default_config.storage_owner_maintenance_workers > 0);
  return 0;
}
