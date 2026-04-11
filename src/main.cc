#include "common/configuration.hh"
#include "common/distance.hh"
#include "memory_node.hh"
#include "service/compute_service.hh"

#include <csignal>

namespace {

void wait_for_shutdown_signal() {
  sigset_t block_set;
  sigemptyset(&block_set);
  sigaddset(&block_set, SIGINT);
  sigaddset(&block_set, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &block_set, nullptr);

  int sig = 0;
  sigwait(&block_set, &sig);
  print_status("received signal " + std::to_string(sig) + ", shutting down...");
}

}  // namespace

int main(int argc, char** argv) {
  configuration::IndexConfiguration config{argc, argv};

  if (config.is_server) {
    MemoryNode memory_node{config};
  } else {
    if (config.ip_distance) {
      ComputeService<IPDistance> service{config, true};
      wait_for_shutdown_signal();
    } else {
      ComputeService<L2Distance> service{config, true};
      wait_for_shutdown_signal();
    }
  }

  return EXIT_SUCCESS;
}
