#include "common/configuration.hh"
#include "service/compute_service.hh"

#include <csignal>
#include <iostream>

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
    std::cerr << "use dvstor_memory_node for a storage process\n";
    return EXIT_FAILURE;
  }
  ComputeService service{config};
  wait_for_shutdown_signal();

  return EXIT_SUCCESS;
}
