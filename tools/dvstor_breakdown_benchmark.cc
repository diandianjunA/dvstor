#include <cstdlib>
#include <execinfo.h>
#include <iostream>
#include <signal.h>
#include <unistd.h>

#include "common/configuration.hh"
#include "service/compute_service.hh"
#include "tools/breakdown_benchmark/args.hh"
#include "tools/breakdown_benchmark/workload.hh"

namespace {

void segfault_handler(int signal) {
  void* frames[64];
  const int count = backtrace(frames, 64);
  const char header[] = "\n[breakdown] fatal signal, backtrace:\n";
  const ssize_t ignored = ::write(STDERR_FILENO, header, sizeof(header) - 1);
  (void)ignored;
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  _exit(128 + signal);
}

}  // namespace

using namespace tools::breakdown_benchmark;

int main(int argc, char** argv) {
  signal(SIGSEGV, segfault_handler);
  try {
    const Args args = parse_args(argc, argv);
    auto service_args = build_service_argv(args.service_config_path);
    auto service_argv = make_argv(service_args);
    configuration::IndexConfiguration config(
      static_cast<int>(service_argv.size()), service_argv.data());
    preflight_workload_inputs(config, args);
    ComputeService service(config);
    (void)run_benchmark(service, args);
  } catch (const std::exception& e) {
    std::cerr << "breakdown benchmark failed: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
