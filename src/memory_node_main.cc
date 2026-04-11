#include "common/configuration.hh"
#include "memory_node.hh"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::vector<std::string> force_server_mode(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc) + 1);
  args.emplace_back(argv[0]);

  const bool has_server_flag = std::any_of(argv + 1, argv + argc, [](const char* arg) {
    return std::string{arg} == "--is-server" || std::string{arg} == "-s";
  });
  if (!has_server_flag) {
    args.emplace_back("--is-server");
  }

  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  return args;
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

int main(int argc, char** argv) {
  auto args = force_server_mode(argc, argv);
  auto parsed_argv = make_argv(args);

  configuration::IndexConfiguration config{static_cast<int>(parsed_argv.size()), parsed_argv.data()};
  MemoryNode memory_node{config};
  return EXIT_SUCCESS;
}
