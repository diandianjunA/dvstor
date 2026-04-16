#include "common/configuration.hh"
#include "memory_node.hh"

#include <cstdlib>

int main(int argc, char** argv) {
  configuration::IndexConfiguration config{argc, argv};
  config.is_server = true;
  MemoryNode memory_node{config};
  return EXIT_SUCCESS;
}
