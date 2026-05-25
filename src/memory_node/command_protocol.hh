#pragma once

#include <cstddef>

#include "common/types.hh"

namespace mn_command {

enum Command : u32 { NOOP = 0, LOAD = 1, STORE = 2, SHUTDOWN = 3 };

struct Request {
  Command cmd;
  size_t path_length;
};

struct Response {
  bool success;
  size_t message_length;
};

}  // namespace mn_command
