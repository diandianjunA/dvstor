#include "utils.hh"

#include <arpa/inet.h>
#include <netdb.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <stdexcept>

void lib_failure(const str&& message) {
  std::cerr << "[ERROR]: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

bool is_ipv4_literal(const str& value) {
  in_addr parsed{};
  return inet_pton(AF_INET, value.c_str(), &parsed) == 1;
}

std::string get_ip(const str& node_name) {
  std::map<str, str> node_to_ip{
    {"cluster1", "127.0.0.1"},
    {"cluster2", "192.168.6.201"},
    {"cluster3", "192.168.6.202"},
  };

  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* resolved = nullptr;
  const int status = getaddrinfo(node_name.c_str(), nullptr, &hints, &resolved);
  if (status != 0 || resolved == nullptr) {
    const auto legacy_alias = node_to_ip.find(node_name);
    lib_assert(legacy_alias != node_to_ip.end(),
               "Cannot resolve IPv4 host " + node_name + ": " +
                 gai_strerror(status));
    return legacy_alias->second;
  }
  const auto* ipv4 = reinterpret_cast<const sockaddr_in*>(resolved->ai_addr);
  char address[INET_ADDRSTRLEN]{};
  const char* converted = inet_ntop(
    AF_INET, &ipv4->sin_addr, address, sizeof(address));
  lib_assert(converted != nullptr,
             "Cannot format resolved IPv4 host: " + node_name);
  const str result{address};
  freeaddrinfo(resolved);
  return result;
}

Endpoint parse_endpoint(const str& endpoint, u32 default_port) {
  lib_assert(!endpoint.empty(), "Endpoint must not be empty");

  str host = endpoint;
  u32 port = default_port;

  const auto colon_pos = endpoint.find(':');
  if (colon_pos != str::npos) {
    host = endpoint.substr(0, colon_pos);
    const str port_str = endpoint.substr(colon_pos + 1);
    lib_assert(!host.empty(), "Endpoint host must not be empty: " + endpoint);
    lib_assert(!port_str.empty(), "Endpoint port must not be empty: " + endpoint);
    lib_assert(std::all_of(port_str.begin(), port_str.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; }),
               "Endpoint port must be numeric: " + endpoint);

    try {
      const auto parsed_port = std::stoul(port_str);
      lib_assert(parsed_port > 0 && parsed_port <= 65535,
                 "Endpoint port out of range: " + endpoint);
      port = static_cast<u32>(parsed_port);
    } catch (const std::exception&) {
      lib_assert(false, "Invalid endpoint port: " + endpoint);
    }
  }

  lib_assert(port > 0 && port <= 65535,
             "Endpoint default port out of range: " + endpoint);

  const str address = is_ipv4_literal(host) ? host : get_ip(host);
  return Endpoint{host, address, port};
}

f64 compute_throughput(i32 message_size,
                       i32 repeats,
                       Timepoint start,
                       Timepoint end) {
  return message_size / (ToSeconds(end - start).count() / repeats) /
         std::pow(1000, 2);
}

f64 compute_latency(i32 repeats,
                    Timepoint start,
                    Timepoint end,
                    bool is_read_or_atomic) {
  i32 rtt_factor = is_read_or_atomic ? 1 : 2;
  return ToMicroSeconds(end - start).count() / repeats / rtt_factor;
}

void print_status(str&& status) {
  std::cerr << "[STATUS]: " << status << std::endl;
}
