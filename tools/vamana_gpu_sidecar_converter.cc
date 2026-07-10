#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "tools/vamana_offline/gpu_sidecar_converter.hh"

namespace {

void usage(const char* program) {
  std::cerr
    << "Usage: " << program << " --index-prefix PATH [options]\n"
    << "  --gpu-hot-degree N          GPU-resident neighbors per node (default 32)\n"
    << "  --gpu-entry-points N        Deterministic entry points (default 256)\n"
    << "  --gpu-graph-page-bytes N    Cold graph page bytes (default 4096)\n"
    << "  --threads N                 Parallel shard workers; 0 = auto\n"
    << "  --seed N                    Entry-point sampling seed (default 1234)\n"
    << "  --rabitq-source MODE        auto, sidecar, or nodes (default auto)\n"
    << "  --overwrite                 Replace existing GPU sidecars\n";
}

u64 parse_u64(const char* option, const char* value) {
  size_t consumed = 0;
  const std::string text{value};
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(std::string{option} + " expects an unsigned integer");
  }
  u64 parsed = 0;
  try {
    parsed = std::stoull(text, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string{option} + " expects an unsigned integer");
  }
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string{option} + " expects an unsigned integer");
  }
  return parsed;
}

u32 parse_u32(const char* option, const char* value) {
  const u64 parsed = parse_u64(option, value);
  if (parsed > std::numeric_limits<u32>::max()) {
    throw std::invalid_argument(std::string{option} + " exceeds uint32 range");
  }
  return static_cast<u32>(parsed);
}

}

int main(int argc, char** argv) {
  tools::vamana_offline::GpuSidecarConversionOptions options;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument{argv[index]};
      const auto require_value = [&](const char* option) -> const char* {
        if (index + 1 >= argc) {
          throw std::invalid_argument(std::string{option} + " requires a value");
        }
        return argv[++index];
      };
      if (argument == "--index-prefix") {
        options.index_prefix = require_value("--index-prefix");
      } else if (argument == "--gpu-hot-degree") {
        options.hot_degree = parse_u32("--gpu-hot-degree", require_value("--gpu-hot-degree"));
      } else if (argument == "--gpu-entry-points") {
        options.entry_points = parse_u32(
          "--gpu-entry-points", require_value("--gpu-entry-points"));
      } else if (argument == "--gpu-graph-page-bytes") {
        options.page_bytes = parse_u32(
          "--gpu-graph-page-bytes", require_value("--gpu-graph-page-bytes"));
      } else if (argument == "--threads") {
        options.threads = parse_u32("--threads", require_value("--threads"));
      } else if (argument == "--seed") {
        options.seed = parse_u64("--seed", require_value("--seed"));
      } else if (argument == "--rabitq-source") {
        options.rabitq_source = tools::vamana_offline::parse_gpu_rabitq_source(
          require_value("--rabitq-source"));
      } else if (argument == "--overwrite") {
        options.overwrite = true;
      } else if (argument == "--help" || argument == "-h") {
        usage(argv[0]);
        return EXIT_SUCCESS;
      } else {
        throw std::invalid_argument("unknown option: " + std::string{argument});
      }
    }
    if (options.index_prefix.empty()) {
      usage(argv[0]);
      return 2;
    }

    const auto result = tools::vamana_offline::convert_gpu_sidecars(options);
    std::cout << "GPU sidecar conversion complete\n"
              << "  index: " << result.index_file << "\n"
              << "  nodes: " << result.node_count << "\n"
              << "  graph edges: " << result.graph_edge_count << "\n"
              << "  hot edges: " << result.hot_edge_count << "\n"
              << "  entry points: " << result.entry_point_count << "\n"
              << "  RaBitQ source: "
              << (result.used_rabitq_sidecars ? "full sidecars" : "storage nodes") << "\n";
    for (size_t shard = 0; shard < result.graph_page_files.size(); ++shard) {
      std::cout << "  shard " << shard + 1 << ": " << result.graph_page_files[shard]
                << " remote_offset=" << result.graph_page_offsets[shard]
                << " bytes=" << result.graph_page_bytes[shard] << "\n";
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "GPU sidecar conversion failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
