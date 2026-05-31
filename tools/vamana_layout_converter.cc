#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/index_path.hh"
#include "common/types.hh"
#include "nlohmann/json.hh"

namespace {

constexpr size_t kShardHeaderBytes = 16;
constexpr size_t kNodeHeaderBytes = 8;
constexpr size_t kNodeMetaBytes = 8;
constexpr size_t kNodeFixedPrefixBytes = kNodeHeaderBytes + kNodeMetaBytes;

enum class NodeLayout {
  legacy,
  rabitq_search_block,
};

struct Options {
  filepath_t input_prefix;
  filepath_t output_prefix;
  std::string from_layout;
  std::string to_layout{"rabitq_search_block"};
  u32 memory_nodes{};
  u32 dim{};
  u32 R{};
  u32 rabitq_bits{};
  bool overwrite{false};
};

struct NodeFormat {
  size_t vector_offset{};
  size_t rabitq_offset{};
  size_t neighbors_offset{};
  size_t vector_bytes{};
  size_t rabitq_bytes{};
  size_t neighbors_bytes{};
  size_t node_bytes{};
  size_t aligned_node_bytes{};
};

[[noreturn]] void usage(const char* argv0, const std::string& error = {}) {
  if (!error.empty()) {
    std::cerr << "error: " << error << "\n\n";
  }
  std::cerr
    << "Usage: " << argv0 << " --input-prefix OLD_PREFIX [--output-prefix NEW_PREFIX]\n"
    << "       [--from-layout legacy|rabitq_search_block]\n"
    << "       [--to-layout legacy|rabitq_search_block]\n"
    << "       [--memory-nodes N --dim D --R R --rabitq-bits B]\n"
    << "       [--overwrite]\n\n"
    << "The converter rewrites shard node layout without rebuilding the Vamana graph.\n"
    << "By default it reads OLD_PREFIX.meta.json and writes OLD_PREFIX_<to-layout>.\n";
  std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

std::string require_value(int& i, int argc, char** argv, const std::string& key) {
  if (i + 1 >= argc) {
    usage(argv[0], key + " requires a value");
  }
  return argv[++i];
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else if (arg == "--input-prefix") {
      options.input_prefix = require_value(i, argc, argv, arg);
    } else if (arg == "--output-prefix") {
      options.output_prefix = require_value(i, argc, argv, arg);
    } else if (arg == "--from-layout") {
      options.from_layout = require_value(i, argc, argv, arg);
    } else if (arg == "--to-layout") {
      options.to_layout = require_value(i, argc, argv, arg);
    } else if (arg == "--memory-nodes") {
      options.memory_nodes = static_cast<u32>(std::stoul(require_value(i, argc, argv, arg)));
    } else if (arg == "--dim") {
      options.dim = static_cast<u32>(std::stoul(require_value(i, argc, argv, arg)));
    } else if (arg == "--R") {
      options.R = static_cast<u32>(std::stoul(require_value(i, argc, argv, arg)));
    } else if (arg == "--rabitq-bits") {
      options.rabitq_bits = static_cast<u32>(std::stoul(require_value(i, argc, argv, arg)));
    } else if (arg == "--overwrite") {
      options.overwrite = true;
    } else {
      usage(argv[0], "unknown argument: " + arg);
    }
  }

  if (options.input_prefix.empty()) {
    usage(argv[0], "--input-prefix is required");
  }
  return options;
}

NodeLayout parse_layout(const std::string& name) {
  if (name == "legacy") {
    return NodeLayout::legacy;
  }
  if (name == "rabitq_search_block") {
    return NodeLayout::rabitq_search_block;
  }
  throw std::runtime_error("unknown node layout: " + name);
}

std::string layout_name(NodeLayout layout) {
  switch (layout) {
    case NodeLayout::legacy:
      return "legacy";
    case NodeLayout::rabitq_search_block:
      return "rabitq_search_block";
  }
  return "unknown";
}

filepath_t metadata_path(const filepath_t& prefix) {
  return filepath_t(prefix.string() + ".meta.json");
}

filepath_t rotation_path(const filepath_t& prefix) {
  return filepath_t(prefix.string() + ".rotation.bin");
}

std::optional<nlohmann::json> load_metadata(const filepath_t& prefix) {
  const filepath_t path = metadata_path(prefix);
  if (!std::filesystem::exists(path)) {
    return std::nullopt;
  }
  std::ifstream input(path);
  if (!input.good()) {
    throw std::runtime_error("failed to open metadata: " + path.string());
  }
  nlohmann::json metadata;
  input >> metadata;
  return metadata;
}

void fill_from_metadata(Options& options, const std::optional<nlohmann::json>& metadata) {
  if (!metadata.has_value()) {
    return;
  }
  const auto& meta = *metadata;
  if (options.from_layout.empty()) {
    options.from_layout = meta.value("node_layout", std::string{"legacy"});
  }
  if (options.memory_nodes == 0) {
    options.memory_nodes = meta.value("num_memory_nodes", 0u);
  }
  if (options.dim == 0) {
    options.dim = meta.value("dim", 0u);
  }
  if (options.R == 0) {
    options.R = meta.value("R", 0u);
  }
  if (options.rabitq_bits == 0) {
    options.rabitq_bits = meta.value("rabitq_bits", 0u);
  }
}

size_t align8(size_t value) {
  return (value + 7) & ~static_cast<size_t>(7);
}

NodeFormat make_format(NodeLayout layout, u32 dim, u32 R, u32 rabitq_bits) {
  NodeFormat format;
  format.vector_bytes = static_cast<size_t>(dim) * sizeof(element_t);
  format.rabitq_bytes = (static_cast<size_t>(rabitq_bits) * dim + 7) / 8 + 2 * sizeof(f32);
  format.neighbors_bytes = static_cast<size_t>(R) * sizeof(u64);
  format.node_bytes = kNodeFixedPrefixBytes + format.vector_bytes + format.rabitq_bytes + format.neighbors_bytes;
  format.aligned_node_bytes = align8(format.node_bytes);

  if (layout == NodeLayout::rabitq_search_block) {
    format.rabitq_offset = kNodeFixedPrefixBytes;
    format.neighbors_offset = format.rabitq_offset + format.rabitq_bytes;
    format.vector_offset = format.neighbors_offset + format.neighbors_bytes;
  } else {
    format.vector_offset = kNodeFixedPrefixBytes;
    format.rabitq_offset = format.vector_offset + format.vector_bytes;
    format.neighbors_offset = format.rabitq_offset + format.rabitq_bytes;
  }
  return format;
}

u64 read_u64_le(const unsigned char* ptr) {
  u64 value = 0;
  std::memcpy(&value, ptr, sizeof(value));
  return value;
}

void transform_node(const std::vector<unsigned char>& input,
                    std::vector<unsigned char>& output,
                    const NodeFormat& from,
                    const NodeFormat& to) {
  std::fill(output.begin(), output.end(), 0);
  std::memcpy(output.data(), input.data(), kNodeFixedPrefixBytes);
  std::memcpy(output.data() + to.vector_offset,
              input.data() + from.vector_offset,
              from.vector_bytes);
  std::memcpy(output.data() + to.rabitq_offset,
              input.data() + from.rabitq_offset,
              from.rabitq_bytes);
  std::memcpy(output.data() + to.neighbors_offset,
              input.data() + from.neighbors_offset,
              from.neighbors_bytes);
  if (from.aligned_node_bytes > from.node_bytes) {
    std::memcpy(output.data() + to.node_bytes,
                input.data() + from.node_bytes,
                from.aligned_node_bytes - from.node_bytes);
  }
}

void read_exact(std::istream& input, void* dst, size_t bytes, const filepath_t& path) {
  input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(input.gcount()) != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
}

void copy_remaining(std::istream& input, std::ostream& output) {
  std::vector<char> buffer(4 * 1024 * 1024);
  while (input.good()) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = input.gcount();
    if (got > 0) {
      output.write(buffer.data(), got);
    }
  }
}

size_t convert_shard(const filepath_t& input_path,
                     const filepath_t& output_path,
                     const NodeFormat& from,
                     const NodeFormat& to,
                     bool overwrite) {
  if (!std::filesystem::exists(input_path)) {
    throw std::runtime_error("input shard does not exist: " + input_path.string());
  }
  if (std::filesystem::exists(output_path) && !overwrite) {
    throw std::runtime_error("output shard exists, pass --overwrite: " + output_path.string());
  }
  const size_t file_size = std::filesystem::file_size(input_path);
  if (file_size < kShardHeaderBytes) {
    throw std::runtime_error("input shard is too small: " + input_path.string());
  }

  std::ifstream input(input_path, std::ios::binary);
  if (!input.good()) {
    throw std::runtime_error("failed to open input shard: " + input_path.string());
  }

  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  const filepath_t temp_path = filepath_t(output_path.string() + ".tmp");
  std::filesystem::remove(temp_path);
  std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
  if (!output.good()) {
    throw std::runtime_error("failed to open output shard: " + temp_path.string());
  }

  unsigned char shard_header[kShardHeaderBytes]{};
  read_exact(input, shard_header, sizeof(shard_header), input_path);
  output.write(reinterpret_cast<const char*>(shard_header), sizeof(shard_header));

  const u64 free_ptr = read_u64_le(shard_header);
  if (free_ptr < kShardHeaderBytes || free_ptr > file_size) {
    throw std::runtime_error("invalid free_ptr in " + input_path.string() +
                             ": " + std::to_string(free_ptr) +
                             " file_size=" + std::to_string(file_size));
  }

  std::vector<unsigned char> input_node(from.aligned_node_bytes);
  std::vector<unsigned char> output_node(to.aligned_node_bytes);
  size_t converted = 0;
  size_t offset = kShardHeaderBytes;
  while (offset + from.aligned_node_bytes <= free_ptr) {
    read_exact(input, input_node.data(), input_node.size(), input_path);
    transform_node(input_node, output_node, from, to);
    output.write(reinterpret_cast<const char*>(output_node.data()),
                 static_cast<std::streamsize>(output_node.size()));
    offset += from.aligned_node_bytes;
    ++converted;
  }

  copy_remaining(input, output);
  output.close();
  if (!output.good()) {
    throw std::runtime_error("failed to write output shard: " + temp_path.string());
  }
  if (std::filesystem::exists(output_path)) {
    std::filesystem::remove(output_path);
  }
  std::filesystem::rename(temp_path, output_path);
  return converted;
}

void copy_rotation_file(const filepath_t& input_prefix, const filepath_t& output_prefix, bool overwrite) {
  const filepath_t input = rotation_path(input_prefix);
  if (!std::filesystem::exists(input)) {
    std::cerr << "warning: rotation file not found: " << input << "\n";
    return;
  }
  const filepath_t output = rotation_path(output_prefix);
  if (std::filesystem::exists(output) && !overwrite) {
    throw std::runtime_error("output rotation file exists, pass --overwrite: " + output.string());
  }
  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  std::filesystem::copy_file(input,
                             output,
                             overwrite ? std::filesystem::copy_options::overwrite_existing
                                       : std::filesystem::copy_options::none);
}

void write_metadata(const filepath_t& output_prefix,
                    const std::optional<nlohmann::json>& input_metadata,
                    const Options& options,
                    const NodeFormat& to,
                    const std::string& to_layout,
                    bool overwrite) {
  const filepath_t output = metadata_path(output_prefix);
  if (std::filesystem::exists(output) && !overwrite) {
    throw std::runtime_error("output metadata exists, pass --overwrite: " + output.string());
  }

  nlohmann::json metadata = input_metadata.value_or(nlohmann::json::object());
  metadata["output_prefix"] = output_prefix.string();
  metadata["num_memory_nodes"] = options.memory_nodes;
  metadata["dim"] = options.dim;
  metadata["R"] = options.R;
  metadata["rabitq_bits"] = options.rabitq_bits;
  metadata["node_size"] = to.node_bytes;
  metadata["node_layout"] = to_layout;
  metadata["rabitq_size"] = to.rabitq_bytes;

  if (!output.parent_path().empty()) {
    std::filesystem::create_directories(output.parent_path());
  }
  std::ofstream out(output, std::ios::trunc);
  if (!out.good()) {
    throw std::runtime_error("failed to open output metadata: " + output.string());
  }
  out << std::setw(2) << metadata << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    const std::optional<nlohmann::json> metadata = load_metadata(options.input_prefix);
    fill_from_metadata(options, metadata);

    if (options.from_layout.empty()) {
      options.from_layout = "legacy";
    }
    if (options.output_prefix.empty()) {
      options.output_prefix = filepath_t(options.input_prefix.string() + "_" + options.to_layout);
    }
    if (options.input_prefix == options.output_prefix) {
      throw std::runtime_error("input-prefix and output-prefix must be different");
    }
    if (options.memory_nodes == 0 || options.dim == 0 || options.R == 0 || options.rabitq_bits == 0) {
      throw std::runtime_error("missing layout parameters; provide metadata or --memory-nodes/--dim/--R/--rabitq-bits");
    }

    const NodeLayout from_layout = parse_layout(options.from_layout);
    const NodeLayout to_layout = parse_layout(options.to_layout);
    const NodeFormat from = make_format(from_layout, options.dim, options.R, options.rabitq_bits);
    const NodeFormat to = make_format(to_layout, options.dim, options.R, options.rabitq_bits);

    if (from.node_bytes != to.node_bytes || from.aligned_node_bytes != to.aligned_node_bytes) {
      throw std::runtime_error("source and target layouts have different node sizes");
    }
    if (metadata.has_value()) {
      const size_t metadata_node_size = metadata->value("node_size", from.node_bytes);
      const size_t metadata_rabitq_size = metadata->value("rabitq_size", from.rabitq_bytes);
      if (metadata_node_size != from.node_bytes || metadata_rabitq_size != from.rabitq_bytes) {
        throw std::runtime_error("metadata node_size/rabitq_size does not match layout parameters");
      }
    }

    std::cerr << "input prefix: " << options.input_prefix << "\n"
              << "output prefix: " << options.output_prefix << "\n"
              << "layout: " << layout_name(from_layout) << " -> " << layout_name(to_layout) << "\n"
              << "memory nodes: " << options.memory_nodes << "\n"
              << "dim=" << options.dim << " R=" << options.R
              << " rabitq_bits=" << options.rabitq_bits
              << " node_size=" << from.node_bytes
              << " aligned_node_size=" << from.aligned_node_bytes << "\n";

    size_t total_nodes = 0;
    for (u32 shard = 0; shard < options.memory_nodes; ++shard) {
      const filepath_t input_shard = index_path::shard_file(options.input_prefix, shard + 1, options.memory_nodes);
      const filepath_t output_shard = index_path::shard_file(options.output_prefix, shard + 1, options.memory_nodes);
      const size_t converted = convert_shard(input_shard, output_shard, from, to, options.overwrite);
      total_nodes += converted;
      std::cerr << "converted shard " << (shard + 1) << "/" << options.memory_nodes
                << ": " << converted << " nodes\n";
    }

    copy_rotation_file(options.input_prefix, options.output_prefix, options.overwrite);
    write_metadata(options.output_prefix, metadata, options, to, layout_name(to_layout), options.overwrite);
    std::cerr << "conversion finished: " << total_nodes << " nodes\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "vamana_layout_converter failed: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
