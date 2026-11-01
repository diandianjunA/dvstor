#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>

#include <boost/program_options.hpp>

#include "common/index_path.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "tools/vamana_offline/checkpoint.hh"
#include "tools/vamana_offline/dataset_io.hh"
#include "tools/vamana_offline/graph.hh"
#include "tools/vamana_offline/progress.hh"
#include "tools/vamana_offline/recall_check.hh"
#include "tools/vamana_offline/reverse_repair.hh"
#include "vamana/hot_graph.hh"
#include "vamana/idmap.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_offline {
namespace {

struct Layout {
  u32 dim{};
  u32 degree{};
  u32 shards{};
  u32 shard_bits{};
  u32 node_bytes{};
  u32 entry_bytes{};
  u32 dynamic_record_bytes{};
  u64 nodes{};
  u64 build_fingerprint{};
  VectorDType dtype{};
  vec<u64> counts;
  vec<u64> offsets;
  vec<u64> dynamic_offsets;
  vec<u64> shard_fingerprints;
};

struct Options {
  filepath_t data_path;
  filepath_t index_prefix;
  vec<filepath_t> shard_prefixes;
  filepath_t query_path;
  filepath_t groundtruth_path;
  filepath_t gate_dir;
  filepath_t checkpoint_prefix;
  size_t medoid{};
  size_t max_vectors{std::numeric_limits<u32>::max()};
  size_t consolidation_chunk_bytes{};
  size_t sample_sources{1000};
  u32 threads{64};
  u32 beam_width{256};
  u32 reverse_slots{32};
  str vector_data_type{"auto"};
  bool apply{};
  bool resume_repaired{};
};

template <typename T>
vec<T> required_array(const nlohmann::json& metadata, const char* name,
                      u32 expected_size) {
  const auto values = metadata.at(name).get<vec<T>>();
  lib_assert(values.size() == expected_size,
             str{"invalid metadata array size: "} + name);
  return values;
}

Layout read_layout(const filepath_t& prefix) {
  const filepath_t path{prefix.string() + ".meta.json"};
  std::ifstream input(path);
  lib_assert(input.good(), "cannot open index metadata: " + path.string());
  nlohmann::json metadata;
  input >> metadata;

  Layout layout;
  layout.dim = metadata.at("dim").get<u32>();
  layout.degree = metadata.at("R").get<u32>();
  layout.shards = metadata.at("num_memory_nodes").get<u32>();
  layout.shard_bits = metadata.at("hot_graph_shard_bits").get<u32>();
  layout.node_bytes = metadata.at("node_size").get<u32>();
  layout.entry_bytes = metadata.at("hot_graph_entry_size").get<u32>();
  layout.dynamic_record_bytes =
    metadata.at("hot_graph_dynamic_record_bytes").get<u32>();
  layout.nodes = metadata.at("num_vectors").get<u64>();
  layout.build_fingerprint =
    metadata.at("index_build_fingerprint").get<u64>();
  layout.dtype = parse_vector_dtype(
    metadata.at("vector_data_type").get<str>());
  layout.counts = required_array<u64>(
    metadata, "hot_graph_entry_counts", layout.shards);
  layout.offsets = required_array<u64>(
    metadata, "hot_graph_offsets", layout.shards);
  layout.dynamic_offsets = required_array<u64>(
    metadata, "hot_graph_dynamic_base_offsets", layout.shards);
  layout.shard_fingerprints = required_array<u64>(
    metadata, "shard_build_fingerprints", layout.shards);

  const u32 provisional = std::min<u32>(
    15, std::max<u32>(2, (layout.degree + 15) / 16));
  lib_assert(layout.dim > 0 && layout.degree > 0 &&
               layout.degree <= kMaxSupportedGraphDegree &&
               layout.shards > 0 &&
               layout.shards <= RemotePtr::MEMORY_NODE_MASK + 1 &&
               layout.shard_bits ==
                 vamana::hot_graph::shard_bits_for(layout.shards) &&
               layout.node_bytes > 0 &&
               layout.node_bytes % RemotePtr::OFFSET_ALIGNMENT == 0 &&
               layout.entry_bytes ==
                 vamana::hot_graph::kTaggedNeighborBaseOffset +
                   (layout.degree + provisional) *
                     vamana::hot_graph::kCompactPointerBytes,
             "unsupported or invalid graph layout");
  u64 total = 0;
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    lib_assert(layout.counts[shard] > 0 &&
                 layout.shard_fingerprints[shard] != 0 &&
                 layout.offsets[shard] + layout.counts[shard] *
                   layout.entry_bytes <= layout.dynamic_offsets[shard],
               "invalid graph shard range");
    total += layout.counts[shard];
  }
  lib_assert(total == layout.nodes && layout.build_fingerprint != 0,
             "graph shard counts do not match metadata");
  return layout;
}

filepath_t shard_path(const Layout& layout,
                      const vec<filepath_t>& prefixes,
                      u32 shard, const char* extension) {
  return filepath_t{prefixes[shard].string() + "_node" +
                    std::to_string(shard + 1) + "_of" +
                    std::to_string(layout.shards) + "." + extension};
}

struct PhysicalMap {
  vec<u64> shard_offsets;
  vec<u32> inverse_ids;
  vec<u64> id_to_raw;
};

PhysicalMap import_idmaps(const Layout& layout,
                          const vec<filepath_t>& prefixes) {
  PhysicalMap map;
  map.shard_offsets.resize(layout.shards + 1, 0);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    map.shard_offsets[shard + 1] =
      map.shard_offsets[shard] + layout.counts[shard];
  }
  map.inverse_ids.assign(layout.nodes,
                         std::numeric_limits<u32>::max());
  map.id_to_raw.assign(layout.nodes, 0);

  for (u32 owner = 0; owner < layout.shards; ++owner) {
    const filepath_t path = shard_path(layout, prefixes, owner, "idmap");
    std::ifstream input(path, std::ios::binary);
    lib_assert(input.good(), "cannot open idmap: " + path.string());
    vamana::idmap::Header header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    lib_assert(input.gcount() == sizeof(header),
               "truncated idmap header: " + path.string());
    vamana::idmap::ValidationContext expected;
    expected.build_fingerprint = layout.build_fingerprint;
    expected.owner_shard_fingerprint = layout.shard_fingerprints[owner];
    expected.node_base_offset = vamana::hot_graph::kNodeBaseOffset;
    expected.owner_shard = owner;
    expected.shard_count = layout.shards;
    expected.node_size = layout.node_bytes;
    expected.id_offset = VamanaNode::offset_id();
    expected.generation_offset = VamanaNode::offset_generation();
    expected.slot_incarnation_offset =
      VamanaNode::offset_slot_incarnation();
    expected.static_entry_counts = layout.counts;
    lib_assert(vamana::idmap::valid_header(
                 header, std::filesystem::file_size(path), expected),
               "invalid idmap header: " + path.string());
    const bool valid = vamana::idmap::read_validated_payload(
      input, header, expected, [&](const vamana::idmap::Entry& entry) {
        const RemotePtr pointer{entry.rptr_raw};
        const u32 shard = pointer.memory_node();
        const u64 slot = (pointer.byte_offset() -
                          vamana::hot_graph::kNodeBaseOffset) /
                         layout.node_bytes;
        const u64 physical = map.shard_offsets[shard] + slot;
        lib_assert(entry.id < layout.nodes && physical < layout.nodes &&
                     map.id_to_raw[entry.id] == 0 &&
                     map.inverse_ids[physical] ==
                       std::numeric_limits<u32>::max(),
                   "duplicate or out-of-range idmap entry");
        map.id_to_raw[entry.id] = entry.rptr_raw;
        map.inverse_ids[physical] = entry.id;
        return true;
      });
    lib_assert(valid, "invalid idmap payload: " + path.string());
    std::cerr << "imported idmap shard " << (owner + 1) << "/"
              << layout.shards << "\n";
  }
  for (size_t index = 0; index < layout.nodes; ++index) {
    lib_assert(map.inverse_ids[index] != std::numeric_limits<u32>::max() &&
                 map.id_to_raw[index] != 0,
               "idmap does not cover every graph node");
  }
  return map;
}

u32 inverse_id(const Layout& layout, const PhysicalMap& map,
               const RemotePtr& pointer) {
  lib_assert(pointer.is_static() && pointer.is_well_formed(),
             "graph edge is not a static base pointer");
  const u32 shard = pointer.memory_node();
  lib_assert(shard < layout.shards, "graph edge shard is out of range");
  const u64 relative = pointer.byte_offset() -
                       vamana::hot_graph::kNodeBaseOffset;
  lib_assert(relative % layout.node_bytes == 0,
             "graph edge has an unaligned node offset");
  const u64 slot = relative / layout.node_bytes;
  lib_assert(slot < layout.counts[shard], "graph edge slot is out of range");
  const u64 physical = map.shard_offsets[shard] + slot;
  const u32 id = map.inverse_ids[physical];
  lib_assert(id < layout.nodes, "graph edge resolves to an invalid ID");
  return id;
}

void import_graph(VamanaGraph& graph, const Layout& layout,
                  const vec<filepath_t>& prefixes, const PhysicalMap& map,
                  u32 threads) {
  ProgressReporter progress{"Importing graph entries", layout.nodes};
  parallel_for(size_t{0}, layout.shards, threads, [&](size_t shard, size_t) {
    const filepath_t path = shard_path(layout, prefixes, shard, "dat");
    std::ifstream input(path, std::ios::binary);
    lib_assert(input.good(), "cannot open graph shard: " + path.string());
    input.seekg(static_cast<std::streamoff>(layout.offsets[shard]));
    lib_assert(input.good(), "cannot seek graph entries: " + path.string());
    vec<byte_t> entry(layout.entry_bytes);
    for (u64 slot = 0; slot < layout.counts[shard]; ++slot) {
      input.read(reinterpret_cast<char*>(entry.data()), entry.size());
      lib_assert(input.good(), "truncated graph entry: " + path.string());
      const u16 expected = vamana::hot_graph::load_u16_le(entry.data() + 2);
      const u16 actual = vamana::hot_graph::checksum16(
        entry.data(), layout.entry_bytes);
      lib_assert(expected == actual && (entry[1] & 0x0eu) == 0,
                 "invalid graph entry checksum or flags");
      const u8 degree = entry[0];
      lib_assert(degree <= layout.degree,
                 "graph entry degree exceeds R");
      const u32 source = map.inverse_ids[map.shard_offsets[shard] + slot];
      lib_assert(source < layout.nodes, "invalid source ID during import");
      const size_t base = graph.offset(source);
      for (u8 index = 0; index < degree; ++index) {
        const RemotePtr pointer = vamana::hot_graph::decode_remote_ptr(
          entry.data() + vamana::hot_graph::neighbor_offset(index),
          layout.shard_bits);
        const u32 target = inverse_id(layout, map, pointer);
        lib_assert(target != source, "graph contains a self edge");
        graph.neighbors[base + index] = target;
      }
      graph.degrees[source] = degree;
      progress.increment();
    }
  });
  progress.finish();
}

double reciprocal_ratio(VamanaGraph& graph, size_t samples) {
  if (samples == 0) return 0.0;
  vec<u32> neighbors;
  size_t edges = 0;
  size_t reciprocal = 0;
  for (size_t index = 0; index < samples; ++index) {
    const u32 source = static_cast<u32>(
      (static_cast<u64>(index) * graph.num_nodes) / samples);
    graph.copy_neighbors(source, neighbors);
    for (const u32 target : neighbors) {
      ++edges;
      reciprocal += graph.contains_neighbor_unlocked(target, source);
    }
  }
  const double ratio = edges == 0 ? 0.0 :
    static_cast<double>(reciprocal) / static_cast<double>(edges);
  std::cerr << "reciprocal sample: sources=" << samples
            << " edges=" << edges << " reciprocal=" << reciprocal
            << " ratio=" << ratio << "\n";
  return ratio;
}

void save_repaired_checkpoint(const VamanaGraph& graph,
                              const Options& options) {
  GraphCheckpointMeta meta;
  meta.num_nodes = graph.num_nodes;
  meta.dim = graph.dim;
  meta.R = graph.R;
  meta.build_capacity = graph.capacity;
  meta.medoid = graph.medoid;
  meta.seed = 1234;
  meta.phase = "consolidate";
  meta.next_step = graph.num_nodes;
  meta.next_chunk = 0;
  meta.segment_size = 1;
  save_graph_checkpoint(options.checkpoint_prefix, graph, meta,
                        options.threads);
}

void write_hot_graph(VamanaGraph& graph, const Layout& layout,
                     const vec<filepath_t>& prefixes,
                     const PhysicalMap& map, u32 threads) {
  ProgressReporter progress{"Writing repaired graph entries", layout.nodes};
  parallel_for(size_t{0}, layout.shards, threads, [&](size_t shard, size_t) {
    const filepath_t path = shard_path(layout, prefixes, shard, "dat");
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    lib_assert(file.good(), "cannot open graph shard for repair: " + path.string());
    file.seekp(static_cast<std::streamoff>(layout.offsets[shard]));
    lib_assert(file.good(), "cannot seek repaired graph entries");
    vec<byte_t> entry(layout.entry_bytes);
    vec<RemotePtr> neighbors(layout.degree);
    vec<u32> ids;
    for (u64 slot = 0; slot < layout.counts[shard]; ++slot) {
      const u32 source = map.inverse_ids[map.shard_offsets[shard] + slot];
      graph.copy_neighbors(source, ids);
      lib_assert(ids.size() <= layout.degree,
                 "repaired graph degree exceeds R");
      std::fill(neighbors.begin(), neighbors.end(), RemotePtr{});
      for (size_t index = 0; index < ids.size(); ++index) {
        neighbors[index] = RemotePtr{map.id_to_raw[ids[index]]};
      }
      VamanaNode::encode_hot_graph_entry(
        entry.data(), static_cast<u8>(ids.size()), neighbors.data(),
        ids.size(), layout.shard_bits, 0);
      file.write(reinterpret_cast<const char*>(entry.data()), entry.size());
      lib_assert(file.good(), "failed writing repaired graph entry");
      progress.increment();
    }
    file.flush();
    lib_assert(file.good(), "failed flushing repaired graph shard");
  });
  progress.finish();
}

Options parse_options(int argc, char** argv) {
  namespace po = boost::program_options;
  Options options;
  vec<str> shard_prefixes;
  po::options_description description{"Repair Vamana reverse edges"};
  description.add_options()
    ("help,h", "Show help")
    ("data-path", po::value<filepath_t>(&options.data_path)->required())
    ("index-prefix", po::value<filepath_t>(&options.index_prefix)->required())
    ("shard-prefix", po::value<vec<str>>(&shard_prefixes)->multitoken())
    ("medoid", po::value<size_t>(&options.medoid)->required())
    ("threads", po::value<u32>(&options.threads)->default_value(options.threads))
    ("beam-width", po::value<u32>(&options.beam_width)->default_value(options.beam_width))
    ("reverse-slots", po::value<u32>(&options.reverse_slots)->default_value(options.reverse_slots))
    ("max-vectors", po::value<size_t>(&options.max_vectors)->default_value(options.max_vectors))
    ("vector-data-type", po::value<str>(&options.vector_data_type)->default_value(options.vector_data_type))
    ("query-path", po::value<filepath_t>(&options.query_path))
    ("groundtruth-path", po::value<filepath_t>(&options.groundtruth_path))
    ("sample-sources", po::value<size_t>(&options.sample_sources)->default_value(options.sample_sources))
    ("consolidation-chunk-bytes", po::value<size_t>(&options.consolidation_chunk_bytes)->default_value(0))
    ("apply", po::bool_switch(&options.apply))
    ("resume-repaired", po::bool_switch(&options.resume_repaired))
    ("gate-dir", po::value<filepath_t>(&options.gate_dir))
    ("checkpoint-prefix", po::value<filepath_t>(&options.checkpoint_prefix));
  po::variables_map variables;
  po::store(po::parse_command_line(argc, argv, description), variables);
  if (variables.count("help")) {
    std::cout << description << '\n';
    std::exit(EXIT_SUCCESS);
  }
  po::notify(variables);
  for (const str& value : shard_prefixes) {
    options.shard_prefixes.emplace_back(value);
  }
  return options;
}

int run(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  const Layout layout = read_layout(options.index_prefix);
  lib_assert(options.shard_prefixes.empty() ||
               options.shard_prefixes.size() == layout.shards,
             "provide exactly one shard prefix per physical shard");
  const vec<filepath_t> prefixes = options.shard_prefixes.empty()
    ? vec<filepath_t>(layout.shards, options.index_prefix)
    : options.shard_prefixes;
  if (options.apply) {
    lib_assert(!options.gate_dir.empty() && !options.checkpoint_prefix.empty(),
               "--apply requires --gate-dir and --checkpoint-prefix");
    lib_assert(!std::filesystem::exists(
                 std::filesystem::path{options.gate_dir} / "CONTINUE"),
               "repair gate CONTINUE already exists");
  }

  VamanaBuildConfig config;
  config.data_path = options.data_path;
  config.vector_data_type = options.vector_data_type;
  config.max_vectors = options.max_vectors;
  config.threads = options.threads;
  config.beam_width = options.beam_width;
  config.query_path = options.query_path;
  config.groundtruth_path = options.groundtruth_path;
  const Dataset dataset = read_dataset(config);
  lib_assert(dataset.size() == layout.nodes && dataset.dim == layout.dim &&
               dataset.dtype == layout.dtype,
             "base dataset does not match graph metadata");
  preflight_optional_recall_inputs(dataset, config);

  VamanaNode::init_static_storage(layout.dim, layout.degree, layout.dtype);
  lib_assert(VamanaNode::total_size() == layout.node_bytes,
             "Vamana node layout mismatch");
  VamanaNode::disable_hot_graph();
  VamanaNode::configure_hot_graph(
    layout.offsets, layout.counts, layout.entry_bytes, layout.shard_bits,
    layout.dynamic_offsets, layout.dynamic_record_bytes, layout.node_bytes);
  lib_assert(VamanaNode::hot_graph_entry_size() == layout.entry_bytes,
             "Vamana hot graph layout mismatch");

  const PhysicalMap map = import_idmaps(layout, prefixes);
  VamanaGraph graph;
  graph.init(layout.nodes, layout.dim, layout.degree, 1 << 24,
             options.threads);
  graph.medoid = options.medoid;
  lib_assert(graph.medoid < graph.num_nodes, "medoid is out of range");

  if (options.resume_repaired) {
    const GraphCheckpointMeta meta =
      read_graph_checkpoint_meta(options.checkpoint_prefix);
    lib_assert(meta.num_nodes == layout.nodes && meta.R == layout.degree &&
                 meta.dim == layout.dim && meta.medoid == options.medoid,
               "repaired checkpoint does not match index");
    load_graph_checkpoint_arrays(
      options.checkpoint_prefix, graph, meta, options.threads);
  } else {
    import_graph(graph, layout, prefixes, map, options.threads);
    reciprocal_ratio(graph, options.sample_sources);
    run_optional_recall_check(graph, dataset, config);
    repair_reverse_edges(graph, dataset, options.threads,
                         options.reverse_slots,
                         options.consolidation_chunk_bytes);
  }

  reciprocal_ratio(graph, options.sample_sources);
  run_optional_recall_check(graph, dataset, config);
  if (!options.apply) return EXIT_SUCCESS;

  save_repaired_checkpoint(graph, options);
  std::filesystem::create_directories(options.gate_dir);
  const filepath_t ready = (std::filesystem::path{options.gate_dir} /
                            "READY").string();
  {
    std::ofstream marker(ready);
    marker << "repaired graph recall completed\n";
  }
  std::cerr << "REPAIR_GATE_READY " << ready << "\n";
  const filepath_t proceed = (std::filesystem::path{options.gate_dir} /
                              "CONTINUE").string();
  while (!std::filesystem::exists(proceed)) {
    std::this_thread::sleep_for(std::chrono::seconds{10});
  }
  write_hot_graph(graph, layout, prefixes, map, options.threads);
  std::cerr << "REPAIR_WRITE_COMPLETE\n";
  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace tools::vamana_offline

int main(int argc, char** argv) {
  try {
    return tools::vamana_offline::run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "vamana_graph_repair: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
