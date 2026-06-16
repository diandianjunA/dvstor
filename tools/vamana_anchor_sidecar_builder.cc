#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>

#include <boost/program_options.hpp>
#include <library/utils.hh>

#include "common/index_path.hh"
#include "common/vector_dtype.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/anchor_index.hh"
#include "vamana/idmap.hh"

namespace po = boost::program_options;

namespace {

struct Sample {
  u64 priority{};
  vamana::idmap::Entry entry;

  bool operator<(const Sample& other) const { return priority < other.priority; }
};

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

struct Config {
  filepath_t index_prefix;
  u32 anchors_per_shard{4096};
  u64 seed{1234};
};

Config parse_config(int argc, char** argv) {
  Config config;
  po::options_description options{"Anchor sidecar builder options"};
  options.add_options()
    ("help,h", "Show help message")
    ("index-prefix", po::value<filepath_t>(&config.index_prefix)->required(),
     "Existing index prefix without the shard suffix")
    ("anchors-per-shard", po::value<u32>(&config.anchors_per_shard)->default_value(4096),
     "Number of deterministic representative anchors per physical shard")
    ("seed", po::value<u64>(&config.seed)->default_value(1234), "Sampling seed");
  po::variables_map values;
  po::store(po::parse_command_line(argc, argv, options), values);
  if (values.count("help")) {
    std::cout << options << std::endl;
    std::exit(EXIT_SUCCESS);
  }
  po::notify(values);
  if (config.anchors_per_shard == 0) lib_failure("--anchors-per-shard must be > 0");
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_config(argc, argv);
    const filepath_t metadata_path{config.index_prefix.string() + ".meta.json"};
    std::ifstream metadata_input(metadata_path);
    lib_assert(metadata_input.good(), "missing index metadata: " + metadata_path.string());
    nlohmann::json metadata;
    metadata_input >> metadata;

    const u32 dim = metadata.at("dim").get<u32>();
    const u32 shard_count = metadata.at("num_memory_nodes").get<u32>();
    const u32 vector_offset = metadata.at("vector_offset").get<u32>();
    const VectorDType dtype = parse_vector_dtype(metadata.at("vector_data_type").get<str>());
    const size_t vector_bytes = metadata.at("vector_bytes").get<size_t>();
    lib_assert(metadata.value("distance", str{"l2"}) == "l2",
               "anchor routing currently supports L2 indexes only");
    lib_assert(vector_bytes == vector_dtype_bytes(dtype, dim), "metadata vector layout mismatch");
    lib_assert(shard_count > 0, "metadata has no memory nodes");

    bool have_all_shard_files = true;
    for (u32 shard = 0; shard < shard_count; ++shard) {
      have_all_shard_files &= std::filesystem::exists(
        index_path::shard_file(config.index_prefix, shard + 1, shard_count));
    }
    const filepath_t data_file{metadata.at("data_file").get<str>()};
    lib_assert(have_all_shard_files || std::filesystem::exists(data_file),
               "anchor generation needs either local index shards or the source dataset: " +
               data_file.string());

    vec<std::priority_queue<Sample>> samples(shard_count);
    constexpr size_t kReadBatch = 1u << 16;
    vec<vamana::idmap::Entry> entries(kReadBatch);
    u64 scanned = 0;
    for (u32 owner = 0; owner < shard_count; ++owner) {
      const filepath_t idmap_path = index_path::owner_idmap_file(
        config.index_prefix, owner + 1, shard_count);
      std::ifstream input(idmap_path, std::ios::binary);
      lib_assert(input.good(), "missing idmap sidecar: " + idmap_path.string());
      vamana::idmap::Header header;
      input.read(reinterpret_cast<char*>(&header), sizeof(header));
      lib_assert(input.good() && header.magic == vamana::idmap::kMagic &&
                   header.version == vamana::idmap::kVersion &&
                   header.owner_shard == owner && header.shard_count == shard_count,
                 "invalid idmap sidecar: " + idmap_path.string());
      u64 remaining = header.entry_count;
      while (remaining != 0) {
        const size_t count = static_cast<size_t>(std::min<u64>(remaining, entries.size()));
        input.read(reinterpret_cast<char*>(entries.data()),
                   static_cast<std::streamsize>(count * sizeof(vamana::idmap::Entry)));
        lib_assert(input.good(), "truncated idmap sidecar: " + idmap_path.string());
        for (size_t i = 0; i < count; ++i) {
          const auto& entry = entries[i];
          const RemotePtr ptr{entry.rptr_raw};
          lib_assert(ptr.memory_node() < shard_count, "idmap pointer has invalid shard");
          const u64 priority = mix64(static_cast<u64>(entry.id) ^ config.seed);
          auto& heap = samples[ptr.memory_node()];
          if (heap.size() < config.anchors_per_shard) {
            heap.push(Sample{priority, entry});
          } else if (priority < heap.top().priority) {
            heap.pop();
            heap.push(Sample{priority, entry});
          }
        }
        scanned += count;
        remaining -= count;
      }
    }

    vec<vec<vamana::idmap::Entry>> selected(shard_count);
    u64 total = 0;
    for (u32 shard = 0; shard < shard_count; ++shard) {
      auto& heap = samples[shard];
      auto& shard_entries = selected[shard];
      shard_entries.reserve(heap.size());
      while (!heap.empty()) {
        shard_entries.push_back(heap.top().entry);
        heap.pop();
      }
      std::sort(shard_entries.begin(), shard_entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.rptr_raw < rhs.rptr_raw;
      });
      total += shard_entries.size();
    }

    vec<std::ifstream> shard_files(shard_count);
    for (u32 shard = 0; shard < shard_count; ++shard) {
      if (!have_all_shard_files) break;
      const filepath_t path = index_path::shard_file(config.index_prefix, shard + 1, shard_count);
      shard_files[shard].open(path, std::ios::binary);
      lib_assert(shard_files[shard].good(), "missing index shard: " + path.string());
    }
    std::ifstream dataset_file;
    u32 dataset_rows = 0;
    if (!have_all_shard_files) {
      dataset_file.open(data_file, std::ios::binary);
      lib_assert(dataset_file.good(), "index shards are unavailable and dataset is missing: " +
                                      data_file.string());
      u32 dataset_dim = 0;
      dataset_file.read(reinterpret_cast<char*>(&dataset_rows), sizeof(dataset_rows));
      dataset_file.read(reinterpret_cast<char*>(&dataset_dim), sizeof(dataset_dim));
      lib_assert(dataset_file.good() && dataset_dim == dim,
                 "dataset header does not match index metadata");
      std::cout << "index shard files unavailable; reading sampled vectors from "
                << data_file << std::endl;
    }

    auto read_vector = [&](const vamana::idmap::Entry& entry, byte_t* destination) {
      if (have_all_shard_files) {
        const RemotePtr ptr{entry.rptr_raw};
        auto& file = shard_files[ptr.memory_node()];
        file.clear();
        file.seekg(static_cast<std::streamoff>(ptr.byte_offset() + vector_offset));
        file.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(vector_bytes));
        lib_assert(file.good(), "failed to read anchor vector from index shard");
        return;
      }
      lib_assert(entry.id < dataset_rows, "idmap ID exceeds source dataset rows");
      dataset_file.clear();
      dataset_file.seekg(static_cast<std::streamoff>(sizeof(u32) * 2 +
                         static_cast<u64>(entry.id) * vector_bytes));
      dataset_file.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(vector_bytes));
      lib_assert(dataset_file.good(), "failed to read anchor vector from source dataset");
    };

    const filepath_t output_path = index_path::anchor_file(config.index_prefix);
    const filepath_t output_tmp{output_path.string() + ".tmp"};
    std::ofstream output(output_tmp, std::ios::binary | std::ios::trunc);
    lib_assert(output.good(), "failed to create anchor sidecar: " + output_tmp.string());
    vamana::anchor::Header header;
    header.dim = dim;
    header.shard_count = shard_count;
    header.vector_dtype = static_cast<u32>(dtype);
    header.vector_bytes = static_cast<u32>(vector_bytes);
    header.anchors_per_shard = config.anchors_per_shard;
    header.total_anchors = total;
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));

    vec<byte_t> raw(vector_bytes);
    vec<float> decoded(dim);
    for (u32 shard = 0; shard < shard_count; ++shard) {
      const auto& shard_entries = selected[shard];
      vamana::anchor::ShardHeader shard_header{shard, static_cast<u32>(shard_entries.size())};
      output.write(reinterpret_cast<const char*>(&shard_header), sizeof(shard_header));

      vec<float> centroid(dim, 0.0f);
      for (const auto& entry : shard_entries) {
        read_vector(entry, raw.data());
        decode_storage_vector_to_float(raw.data(), dtype, dim, decoded.data());
        for (u32 d = 0; d < dim; ++d) centroid[d] += decoded[d];
      }
      if (!shard_entries.empty()) {
        const float scale = 1.0f / static_cast<float>(shard_entries.size());
        for (float& value : centroid) value *= scale;
      }
      output.write(reinterpret_cast<const char*>(centroid.data()),
                   static_cast<std::streamsize>(centroid.size() * sizeof(float)));

      for (const auto& entry : shard_entries) {
        read_vector(entry, raw.data());
        vamana::anchor::EntryHeader anchor_entry;
        anchor_entry.rptr_raw = entry.rptr_raw;
        anchor_entry.id = entry.id;
        output.write(reinterpret_cast<const char*>(&anchor_entry), sizeof(anchor_entry));
        output.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
      }
    }
    lib_assert(output.good(), "failed to write anchor sidecar");
    output.close();
    std::filesystem::rename(output_tmp, output_path);

    vamana::anchor::Index validation_index;
    str validation_error;
    lib_assert(validation_index.load(config.index_prefix, dim, shard_count, &validation_error),
               "failed to validate anchor sidecar: " + validation_error);
    lib_assert(validation_index.anchor_count() == total, "validated anchor count mismatch");
    vec<float> probe(dim, 0.0f);
    const u32 probe_owner = shard_count > 1 ? 1 : 0;
    const auto probe_route = validation_index.route(probe, std::min<u32>(4, config.anchors_per_shard),
                                                    probe_owner);
    lib_assert(probe_route.owner == probe_owner && !probe_route.hints.empty(),
               "anchor route validation failed");

    metadata["anchor_format"] = "owner_anchor_v1";
    metadata["anchor_count_per_shard"] = config.anchors_per_shard;
    const filepath_t metadata_tmp{metadata_path.string() + ".anchor.tmp"};
    {
      std::ofstream metadata_output(metadata_tmp, std::ios::trunc);
      lib_assert(metadata_output.good(), "failed to write temporary metadata");
      metadata_output << std::setw(2) << metadata << std::endl;
    }
    std::filesystem::rename(metadata_tmp, metadata_path);

    std::cout << "anchor sidecar: " << output_path
              << " scanned=" << scanned
              << " anchors=" << total
              << " memory_estimate_bytes="
              << total * (static_cast<u64>(dim) * sizeof(float) + sizeof(RemotePtr))
              << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "anchor sidecar build failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}
