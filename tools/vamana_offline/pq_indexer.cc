#include "tools/vamana_offline/pq_indexer.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <faiss/VectorTransform.h>
#include <faiss/impl/ProductQuantizer.h>
#include <omp.h>

#include "common/index_path.hh"
#include "common/vector_dtype.hh"
#include "gpu_search/index_format.hh"
#include "gpu_search/pq_index.hh"
#include "nlohmann/json.hh"

namespace tools::vamana_offline {
namespace {

struct Layout {
  u32 dim{};
  u32 shards{};
  u32 node_bytes{};
  u32 vector_offset{};
  u32 vector_bytes{};
  VectorDType dtype{VectorDType::float32};
  u64 node_count{};
  vec<u64> counts;
  vec<u64> dynamic_offsets;
};

Layout parse_layout(const nlohmann::json& metadata) {
  if (metadata.value("schema_version", 0u) != 14 ||
      metadata.value("node_layout", str{}) != "plain" ||
      metadata.value("storage_format", str{}) != "vamana_compact_v1" ||
      metadata.value("distance", str{"l2"}) != "l2") {
    throw std::runtime_error(
      "PQ indexer requires a schema-14 plain compact L2 index");
  }
  Layout layout;
  layout.dim = metadata.at("dim").get<u32>();
  layout.shards = metadata.at("num_memory_nodes").get<u32>();
  layout.node_bytes = metadata.at("node_size").get<u32>();
  layout.vector_offset = metadata.at("vector_offset").get<u32>();
  layout.vector_bytes = metadata.at("vector_bytes").get<u32>();
  layout.dtype = parse_vector_dtype(metadata.at("vector_data_type").get<str>());
  layout.node_count = metadata.at("num_vectors").get<u64>();
  layout.counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
  layout.dynamic_offsets =
    metadata.at("hot_graph_dynamic_base_offsets").get<vec<u64>>();
  if (layout.dim == 0 || layout.shards == 0 || layout.node_bytes == 0 ||
      layout.vector_offset + layout.vector_bytes > layout.node_bytes ||
      vector_dtype_bytes(layout.dtype, layout.dim) != layout.vector_bytes ||
      layout.counts.size() != layout.shards ||
      layout.dynamic_offsets.size() != layout.shards) {
    throw std::runtime_error("schema-14 index metadata contains an invalid node layout");
  }
  u64 total = 0;
  for (u64 count : layout.counts) total += count;
  if (total != layout.node_count || total == 0) {
    throw std::runtime_error("schema-14 index metadata contains an invalid node count");
  }
  return layout;
}

u64 mix64(u64 value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

vec<f32> sample_training_vectors(const filepath_t& prefix, const Layout& layout,
                                 u32 requested, u64 seed) {
  const u64 sample_count = std::min<u64>(requested, layout.node_count);
  if (sample_count < gpu_search::pq::kCentroidsPerSubquantizer) {
    throw std::runtime_error("PQ training requires at least 256 samples");
  }
  vec<u64> ordinal_bases(layout.shards + 1, 0);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    ordinal_bases[shard + 1] = ordinal_bases[shard] + layout.counts[shard];
  }
  vec<std::ifstream> inputs(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    const filepath_t path = index_path::shard_file(prefix, shard + 1, layout.shards);
    inputs[shard].open(path, std::ios::binary);
    if (!inputs[shard].good()) throw std::runtime_error("missing index shard: " + path.string());
  }
  vec<f32> samples(static_cast<size_t>(sample_count) * layout.dim);
  vec<byte_t> raw(layout.vector_bytes);
  const u64 phase = mix64(seed) % layout.node_count;
  for (u64 sample = 0; sample < sample_count; ++sample) {
    const u64 ordinal = (phase + sample * layout.node_count / sample_count) %
      layout.node_count;
    const auto upper = std::upper_bound(ordinal_bases.begin(), ordinal_bases.end(), ordinal);
    const u32 shard = static_cast<u32>(upper - ordinal_bases.begin() - 1);
    const u64 slot = ordinal - ordinal_bases[shard];
    inputs[shard].seekg(static_cast<std::streamoff>(
      gpu_search::format::kNodeBaseOffset + slot * layout.node_bytes +
      layout.vector_offset));
    inputs[shard].read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (static_cast<size_t>(inputs[shard].gcount()) != raw.size()) {
      throw std::runtime_error("short read while sampling PQ training vectors");
    }
    decode_storage_vector_to_float(
      raw.data(), layout.dtype, layout.dim,
      samples.data() + static_cast<size_t>(sample) * layout.dim);
  }
  return samples;
}

gpu_search::pq::Model train_model(const vec<f32>& samples, const Layout& layout,
                                  const PqIndexOptions& options) {
  if (layout.dim % options.subquantizers != 0) {
    throw std::runtime_error("dimension must be divisible by the PQ subquantizer count");
  }
  const size_t count = samples.size() / layout.dim;
  faiss::OPQMatrix opq(layout.dim, options.subquantizers);
  opq.niter = static_cast<int>(options.opq_iterations);
  opq.niter_pq = std::max<int>(1, static_cast<int>(options.pq_iterations / 4));
  opq.niter_pq_0 = static_cast<int>(options.pq_iterations);
  opq.max_train_points = count;
  opq.verbose = true;
  opq.train(static_cast<faiss::idx_t>(count), samples.data());

  vec<f32> transformed(samples.size());
  opq.apply_noalloc(static_cast<faiss::idx_t>(count), samples.data(),
                    transformed.data());
  faiss::ProductQuantizer product(layout.dim, options.subquantizers, 8);
  product.cp.niter = static_cast<int>(options.pq_iterations);
  product.cp.seed = static_cast<int>(options.seed);
  product.verbose = true;
  product.train(count, transformed.data());

  gpu_search::pq::Model model;
  model.dim = layout.dim;
  model.subquantizers = options.subquantizers;
  model.rotation = opq.A;
  model.centroids = product.centroids;
  std::string error;
  if (!gpu_search::pq::validate(model, &error)) throw std::runtime_error(error);
  return model;
}

void encode_shard(const filepath_t& prefix, const Layout& layout,
                  const gpu_search::pq::Model& model,
                  const PqIndexOptions& options, u32 shard,
                  const filepath_t& output_path, u64 remote_offset) {
  const filepath_t input_path = index_path::shard_file(prefix, shard + 1, layout.shards);
  std::ifstream input(input_path, std::ios::binary);
  if (!input.good()) throw std::runtime_error("missing index shard: " + input_path.string());
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output.good()) throw std::runtime_error("failed to create PQ sidecar: " + output_path.string());
  gpu_search::format::CodeHeader header;
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));

  faiss::LinearTransform transform(layout.dim, layout.dim, false);
  transform.A = model.rotation;
  transform.is_trained = true;
  transform.is_orthonormal = true;
  faiss::ProductQuantizer product(layout.dim, model.subquantizers, 8);
  product.centroids = model.centroids;

  const u64 count = layout.counts[shard];
  const u32 chunk_vectors = std::max<u32>(1, options.chunk_vectors);
  vec<byte_t> nodes(static_cast<size_t>(chunk_vectors) * layout.node_bytes);
  vec<f32> decoded(static_cast<size_t>(chunk_vectors) * layout.dim);
  vec<f32> transformed(static_cast<size_t>(chunk_vectors) * layout.dim);
  vec<byte_t> codes(static_cast<size_t>(chunk_vectors) * model.code_bytes());
  u64 checksum = gpu_search::format::checksum64_initial();
  for (u64 base = 0; base < count; base += chunk_vectors) {
    const u32 batch = static_cast<u32>(std::min<u64>(chunk_vectors, count - base));
    const size_t node_bytes = static_cast<size_t>(batch) * layout.node_bytes;
    input.seekg(static_cast<std::streamoff>(
      gpu_search::format::kNodeBaseOffset + base * layout.node_bytes));
    input.read(reinterpret_cast<char*>(nodes.data()), node_bytes);
    if (static_cast<size_t>(input.gcount()) != node_bytes) {
      throw std::runtime_error("short read while encoding PQ shard " + input_path.string());
    }
#pragma omp parallel for schedule(static)
    for (i64 index = 0; index < static_cast<i64>(batch); ++index) {
      decode_storage_vector_to_float(
        nodes.data() + static_cast<size_t>(index) * layout.node_bytes +
          layout.vector_offset,
        layout.dtype, layout.dim,
        decoded.data() + static_cast<size_t>(index) * layout.dim);
    }
    if (model.has_rotation()) {
      transform.apply_noalloc(batch, decoded.data(), transformed.data());
    } else {
      std::copy(decoded.begin(), decoded.begin() + static_cast<size_t>(batch) * layout.dim,
                transformed.begin());
    }
    product.compute_codes(transformed.data(), codes.data(), batch);
    const size_t payload_bytes = static_cast<size_t>(batch) * model.code_bytes();
    output.write(reinterpret_cast<const char*>(codes.data()), payload_bytes);
    if (!output.good()) throw std::runtime_error("failed to write PQ sidecar payload");
    checksum = gpu_search::format::checksum64_update(checksum, codes.data(), payload_bytes);
    std::cerr << "\rPQ encoding shard " << (shard + 1) << "/" << layout.shards
              << ": " << (base + batch) << "/" << count << std::flush;
  }
  std::cerr << '\n';
  header.memory_node = shard;
  header.code_bytes = model.code_bytes();
  header.node_size = layout.node_bytes;
  header.entry_count = count;
  header.remote_offset = remote_offset;
  header.payload_bytes = count * model.code_bytes();
  header.model_checksum = model.checksum();
  header.payload_checksum = checksum;
  std::string error;
  if (!gpu_search::format::write_code_header(output, header, &error)) {
    throw std::runtime_error(error);
  }
  output.flush();
  if (!output.good()) throw std::runtime_error("failed to finalize PQ sidecar");
}

}  // namespace

PqIndexResult build_pq_index(const PqIndexOptions& options) {
  if (options.index_prefix.empty()) throw std::invalid_argument("index prefix is required");
  const filepath_t metadata_path{options.index_prefix.string() + ".meta.json"};
  std::ifstream metadata_input(metadata_path);
  if (!metadata_input.good()) throw std::runtime_error("missing metadata: " + metadata_path.string());
  nlohmann::json metadata;
  metadata_input >> metadata;
  const Layout layout = parse_layout(metadata);
  omp_set_num_threads(static_cast<int>(options.threads == 0
    ? std::max(1u, std::thread::hardware_concurrency()) : options.threads));

  PqIndexResult result;
  result.model_file = index_path::navigation_model_file(options.index_prefix);
  result.node_count = layout.node_count;
  result.code_files.resize(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    result.code_files[shard] = index_path::navigation_code_file(
      options.index_prefix, shard + 1, layout.shards);
  }
  if (!options.overwrite) {
    if (std::filesystem::exists(result.model_file) ||
        std::any_of(result.code_files.begin(), result.code_files.end(),
                    [](const filepath_t& path) { return std::filesystem::exists(path); })) {
      throw std::runtime_error("PQ index output already exists; pass --overwrite to replace it");
    }
  }

  gpu_search::pq::Model model;
  std::string error;
  if (!options.reuse_model.empty()) {
    if (!gpu_search::pq::read_model(options.reuse_model, model, &error)) {
      throw std::runtime_error(error);
    }
    if (model.dim != layout.dim || model.subquantizers != options.subquantizers) {
      throw std::runtime_error("reused PQ model does not match the index layout");
    }
  } else {
    const vec<f32> samples = sample_training_vectors(
      options.index_prefix, layout, options.train_samples, options.seed);
    model = train_model(samples, layout, options);
  }
  if (!gpu_search::pq::write_model(result.model_file, model, &error)) {
    throw std::runtime_error(error);
  }
  result.model_checksum = model.checksum();

  vec<u64> remote_offsets(layout.shards);
  vec<u64> region_bytes(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    remote_offsets[shard] = gpu_search::format::align_up(
      layout.dynamic_offsets[shard], 64);
    region_bytes[shard] = layout.counts[shard] * model.code_bytes();
    encode_shard(options.index_prefix, layout, model, options, shard,
                 result.code_files[shard], remote_offsets[shard]);
    result.code_bytes += region_bytes[shard];
  }

  metadata["navigation_quantizer"] = "opq_pq16";
  metadata["navigation_code_bytes"] = model.code_bytes();
  metadata["pq_subquantizers"] = model.subquantizers;
  metadata["pq_bits"] = model.bits_per_code;
  metadata["navigation_model_checksum"] = model.checksum();
  metadata["navigation_model_file"] = result.model_file.string();
  metadata["navigation_format"] = "opq_pq16_graph_v1";
  metadata["navigation_entry_points"] = options.entry_points;
  metadata["navigation_code_remote_offsets"] = remote_offsets;
  metadata["navigation_code_region_bytes"] = region_bytes;
  metadata["navigation_code_materialization"] = "storage_startup_sidecar";
  metadata["navigation_graph_source"] = "storage_compact_graph";
  metadata["navigation_execution"] = "gpu_beam_v1";
  {
    std::ofstream output(metadata_path, std::ios::trunc);
    output << std::setw(2) << metadata << '\n';
    if (!output.good()) throw std::runtime_error("failed to update index metadata");
  }

  gpu_search::format::View manifest;
  bool used_anchors = false;
  if (!gpu_search::format::synthesize_distributed_view(
        options.index_prefix, manifest,
        {.entry_points = options.entry_points, .seed = options.seed},
        &used_anchors, &error)) {
    throw std::runtime_error(error);
  }
  std::cerr << "PQ index ready: nodes=" << result.node_count
            << " code_bytes=" << result.code_bytes
            << " model_checksum=" << result.model_checksum
            << " entry_points=" << manifest.entry_points.size()
            << " anchors=" << (used_anchors ? "yes" : "no") << '\n';
  return result;
}

}  // namespace tools::vamana_offline
