#include "tools/vamana_offline/pq_indexer.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

#include <faiss/VectorTransform.h>
#include <faiss/impl/ProductQuantizer.h>
#include <omp.h>

#include "common/constants.hh"
#include "common/index_path.hh"
#include "common/vector_dtype.hh"
#include "gpu_search/index_format.hh"
#include "gpu_search/pq_index.hh"
#include "nlohmann/json.hh"
#include "remote_pointer.hh"
#include "vamana/centroid_state.hh"
#include "vamana/vamana_node.hh"

namespace tools::vamana_offline {
namespace {

struct Layout {
  u32 dim{};
  u32 graph_degree{};
  u32 shards{};
  u32 node_bytes{};
  u32 vector_offset{};
  u32 vector_bytes{};
  VectorDType dtype{VectorDType::float32};
  u64 build_fingerprint{};
  u64 node_count{};
  vec<u64> counts;
  vec<u64> dynamic_offsets;
  vec<u64> shard_fingerprints;
};

struct PersistentLayout {
  vec<u64> code_offsets;
  vec<u64> code_region_bytes;
  vec<u64> control_offsets;
  vec<u64> dynamic_node_offsets;
  u32 dynamic_code_offset{};
  u32 dynamic_record_bytes{};
};

Layout parse_layout(const nlohmann::json& metadata) {
  const u32 schema_version = metadata.value("schema_version", 0u);
  if (schema_version != 15 ||
      metadata.value("node_layout", str{}) != "plain" ||
      metadata.value("storage_format", str{}) != "vamana_tagged_v2" ||
      metadata.value("remote_ptr_format", str{}) !=
        "tagged_inc24_shard6_off34x16_v1" ||
      metadata.value("centroid_state_format", str{}) !=
        "physical_shard_centroid_v2_bound" ||
      metadata.value("index_build_fingerprint", 0ull) == 0 ||
      metadata.value("distance", str{"l2"}) != "l2") {
    throw std::runtime_error(
      "PQ indexer requires a schema-15 tagged L2 index");
  }
  Layout layout;
  layout.dim = metadata.at("dim").get<u32>();
  layout.graph_degree = metadata.at("R").get<u32>();
  layout.shards = metadata.at("num_memory_nodes").get<u32>();
  layout.node_bytes = metadata.at("node_size").get<u32>();
  layout.vector_offset = metadata.at("vector_offset").get<u32>();
  layout.vector_bytes = metadata.at("vector_bytes").get<u32>();
  layout.dtype = parse_vector_dtype(metadata.at("vector_data_type").get<str>());
  layout.build_fingerprint =
    metadata.at("index_build_fingerprint").get<u64>();
  layout.node_count = metadata.at("num_vectors").get<u64>();
  layout.counts = metadata.at("hot_graph_entry_counts").get<vec<u64>>();
  layout.dynamic_offsets =
    metadata.at("hot_graph_dynamic_base_offsets").get<vec<u64>>();
  layout.shard_fingerprints =
    metadata.at("shard_build_fingerprints").get<vec<u64>>();
  if (layout.dim == 0 || layout.graph_degree == 0 ||
      layout.graph_degree > kMaxSupportedGraphDegree ||
      layout.shards == 0 ||
      layout.shards > RemotePtr::MEMORY_NODE_MASK + 1 ||
      layout.node_bytes == 0 ||
      layout.vector_offset + layout.vector_bytes > layout.node_bytes ||
      vector_dtype_bytes(layout.dtype, layout.dim) != layout.vector_bytes ||
      layout.counts.size() != layout.shards ||
      layout.dynamic_offsets.size() != layout.shards ||
      layout.shard_fingerprints.size() != layout.shards ||
      std::find(layout.shard_fingerprints.begin(),
                layout.shard_fingerprints.end(), 0) !=
        layout.shard_fingerprints.end()) {
    throw std::runtime_error("schema-15 index metadata contains an invalid node layout");
  }
  u64 total = 0;
  for (u64 count : layout.counts) total += count;
  if (total != layout.node_count || total == 0) {
    throw std::runtime_error("schema-15 index metadata contains an invalid node count");
  }
  return layout;
}

void validate_shard_fingerprint(std::ifstream& input,
                                const filepath_t& path,
                                u64 expected) {
  u64 actual = 0;
  input.seekg(static_cast<std::streamoff>(
    vamana::centroid_state::kShardFingerprintOffset));
  input.read(reinterpret_cast<char*>(&actual), sizeof(actual));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(actual)) ||
      expected == 0 || actual != expected) {
    throw std::runtime_error(
      "index shard does not belong to metadata build: " + path.string());
  }
}

PersistentLayout make_persistent_layout(const nlohmann::json& metadata,
                                        const Layout& layout,
                                        u32 code_bytes) {
  if (code_bytes == 0 || code_bytes > layout.dim ||
      metadata.value("pq_bits", 8u) != 8u) {
    throw std::runtime_error("invalid OPQ/PQ code layout");
  }
  const u32 dynamic_hot_offset = metadata.at("hot_graph_dynamic_hot_offset").get<u32>();
  const u32 graph_entry_bytes = metadata.at("hot_graph_entry_size").get<u32>();
  const u64 dynamic_code_offset = static_cast<u64>(dynamic_hot_offset) + graph_entry_bytes;
  const u64 dynamic_record_bytes = gpu_search::format::align_up(
    dynamic_code_offset + VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES +
      code_bytes, 16);
  if (dynamic_code_offset > std::numeric_limits<u32>::max() ||
      dynamic_record_bytes == 0 ||
      dynamic_record_bytes > std::numeric_limits<u32>::max()) {
    throw std::runtime_error("persistent dynamic record layout overflows");
  }

  PersistentLayout result;
  result.code_offsets.resize(layout.shards);
  result.code_region_bytes.resize(layout.shards);
  result.control_offsets.resize(layout.shards);
  result.dynamic_node_offsets.resize(layout.shards);
  result.dynamic_code_offset = static_cast<u32>(dynamic_code_offset);
  result.dynamic_record_bytes = static_cast<u32>(dynamic_record_bytes);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    const u64 control_offset = gpu_search::format::align_up(
      layout.dynamic_offsets[shard], 64);
    if (control_offset == 0 ||
        control_offset > std::numeric_limits<u64>::max() -
          gpu_search::format::kStorageControlBytes) {
      throw std::runtime_error("persistent storage control layout overflows");
    }
    const u64 code_offset = control_offset + gpu_search::format::kStorageControlBytes;
    if (layout.counts[shard] >
        std::numeric_limits<u64>::max() / code_bytes) {
      throw std::runtime_error("persistent PQ code region overflows");
    }
    const u64 region_bytes = layout.counts[shard] * code_bytes;
    if (code_offset > std::numeric_limits<u64>::max() - region_bytes) {
      throw std::runtime_error("persistent PQ code offset overflows");
    }
    const u64 code_end = code_offset + region_bytes;
    const u64 relative_end = code_end - layout.dynamic_offsets[shard];
    const u64 aligned_end = gpu_search::format::align_up(
      relative_end, dynamic_record_bytes);
    if (aligned_end == 0 || layout.dynamic_offsets[shard] >
        std::numeric_limits<u64>::max() - aligned_end) {
      throw std::runtime_error("persistent dynamic node region overflows");
    }
    result.control_offsets[shard] = control_offset;
    result.code_offsets[shard] = code_offset;
    result.code_region_bytes[shard] = region_bytes;
    result.dynamic_node_offsets[shard] =
      layout.dynamic_offsets[shard] + aligned_end;
  }
  return result;
}

void apply_persistent_layout(nlohmann::json& metadata,
                             const PersistentLayout& layout,
                             u32 code_bytes) {
  metadata["schema_version"] = gpu_search::format::kMetadataSchemaVersion;
  metadata["navigation_code_bytes"] = code_bytes;
  metadata["navigation_code_remote_offsets"] = layout.code_offsets;
  metadata["navigation_code_region_bytes"] = layout.code_region_bytes;
  metadata["storage_control_remote_offsets"] = layout.control_offsets;
  metadata["dynamic_node_base_offsets"] = layout.dynamic_node_offsets;
  metadata["hot_graph_dynamic_record_bytes"] = layout.dynamic_record_bytes;
  metadata["allocation_size"] = layout.dynamic_record_bytes;
  metadata["dynamic_navigation_code_offset"] = layout.dynamic_code_offset;
  metadata["dynamic_navigation_code_validation_bytes"] =
    VamanaNode::DYNAMIC_CODE_INCARNATION_BYTES;
  metadata["navigation_code_materialization"] = "storage_startup_sidecar";
  metadata["navigation_graph_source"] = "storage_compact_graph";
  metadata["navigation_execution"] = "gpu_beam_v1";
}

void write_metadata_atomic(const filepath_t& path,
                           const nlohmann::json& metadata) {
  const filepath_t temporary{path.string() + ".schema15.tmp"};
  {
    std::ofstream output(temporary, std::ios::trunc);
    output << std::setw(2) << metadata << '\n';
    if (!output.good()) {
      std::filesystem::remove(temporary);
      throw std::runtime_error("failed to write final index metadata");
    }
  }
  std::filesystem::rename(temporary, path);
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
    validate_shard_fingerprint(
      inputs[shard], path, layout.shard_fingerprints[shard]);
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
  validate_shard_fingerprint(
    input, input_path, layout.shard_fingerprints[shard]);
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
    if (base == 0) {
      const u32 audit_vectors = std::min<u32>(batch, 64);
      vec<f32> audit_transformed(layout.dim);
      vec<byte_t> audit_code(model.code_bytes());
      u64 mismatched_components = 0;
      for (u32 index = 0; index < audit_vectors; ++index) {
        gpu_search::pq::encode(
          model,
          span<const f32>{decoded.data() + static_cast<size_t>(index) * layout.dim,
                          layout.dim},
          audit_code, audit_transformed);
        const byte_t* faiss_code =
          codes.data() + static_cast<size_t>(index) * model.code_bytes();
        for (u32 component = 0; component < model.code_bytes(); ++component) {
          mismatched_components += audit_code[component] != faiss_code[component];
        }
      }
      const u64 audited_components =
        static_cast<u64>(audit_vectors) * model.code_bytes();
      if (mismatched_components * 100 > audited_components) {
        throw std::runtime_error(
          "Faiss and runtime OPQ/PQ encoders disagree on more than 1% of audited components");
      }
      std::cerr << "PQ encoder audit shard " << (shard + 1) << "/"
                << layout.shards << ": vectors=" << audit_vectors
                << " component_mismatches=" << mismatched_components << '\n';
    }
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
  header.vector_dtype = static_cast<u32>(layout.dtype);
  header.entry_count = count;
  header.remote_offset = remote_offset;
  header.payload_bytes = count * model.code_bytes();
  header.model_checksum = model.checksum();
  header.payload_checksum = checksum;
  header.build_fingerprint = layout.build_fingerprint;
  header.shard_fingerprint = layout.shard_fingerprints[shard];
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
  const u32 hardware_threads = std::max(1u, std::thread::hardware_concurrency());
  const u32 training_threads = options.threads == 0
    ? std::min<u32>(hardware_threads, 32) : std::min(options.threads, hardware_threads);
  if (training_threads == 0) {
    throw std::invalid_argument("PQ training threads must be greater than zero");
  }
  omp_set_dynamic(0);
  omp_set_max_active_levels(1);
  omp_set_num_threads(static_cast<int>(training_threads));
  std::cerr << "PQ CPU runtime: threads=" << training_threads
            << " hardware_threads=" << hardware_threads << '\n';

  PqIndexResult result;
  result.model_file = index_path::navigation_model_file(
    options.index_prefix, options.subquantizers);
  result.node_count = layout.node_count;
  result.code_files.resize(layout.shards);
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    result.code_files[shard] = index_path::navigation_code_file(
      options.index_prefix, shard + 1, layout.shards, options.subquantizers);
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

  const PersistentLayout persistent = make_persistent_layout(
    metadata, layout, model.code_bytes());
  for (u32 shard = 0; shard < layout.shards; ++shard) {
    encode_shard(options.index_prefix, layout, model, options, shard,
                 result.code_files[shard], persistent.code_offsets[shard]);
    result.code_bytes += persistent.code_region_bytes[shard];
  }

  metadata["navigation_quantizer"] = "opq_pq";
  metadata["pq_subquantizers"] = model.subquantizers;
  metadata["pq_bits"] = model.bits_per_code;
  metadata["navigation_model_checksum"] = model.checksum();
  metadata["navigation_model_file"] = result.model_file.string();
  metadata["navigation_format"] = "opq_pq_graph_v1";
  apply_persistent_layout(metadata, persistent, model.code_bytes());
  write_metadata_atomic(metadata_path, metadata);

  gpu_search::format::View manifest;
  if (!gpu_search::format::synthesize_distributed_view(
        options.index_prefix, manifest, &error)) {
    throw std::runtime_error(error);
  }
  std::cerr << "PQ index ready: nodes=" << result.node_count
            << " code_bytes=" << result.code_bytes
            << " model_checksum=" << result.model_checksum << '\n';
  return result;
}

}  // namespace tools::vamana_offline
