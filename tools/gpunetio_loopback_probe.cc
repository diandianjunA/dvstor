#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <library/connection_manager.hh>
#include <library/context.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
#include "gpu/gpunetio_probe.hh"
#include "gpu/gpunetio_transport.hh"
#include "gpu_search/persistent_kernel.hh"
#include "memory_node/startup_protocol.hh"

namespace {

void check_cuda(const char* operation, cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

u32 environment_u32(const char* name, u32 fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
    throw std::runtime_error(std::string("invalid ") + name + "=" + value);
  }
  return static_cast<u32>(parsed);
}

bool environment_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return false;
  const std::string normalized{value};
  return normalized == "1" || normalized == "true" ||
    normalized == "on" || normalized == "yes";
}

std::vector<u32> environment_u32_list(
    const char* name, std::vector<u32> fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  std::string normalized{raw};
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::istringstream input{normalized};
  std::vector<u32> values;
  std::string token;
  while (input >> token) {
    size_t parsed = 0;
    const unsigned long value = std::stoul(token, &parsed, 10);
    if (parsed != token.size() || value == 0 || value > UINT32_MAX) {
      throw std::runtime_error(
        std::string("invalid value in ") + name + "=" + raw);
    }
    values.push_back(static_cast<u32>(value));
  }
  if (values.empty()) {
    throw std::runtime_error(std::string(name) + " must not be empty");
  }
  return values;
}

size_t align_up(const size_t value, const size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

uint64_t percentile(const std::vector<uint64_t>& sorted, const double p) {
  if (sorted.empty()) return 0;
  const size_t index = static_cast<size_t>(
    std::ceil(p * static_cast<double>(sorted.size())) - 1.0);
  return sorted[std::min(index, sorted.size() - 1)];
}

struct PayloadSweepStorage {
  i32* statuses{};
  uint64_t* completed_reads{};
  u32* dump_wqe_flags{};
  uint64_t* batch_latency_ns{};
  cudaEvent_t started{};
  cudaEvent_t finished{};

  ~PayloadSweepStorage() {
    if (finished != nullptr) cudaEventDestroy(finished);
    if (started != nullptr) cudaEventDestroy(started);
    if (batch_latency_ns != nullptr) cudaFree(batch_latency_ns);
    if (dump_wqe_flags != nullptr) cudaFree(dump_wqe_flags);
    if (completed_reads != nullptr) cudaFree(completed_reads);
    if (statuses != nullptr) cudaFree(statuses);
  }
};

void run_payload_case(
    cudaStream_t stream, const gpu::GpuNetioPersistentView& view,
    PayloadSweepStorage& storage, const u32 workers, const u32 batch_reads,
    const u32 warmup_batches, const u32 measured_batches,
    const u32 destination_stride, const u32 first_stage_bytes,
    const u32 second_stage_bytes, const uint64_t timeout_ns,
    const uint64_t remote_span_bytes, const u32 repeat,
    const char* order) {
  const size_t latency_count =
    static_cast<size_t>(workers) * measured_batches;
  const size_t storage_latency_count =
    static_cast<size_t>(workers) *
    std::max(warmup_batches, measured_batches);
  check_cuda("cudaMemset(payload statuses)",
             cudaMemset(storage.statuses, 0, workers * sizeof(*storage.statuses)));
  check_cuda("cudaMemset(payload completed)",
             cudaMemset(storage.completed_reads, 0,
                        workers * sizeof(*storage.completed_reads)));
  check_cuda("cudaMemset(payload dump flags)",
             cudaMemset(storage.dump_wqe_flags, 0,
                        workers * sizeof(*storage.dump_wqe_flags)));
  check_cuda("cudaMemset(payload latencies)",
             cudaMemset(storage.batch_latency_ns, 0,
                        storage_latency_count *
                          sizeof(*storage.batch_latency_ns)));

  if (warmup_batches != 0) {
    gpu::launch_gpunetio_payload_probe(
      stream,
      gpu::GpuNetioPayloadProbeParams{
        .local_mkey = view.local_mkey,
        .local_iova_base = view.local_iova_base,
        .remote_regions =
          reinterpret_cast<const gpu::GpuNetioRemoteMemoryRegion*>(
            view.remote_regions),
        .remote_region_count = view.remote_region_count,
        .qp_array = view.qp_array,
        .qp_count = view.remote_region_count * view.qps_per_node,
        .active_qps = workers,
        .destination = view.data,
        .destination_stride = destination_stride,
        .remote_record_stride = 832,
        .remote_span_bytes = remote_span_bytes,
        .dump_ptr = view.dump,
        .first_stage_bytes = first_stage_bytes,
        .second_stage_bytes = second_stage_bytes,
        .batch_reads = batch_reads,
        .warmup_batches = 0,
        .measured_batches = warmup_batches,
        .timeout_ns = timeout_ns,
        .status_codes = storage.statuses,
        .completed_reads = storage.completed_reads,
        .dump_wqe_flags = storage.dump_wqe_flags,
        .batch_latency_ns = storage.batch_latency_ns,
      });
    check_cuda("launch_gpunetio_payload_probe(warmup)", cudaGetLastError());
    check_cuda("cudaStreamSynchronize(payload warmup)",
               cudaStreamSynchronize(stream));
    std::vector<i32> warmup_statuses(workers);
    check_cuda("cudaMemcpy(payload warmup statuses)",
               cudaMemcpy(warmup_statuses.data(), storage.statuses,
                          workers * sizeof(*storage.statuses),
                          cudaMemcpyDeviceToHost));
    const auto bad_warmup = std::find_if(
      warmup_statuses.begin(), warmup_statuses.end(),
      [](const i32 value) { return value != 0; });
    if (bad_warmup != warmup_statuses.end()) {
      const size_t worker =
        static_cast<size_t>(bad_warmup - warmup_statuses.begin());
      throw std::runtime_error(
        "payload warmup failed: worker=" + std::to_string(worker) +
        " status=" + std::to_string(*bad_warmup));
    }
    check_cuda("cudaMemset(payload statuses after warmup)",
               cudaMemset(storage.statuses, 0,
                          workers * sizeof(*storage.statuses)));
    check_cuda("cudaMemset(payload completed after warmup)",
               cudaMemset(storage.completed_reads, 0,
                          workers * sizeof(*storage.completed_reads)));
    check_cuda("cudaMemset(payload dump flags after warmup)",
               cudaMemset(storage.dump_wqe_flags, 0,
                          workers * sizeof(*storage.dump_wqe_flags)));
    check_cuda("cudaMemset(payload latencies after warmup)",
               cudaMemset(storage.batch_latency_ns, 0,
                          storage_latency_count *
                            sizeof(*storage.batch_latency_ns)));
  }

  check_cuda("cudaEventRecord(payload start)",
             cudaEventRecord(storage.started, stream));
  gpu::launch_gpunetio_payload_probe(
    stream,
    gpu::GpuNetioPayloadProbeParams{
      .local_mkey = view.local_mkey,
      .local_iova_base = view.local_iova_base,
      .remote_regions =
        reinterpret_cast<const gpu::GpuNetioRemoteMemoryRegion*>(
          view.remote_regions),
      .remote_region_count = view.remote_region_count,
      .qp_array = view.qp_array,
      .qp_count = view.remote_region_count * view.qps_per_node,
      .active_qps = workers,
      .destination = view.data,
      .destination_stride = destination_stride,
      .remote_record_stride = 832,
      .remote_span_bytes = remote_span_bytes,
      .dump_ptr = view.dump,
      .first_stage_bytes = first_stage_bytes,
      .second_stage_bytes = second_stage_bytes,
      .batch_reads = batch_reads,
      .warmup_batches = 0,
      .measured_batches = measured_batches,
      .timeout_ns = timeout_ns,
      .status_codes = storage.statuses,
      .completed_reads = storage.completed_reads,
      .dump_wqe_flags = storage.dump_wqe_flags,
      .batch_latency_ns = storage.batch_latency_ns,
    });
  check_cuda("launch_gpunetio_payload_probe", cudaGetLastError());
  check_cuda("cudaEventRecord(payload finish)",
             cudaEventRecord(storage.finished, stream));
  check_cuda("cudaEventSynchronize(payload finish)",
             cudaEventSynchronize(storage.finished));

  float elapsed_ms = 0.0f;
  check_cuda("cudaEventElapsedTime(payload)",
             cudaEventElapsedTime(&elapsed_ms, storage.started, storage.finished));
  std::vector<i32> statuses(workers);
  std::vector<uint64_t> completed_reads(workers);
  std::vector<u32> dump_wqe_flags(workers);
  std::vector<uint64_t> latencies(latency_count);
  check_cuda("cudaMemcpy(payload statuses)",
             cudaMemcpy(statuses.data(), storage.statuses,
                        workers * sizeof(*storage.statuses),
                        cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(payload completed)",
             cudaMemcpy(completed_reads.data(), storage.completed_reads,
                        workers * sizeof(*storage.completed_reads),
                        cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(payload dump flags)",
             cudaMemcpy(dump_wqe_flags.data(), storage.dump_wqe_flags,
                        workers * sizeof(*storage.dump_wqe_flags),
                        cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(payload latencies)",
             cudaMemcpy(latencies.data(), storage.batch_latency_ns,
                        latency_count * sizeof(*storage.batch_latency_ns),
                        cudaMemcpyDeviceToHost));

  const u32 stages = second_stage_bytes == 0 ? 1u : 2u;
  const uint64_t expected_reads =
    static_cast<uint64_t>(workers) * measured_batches * batch_reads * stages;
  const uint64_t observed_reads =
    std::accumulate(completed_reads.begin(), completed_reads.end(), uint64_t{0});
  const auto bad_status = std::find_if(
    statuses.begin(), statuses.end(), [](const i32 value) { return value != 0; });
  if (bad_status != statuses.end() || observed_reads != expected_reads) {
    const size_t worker = bad_status == statuses.end()
      ? 0 : static_cast<size_t>(bad_status - statuses.begin());
    throw std::runtime_error(
      "payload probe failed: worker=" + std::to_string(worker) +
      " status=" + std::to_string(statuses[worker]) +
      " completed=" + std::to_string(observed_reads) + "/" +
      std::to_string(expected_reads));
  }
  if (std::find(latencies.begin(), latencies.end(), uint64_t{0}) !=
      latencies.end()) {
    throw std::runtime_error("payload probe emitted a zero latency sample");
  }

  std::sort(latencies.begin(), latencies.end());
  const long double latency_sum =
    std::accumulate(latencies.begin(), latencies.end(), static_cast<long double>(0));
  const double mean_latency_ns =
    static_cast<double>(latency_sum / static_cast<long double>(latencies.size()));
  const uint64_t dump_workers =
    std::count(dump_wqe_flags.begin(), dump_wqe_flags.end(), u32{1});
  const uint64_t measured_submissions =
    static_cast<uint64_t>(workers) * measured_batches * stages;
  const uint64_t dump_wqes = dump_workers * measured_batches * stages;
  const uint64_t transport_wqes = observed_reads + dump_wqes;
  const uint64_t requested_bytes =
    static_cast<uint64_t>(workers) * measured_batches * batch_reads *
    (static_cast<uint64_t>(first_stage_bytes) + second_stage_bytes);
  const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
  const double read_wqe_rate = static_cast<double>(observed_reads) / seconds;
  const double requested_gb_s =
    static_cast<double>(requested_bytes) / seconds / 1.0e9;

  std::cout << std::fixed << std::setprecision(3)
            << "LIVE_EXTENT_RDMA_CSV,"
            << repeat << ','
            << order << ','
            << first_stage_bytes << ','
            << second_stage_bytes << ','
            << first_stage_bytes + second_stage_bytes << ','
            << stages << ','
            << workers << ','
            << batch_reads << ','
            << remote_span_bytes << ','
            << measured_batches << ','
            << observed_reads << ','
            << dump_wqes << ','
            << transport_wqes << ','
            << measured_submissions << ','
            << elapsed_ms << ','
            << read_wqe_rate << ','
            << requested_gb_s << ','
            << mean_latency_ns / 1000.0 << ','
            << static_cast<double>(percentile(latencies, 0.50)) / 1000.0 << ','
            << static_cast<double>(percentile(latencies, 0.95)) / 1000.0 << ','
            << static_cast<double>(percentile(latencies, 0.99)) / 1000.0
            << '\n';
}

void run_payload_sweep(
    cudaStream_t stream, const gpu::GpuNetioPersistentView& view,
    const std::vector<u32>& payload_bytes,
    const std::vector<u32>& paired_body_bytes,
    const u32 workers, const u32 batch_reads, const u32 warmup_batches,
    const u32 measured_batches, const u32 destination_stride,
    const uint64_t timeout_ns, const u32 repeat, const bool reverse_order,
    const bool print_header, const uint64_t remote_span_bytes) {
  PayloadSweepStorage storage;
  const size_t latency_count =
    static_cast<size_t>(workers) *
    std::max(warmup_batches, measured_batches);
  check_cuda("cudaMalloc(payload statuses)",
             cudaMalloc(&storage.statuses, workers * sizeof(*storage.statuses)));
  check_cuda("cudaMalloc(payload completed)",
             cudaMalloc(&storage.completed_reads,
                        workers * sizeof(*storage.completed_reads)));
  check_cuda("cudaMalloc(payload dump flags)",
             cudaMalloc(&storage.dump_wqe_flags,
                        workers * sizeof(*storage.dump_wqe_flags)));
  check_cuda("cudaMalloc(payload latencies)",
             cudaMalloc(&storage.batch_latency_ns,
                        latency_count * sizeof(*storage.batch_latency_ns)));
  check_cuda("cudaEventCreate(payload start)",
             cudaEventCreate(&storage.started));
  check_cuda("cudaEventCreate(payload finish)",
             cudaEventCreate(&storage.finished));

  if (print_header) {
    std::cout
      << "LIVE_EXTENT_RDMA_HEADER,"
      << "repeat,order,stage1_B,stage2_B,payload_B,stages,active_QPs,"
      << "batch_reads,working_set_B,measured_batches_per_QP,"
      << "read_WQEs,dump_WQEs,"
      << "transport_WQEs,CQEs,elapsed_ms,read_WQE_per_s,"
      << "application_payload_GB_per_s,batch_latency_mean_us,"
      << "batch_latency_p50_us,batch_latency_p95_us,"
      << "batch_latency_p99_us\n";
  }
  std::vector<u32> ordered_payloads = payload_bytes;
  std::vector<u32> ordered_bodies = paired_body_bytes;
  if (reverse_order) {
    std::reverse(ordered_payloads.begin(), ordered_payloads.end());
    std::reverse(ordered_bodies.begin(), ordered_bodies.end());
  }
  for (const u32 bytes : ordered_payloads) {
    run_payload_case(
      stream, view, storage, workers, batch_reads, warmup_batches,
      measured_batches, destination_stride, bytes, 0, timeout_ns,
      remote_span_bytes, repeat, reverse_order ? "reverse" : "forward");
  }
  for (const u32 body_bytes : ordered_bodies) {
    run_payload_case(
      stream, view, storage, workers, batch_reads, warmup_batches,
      measured_batches, destination_stride, 16, body_bytes, timeout_ns,
      remote_span_bytes, repeat, reverse_order ? "reverse" : "forward");
  }
}

}  // namespace

int main(int argc, char** argv) {
  configuration::IndexConfiguration config{argc, argv};
  Context context{config};
  ClientConnectionManager connection_manager{context, config};
  connection_manager.connect();

  configuration::Parameters parameters{
    .num_threads = 1,
    .gpu_rdma_qps = std::max<u32>(1, config.gpu_rdma_qps),
  };
  for (const QP& qp : connection_manager.server_qps) {
    qp->post_send_inlined(&parameters, sizeof(parameters), IBV_WR_SEND);
    context.poll_send_cq_until_completion();
  }

  MemoryRegionTokens remote_regions(connection_manager.server_qps.size());
  for (size_t i = 0; i < remote_regions.size(); ++i) {
    remote_regions[i] = std::make_unique<MemoryRegionToken>();
    LocalMemoryRegion token_region{context, remote_regions[i].get(), sizeof(MemoryRegionToken)};
    connection_manager.server_qps[i]->post_receive(token_region);
    context.receive();
  }

  const bool payload_sweep =
    environment_enabled("DVSTOR_GPUNETIO_PAYLOAD_SWEEP");
  const u32 qp_count =
    parameters.gpu_rdma_qps *
    static_cast<u32>(connection_manager.server_qps.size());
  const u32 blocks = environment_u32("DVSTOR_GPUNETIO_STRESS_BLOCKS", 64);
  const u32 iterations =
    environment_u32("DVSTOR_GPUNETIO_STRESS_ITERATIONS", 32);
  const u32 legacy_batch_reads =
    environment_u32("DVSTOR_GPUNETIO_BATCH_READS", 1);
  if (legacy_batch_reads > gpu_search::kPersistentMaxExact) {
    throw std::runtime_error(
      "DVSTOR_GPUNETIO_BATCH_READS exceeds kernel capacity");
  }
  const u32 legacy_worker_count =
    std::min<u32>(parameters.gpu_rdma_qps, 4);
  const size_t legacy_stream_count = legacy_batch_reads == 1
    ? static_cast<size_t>(blocks) * legacy_worker_count : blocks;

  const std::vector<u32> payload_bytes = payload_sweep
    ? environment_u32_list(
        "DVSTOR_GPUNETIO_PAYLOAD_BYTES",
        {16, 80, 144, 272, 400, 448, 528, 832})
    : std::vector<u32>{};
  const std::vector<u32> paired_body_bytes = payload_sweep
    ? environment_u32_list(
        "DVSTOR_GPUNETIO_PAIRED_BODY_BYTES", {400, 448})
    : std::vector<u32>{};
  const u32 payload_batch_reads = payload_sweep
    ? environment_u32("DVSTOR_GPUNETIO_PAYLOAD_BATCH_READS", 16) : 1;
  const u32 payload_measured_batches = payload_sweep
    ? environment_u32("DVSTOR_GPUNETIO_PAYLOAD_ITERATIONS", 512) : 1;
  const u32 payload_warmup_batches = payload_sweep
    ? environment_u32("DVSTOR_GPUNETIO_PAYLOAD_WARMUP_ITERATIONS", 32) : 1;
  std::vector<u32> payload_active_qps = payload_sweep
    ? environment_u32_list(
        "DVSTOR_GPUNETIO_PAYLOAD_ACTIVE_QPS_LIST",
        {1, std::min<u32>(8, qp_count), std::min<u32>(32, qp_count), qp_count})
    : std::vector<u32>{1};
  std::sort(payload_active_qps.begin(), payload_active_qps.end());
  payload_active_qps.erase(
    std::unique(payload_active_qps.begin(), payload_active_qps.end()),
    payload_active_qps.end());
  if (payload_active_qps.back() > qp_count) {
    throw std::runtime_error(
      "DVSTOR_GPUNETIO_PAYLOAD_ACTIVE_QPS_LIST exceeds the available QP count");
  }
  const u32 maximum_payload_workers = payload_active_qps.back();
  const u32 payload_repeats = payload_sweep
    ? environment_u32("DVSTOR_GPUNETIO_PAYLOAD_REPEATS", 3) : 1;
  const uint64_t requested_remote_span_bytes = payload_sweep
    ? environment_u32(
        "DVSTOR_GPUNETIO_PAYLOAD_REMOTE_SPAN_BYTES", 64u * 1024u * 1024u)
    : 832;
  const uint64_t payload_remote_span_bytes =
    requested_remote_span_bytes / 832 * 832;
  const uint64_t configured_remote_region_bytes =
    static_cast<uint64_t>(config.mn_memory_gb) << 30;
  if (payload_remote_span_bytes < 832 ||
      payload_remote_span_bytes > configured_remote_region_bytes) {
    throw std::runtime_error(
      "DVSTOR_GPUNETIO_PAYLOAD_REMOTE_SPAN_BYTES is outside the "
      "configured remote memory region");
  }
  if (payload_batch_reads > 16) {
    throw std::runtime_error(
      "DVSTOR_GPUNETIO_PAYLOAD_BATCH_READS must be <= 16");
  }
  u32 maximum_stage_bytes = 0;
  if (payload_sweep) {
    for (const u32 bytes : payload_bytes) {
      if (bytes > 832) {
        throw std::runtime_error(
          "payload bytes exceed the 832-byte source-record stride");
      }
      maximum_stage_bytes = std::max(maximum_stage_bytes, bytes);
    }
    for (const u32 body_bytes : paired_body_bytes) {
      if (body_bytes + 16 > 832) {
        throw std::runtime_error(
          "paired header/body bytes exceed the 832-byte source-record stride");
      }
      maximum_stage_bytes =
        std::max(maximum_stage_bytes, body_bytes + 16);
    }
  }
  const u32 payload_destination_stride = payload_sweep
    ? static_cast<u32>(align_up(maximum_stage_bytes, 64)) : sizeof(u64);
  const size_t destination_bytes = payload_sweep
    ? static_cast<size_t>(maximum_payload_workers) * payload_batch_reads *
        payload_destination_stride
    : legacy_stream_count * legacy_batch_reads * sizeof(u64);
  gpu::GpuNetioPersistentTransport transport{
    config, std::max<size_t>(4096, destination_bytes), context,
    connection_manager, remote_regions};
  const gpu::GpuNetioPersistentView view = transport.view();
  cudaStream_t stream = nullptr;
  check_cuda("cudaStreamCreate", cudaStreamCreate(&stream));

  if (payload_sweep) {
    std::cout << "GPUNetIO live-extent payload sweep: remote_regions="
              << view.remote_region_count
              << " qps_per_region=" << view.qps_per_node
              << " active_qps_min/max=" << payload_active_qps.front()
              << "/" << payload_active_qps.back()
              << " batch_reads=" << payload_batch_reads
              << " warmup_batches_per_qp=" << payload_warmup_batches
              << " measured_batches_per_qp=" << payload_measured_batches
              << " repeats=" << payload_repeats
              << " remote_record_stride=832"
              << " remote_span_bytes=" << payload_remote_span_bytes
              << " completion=one final CQE per stage\n";
    bool print_header = true;
    for (u32 repeat = 1; repeat <= payload_repeats; ++repeat) {
      std::vector<u32> ordered_qps = payload_active_qps;
      const bool reverse_order = repeat % 2 == 0;
      if (reverse_order) {
        std::reverse(ordered_qps.begin(), ordered_qps.end());
      }
      for (const u32 active_qps : ordered_qps) {
        run_payload_sweep(
          stream, view, payload_bytes, paired_body_bytes, active_qps,
          payload_batch_reads, payload_warmup_batches,
          payload_measured_batches, payload_destination_stride,
          static_cast<uint64_t>(config.gpu_direct_timeout_ms) *
            1'000'000ULL,
          repeat, reverse_order, print_header,
          payload_remote_span_bytes);
        print_header = false;
      }
    }
    check_cuda("cudaStreamSynchronize(payload sweep)",
               cudaStreamSynchronize(stream));
  } else {
  u32* stop = nullptr;
  u32* disabled = nullptr;
  i32* error = nullptr;
  i32* statuses = nullptr;
  u32* completed = nullptr;
  check_cuda("cudaMalloc(stop)", cudaMalloc(&stop, sizeof(*stop)));
  check_cuda("cudaMalloc(disabled)", cudaMalloc(&disabled, sizeof(*disabled)));
  check_cuda("cudaMalloc(error)", cudaMalloc(&error, sizeof(*error)));
  check_cuda("cudaMalloc(statuses)",
             cudaMalloc(&statuses,
                        legacy_stream_count * sizeof(*statuses)));
  check_cuda("cudaMalloc(completed)", cudaMalloc(&completed, sizeof(*completed)));
  check_cuda("cudaMemset(stop)", cudaMemset(stop, 0, sizeof(*stop)));
  check_cuda("cudaMemset(disabled)", cudaMemset(disabled, 0, sizeof(*disabled)));
  check_cuda("cudaMemset(error)", cudaMemset(error, 0, sizeof(*error)));
  check_cuda("cudaMemset(statuses)",
             cudaMemset(statuses, 0,
                        legacy_stream_count * sizeof(*statuses)));
  check_cuda("cudaMemset(completed)", cudaMemset(completed, 0, sizeof(*completed)));
  gpu_search::PersistentKernelParams probe_params{
    .submissions = {},
    .device_submissions = {},
    .completions = {},
    .route_submissions = {},
    .route_completions = {},
    .direct_region_count = view.remote_region_count,
    .direct_qps_per_node = view.qps_per_node,
    .direct_local_mkey = view.local_mkey,
    .direct_local_iova_base = view.local_iova_base,
    .direct_timeout_ns = 20000000ULL,
    .direct_regions = reinterpret_cast<const gpu_search::DirectRemoteRegion*>(view.remote_regions),
    .direct_qps = view.qp_array,
    .direct_qp_locks = view.qp_locks,
    .direct_dump = view.dump,
    .direct_disabled = disabled,
    .direct_error = error,
    .stop = stop,
  };
  const auto started = std::chrono::steady_clock::now();
  if (legacy_batch_reads == 1) {
    gpu_search::launch_gpunetio_locked_read_probe(
      stream, probe_params, view.data, sizeof(u64), statuses, completed,
      blocks, iterations);
  } else {
    gpu_search::launch_gpunetio_batched_read_probe(
      stream, probe_params, view.data, sizeof(u64), statuses, completed,
      blocks, legacy_batch_reads);
  }
  const cudaError_t launch_status = cudaGetLastError();
  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  std::vector<i32> host_statuses(legacy_stream_count);
  u32 host_completed = 0;
  u32 host_disabled = 0;
  i32 host_error = 0;
  check_cuda("cudaMemcpy(statuses)", cudaMemcpy(
    host_statuses.data(), statuses,
    legacy_stream_count * sizeof(i32), cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(completed)", cudaMemcpy(
    &host_completed, completed, sizeof(host_completed), cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(disabled)", cudaMemcpy(
    &host_disabled, disabled, sizeof(host_disabled), cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(error)", cudaMemcpy(
    &host_error, error, sizeof(host_error), cudaMemcpyDeviceToHost));
  check_cuda("cudaFree(completed)", cudaFree(completed));
  check_cuda("cudaFree(statuses)", cudaFree(statuses));
  check_cuda("cudaFree(error)", cudaFree(error));
  check_cuda("cudaFree(disabled)", cudaFree(disabled));
  check_cuda("cudaFree(stop)", cudaFree(stop));
  const size_t expected = legacy_batch_reads == 1
    ? legacy_stream_count * iterations
    : static_cast<size_t>(blocks) * (legacy_batch_reads + 1);
  const bool stress_ok = launch_status == cudaSuccess && sync_status == cudaSuccess &&
    host_error == 0 && host_completed == expected &&
    std::all_of(host_statuses.begin(), host_statuses.end(), [](i32 status) { return status == 0; });
  if (!stress_ok) {
    std::cerr << "GPUNetIO locked-read stress failed: launch=" << cudaGetErrorString(launch_status)
              << " sync=" << cudaGetErrorString(sync_status)
              << " completed=" << host_completed << "/" << expected
              << " disabled=" << host_disabled
              << " error=" << host_error << " statuses=";
    for (size_t index = 0;
         index < std::min<size_t>(legacy_stream_count, 16); ++index) {
      std::cerr << (index == 0 ? "[" : ",") << host_statuses[index];
    }
    std::cerr << "]\n";
    return EXIT_FAILURE;
  }
  const double seconds = std::chrono::duration<double>(elapsed).count();
  std::cout << "GPUNetIO locked-read stress passed: operations=" << host_completed
            << " qps=" << view.qps_per_node
            << " warp_workers=" << legacy_worker_count
            << " batch_reads=" << legacy_batch_reads
            << " rate=" << static_cast<double>(host_completed) / seconds << " ops/s\n";
  }
  check_cuda("cudaStreamDestroy", cudaStreamDestroy(stream));
  if (!connection_manager.synchronize()) {
    std::cerr << "GPUNetIO loopback storage synchronization failed\n";
    return EXIT_FAILURE;
  }

  storage_startup::Request request{};
  connection_manager.server_qps.front()->post_send_inlined(
    &request, sizeof(request), IBV_WR_SEND);
  context.poll_send_cq_until_completion();

  storage_startup::Response response{};
  LocalMemoryRegion response_region{context, &response, sizeof(response)};
  connection_manager.server_qps.front()->post_receive(response_region);
  context.receive();
  if (!response.ready) {
    std::cerr << "GPUNetIO loopback storage startup failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "GPUNetIO project QP RDMA Read loopback passed\n";
  return EXIT_SUCCESS;
}
