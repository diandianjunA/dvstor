#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <library/connection_manager.hh>
#include <library/context.hh>
#include <library/memory_region.hh>

#include "common/configuration.hh"
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

  const u32 blocks = environment_u32("DVSTOR_GPUNETIO_STRESS_BLOCKS", 64);
  const u32 iterations = environment_u32("DVSTOR_GPUNETIO_STRESS_ITERATIONS", 32);
  const u32 batch_reads = environment_u32("DVSTOR_GPUNETIO_BATCH_READS", 1);
  if (batch_reads > gpu_search::kPersistentMaxExact) {
    throw std::runtime_error("DVSTOR_GPUNETIO_BATCH_READS exceeds kernel capacity");
  }
  const u32 worker_count = std::min<u32>(parameters.gpu_rdma_qps, 4);
  const size_t stream_count = batch_reads == 1
    ? static_cast<size_t>(blocks) * worker_count : blocks;
  const size_t destination_bytes = stream_count * batch_reads * sizeof(u64);
  gpu::GpuNetioPersistentTransport transport{
    config, std::max<size_t>(4096, destination_bytes), context,
    connection_manager, remote_regions};
  const gpu::GpuNetioPersistentView view = transport.view();
  u32* stop = nullptr;
  u32* disabled = nullptr;
  i32* error = nullptr;
  i32* statuses = nullptr;
  u32* completed = nullptr;
  cudaStream_t stream = nullptr;
  check_cuda("cudaMalloc(stop)", cudaMalloc(&stop, sizeof(*stop)));
  check_cuda("cudaMalloc(disabled)", cudaMalloc(&disabled, sizeof(*disabled)));
  check_cuda("cudaMalloc(error)", cudaMalloc(&error, sizeof(*error)));
  check_cuda("cudaMalloc(statuses)",
             cudaMalloc(&statuses, stream_count * sizeof(*statuses)));
  check_cuda("cudaMalloc(completed)", cudaMalloc(&completed, sizeof(*completed)));
  check_cuda("cudaMemset(stop)", cudaMemset(stop, 0, sizeof(*stop)));
  check_cuda("cudaMemset(disabled)", cudaMemset(disabled, 0, sizeof(*disabled)));
  check_cuda("cudaMemset(error)", cudaMemset(error, 0, sizeof(*error)));
  check_cuda("cudaMemset(statuses)",
             cudaMemset(statuses, 0, stream_count * sizeof(*statuses)));
  check_cuda("cudaMemset(completed)", cudaMemset(completed, 0, sizeof(*completed)));
  check_cuda("cudaStreamCreate", cudaStreamCreate(&stream));
  gpu_search::PersistentKernelParams probe_params{
    .submissions = {},
    .device_submissions = {},
    .completions = {},
    .delta_submissions = {},
    .delta_completions = {},
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
  if (batch_reads == 1) {
    gpu_search::launch_gpunetio_locked_read_probe(
      stream, probe_params, view.data, sizeof(u64), statuses, completed,
      blocks, iterations);
  } else {
    gpu_search::launch_gpunetio_batched_read_probe(
      stream, probe_params, view.data, sizeof(u64), statuses, completed,
      blocks, batch_reads);
  }
  const cudaError_t launch_status = cudaGetLastError();
  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  std::vector<i32> host_statuses(stream_count);
  u32 host_completed = 0;
  u32 host_disabled = 0;
  i32 host_error = 0;
  check_cuda("cudaMemcpy(statuses)", cudaMemcpy(
    host_statuses.data(), statuses, stream_count * sizeof(i32), cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(completed)", cudaMemcpy(
    &host_completed, completed, sizeof(host_completed), cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(disabled)", cudaMemcpy(
    &host_disabled, disabled, sizeof(host_disabled), cudaMemcpyDeviceToHost));
  check_cuda("cudaMemcpy(error)", cudaMemcpy(
    &host_error, error, sizeof(host_error), cudaMemcpyDeviceToHost));
  check_cuda("cudaStreamDestroy", cudaStreamDestroy(stream));
  check_cuda("cudaFree(completed)", cudaFree(completed));
  check_cuda("cudaFree(statuses)", cudaFree(statuses));
  check_cuda("cudaFree(error)", cudaFree(error));
  check_cuda("cudaFree(disabled)", cudaFree(disabled));
  check_cuda("cudaFree(stop)", cudaFree(stop));
  const size_t expected = batch_reads == 1
    ? stream_count * iterations : static_cast<size_t>(blocks) * (batch_reads + 1);
  const bool stress_ok = launch_status == cudaSuccess && sync_status == cudaSuccess &&
    host_error == 0 && host_completed == expected &&
    std::all_of(host_statuses.begin(), host_statuses.end(), [](i32 status) { return status == 0; });
  if (!stress_ok) {
    std::cerr << "GPUNetIO locked-read stress failed: launch=" << cudaGetErrorString(launch_status)
              << " sync=" << cudaGetErrorString(sync_status)
              << " completed=" << host_completed << "/" << expected
              << " disabled=" << host_disabled
              << " error=" << host_error << " statuses=";
    for (size_t index = 0; index < std::min<size_t>(stream_count, 16); ++index) {
      std::cerr << (index == 0 ? "[" : ",") << host_statuses[index];
    }
    std::cerr << "]\n";
    return EXIT_FAILURE;
  }
  const double seconds = std::chrono::duration<double>(elapsed).count();
  std::cout << "GPUNetIO locked-read stress passed: operations=" << host_completed
            << " qps=" << view.qps_per_node
            << " warp_workers=" << worker_count
            << " batch_reads=" << batch_reads
            << " rate=" << static_cast<double>(host_completed) / seconds << " ops/s\n";
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
