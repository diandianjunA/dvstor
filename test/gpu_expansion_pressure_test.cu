#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "gpu_search/persistent_kernel.hh"

namespace {

using namespace gpu_search;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

__global__ void pressure_test_kernel(
    ExpansionPressureState* state, u32* checks) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  checks[0] = expansion_pressure_grant_idle(state) ? 1u : 0u;
  expansion_pressure_query_enter(state);
  checks[1] = expansion_pressure_active(
    expansion_pressure_load(state));
  checks[2] = expansion_pressure_grant_idle(state) ? 1u : 0u;
  checks[3] = expansion_pressure_grant_idle(state) ? 1u : 0u;
  checks[4] = expansion_pressure_grant_idle(state) ? 1u : 0u;
  checks[5] = expansion_pressure_credit(
    expansion_pressure_load(state));
  const unsigned long long before_read = expansion_pressure_load(state);
  checks[6] = expansion_pressure_credit(before_read);
  checks[7] = expansion_pressure_load(state) == before_read ? 1u : 0u;
  expansion_pressure_query_exit(state);
  checks[8] = expansion_pressure_active(
    expansion_pressure_load(state));
  checks[9] = expansion_pressure_credit(
    expansion_pressure_load(state));

  expansion_pressure_query_enter(state);
  (void)expansion_pressure_grant_idle(state);
  expansion_pressure_clear_credit(state, true, false);
  (void)expansion_pressure_grant_idle(state);
  expansion_pressure_clear_credit(state, false, true);
  expansion_pressure_query_exit(state);
}

void test_idle_episode_transition() {
  bool announced = false;
  if (expansion_owner_idle_episode_transition(
        0, true, true, false, announced) || announced) {
    throw std::runtime_error("idle owner granted without active query");
  }
  if (!expansion_owner_idle_episode_transition(
        1, true, true, false, announced) || !announced) {
    throw std::runtime_error("first owner idle episode was not announced");
  }
  if (expansion_owner_idle_episode_transition(
        1, true, true, false, announced)) {
    throw std::runtime_error("continuous idle loop granted twice");
  }
  if (expansion_owner_idle_episode_transition(
        1, false, false, true, announced) || announced) {
    throw std::runtime_error("owner batch did not reset idle episode");
  }
  if (!expansion_owner_idle_episode_transition(
        1, true, true, false, announced)) {
    throw std::runtime_error("next idle episode did not grant");
  }
}

}  // namespace

int main() {
  try {
    test_idle_episode_transition();
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
      std::cout << "SKIP: no CUDA device available\n";
      return 0;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    ExpansionPressureState initial{};
    initial.maximum_credit_tiles = 2;
    ExpansionPressureState* device_state = nullptr;
    u32* device_checks = nullptr;
    check_cuda(cudaMalloc(
      reinterpret_cast<void**>(&device_state), sizeof(initial)), "cudaMalloc");
    check_cuda(cudaMalloc(
      reinterpret_cast<void**>(&device_checks), 10 * sizeof(u32)), "cudaMalloc");
    check_cuda(cudaMemcpy(
      device_state, &initial, sizeof(initial), cudaMemcpyHostToDevice),
      "cudaMemcpy state H2D");
    pressure_test_kernel<<<1, 1>>>(device_state, device_checks);
    check_cuda(cudaGetLastError(), "pressure_test_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "pressure_test_kernel");
    ExpansionPressureState result{};
    u32 checks[10]{};
    check_cuda(cudaMemcpy(
      &result, device_state, sizeof(result), cudaMemcpyDeviceToHost),
      "cudaMemcpy state D2H");
    check_cuda(cudaMemcpy(
      checks, device_checks, sizeof(checks), cudaMemcpyDeviceToHost),
      "cudaMemcpy checks D2H");
    (void)cudaFree(device_checks);
    (void)cudaFree(device_state);
    if (checks[0] != 0 || checks[1] != 1 ||
        checks[2] != 1 || checks[3] != 1 || checks[4] != 0 ||
        checks[5] != 2 || checks[6] != 2 || checks[7] != 1 ||
        checks[8] != 0 || checks[9] != 0 ||
        expansion_pressure_active(result.control) != 0 ||
        expansion_pressure_credit(result.control) != 0 ||
        expansion_pressure_active_peak(result.control) != 1 ||
        expansion_pressure_credit_peak(result.control) != 2 ||
        result.hunger_grants != 4 ||
        result.congestion_clears != 2 ||
        result.ring_backpressure_events != 1 ||
        result.sq_defer_events != 1) {
      throw std::runtime_error("expansion pressure state transition mismatch");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
