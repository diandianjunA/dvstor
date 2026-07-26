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

__global__ void qp_lease_test_kernel(
    QpExpansionLeaseState* state, u32* checks) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  checks[0] = qp_expansion_lease_publish(state, 4) ? 1u : 0u;
  checks[1] = qp_expansion_lease_epoch(qp_expansion_lease_load(state));
  checks[2] = qp_expansion_lease_available(qp_expansion_lease_load(state));
  QpExpansionLeaseClaim claim{};
  checks[3] = qp_expansion_lease_try_claim(state, 1, 0, 2, claim) ? 1u : 0u;
  checks[4] = qp_expansion_lease_available(qp_expansion_lease_load(state));
  QpExpansionLeaseClaim rejected{};
  checks[5] =
    qp_expansion_lease_try_claim(state, 1, 0, 3, rejected) ? 1u : 0u;
  qp_expansion_lease_return(state, 1, claim);
  checks[6] = qp_expansion_lease_available(qp_expansion_lease_load(state));
  qp_expansion_lease_revoke(state);
  checks[7] = qp_expansion_lease_epoch(qp_expansion_lease_load(state));
  checks[8] = qp_expansion_lease_available(qp_expansion_lease_load(state));
  qp_expansion_lease_return(state, 1, claim);
  checks[9] = qp_expansion_lease_publish(state, 3) ? 1u : 0u;
  checks[10] = qp_expansion_lease_publish(state, 5) ? 1u : 0u;
  checks[11] = qp_expansion_lease_available(qp_expansion_lease_load(state));
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
    QpExpansionLeaseState* device_lease = nullptr;
    u32* device_checks = nullptr;
    check_cuda(cudaMalloc(
      reinterpret_cast<void**>(&device_state), sizeof(initial)), "cudaMalloc");
    check_cuda(cudaMalloc(
      reinterpret_cast<void**>(&device_lease), sizeof(QpExpansionLeaseState)),
      "cudaMalloc lease");
    check_cuda(cudaMalloc(
      reinterpret_cast<void**>(&device_checks), 12 * sizeof(u32)), "cudaMalloc");
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
    check_cuda(cudaMemset(device_lease, 0, sizeof(QpExpansionLeaseState)),
               "cudaMemset lease");
    check_cuda(cudaMemset(device_checks, 0, 12 * sizeof(u32)),
               "cudaMemset checks");
    qp_lease_test_kernel<<<1, 1>>>(device_lease, device_checks);
    check_cuda(cudaGetLastError(), "qp_lease_test_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "qp_lease_test_kernel");
    QpExpansionLeaseState lease{};
    u32 lease_checks[12]{};
    check_cuda(cudaMemcpy(
      &lease, device_lease, sizeof(lease), cudaMemcpyDeviceToHost),
      "cudaMemcpy lease D2H");
    check_cuda(cudaMemcpy(
      lease_checks, device_checks, sizeof(lease_checks),
      cudaMemcpyDeviceToHost), "cudaMemcpy lease checks D2H");
    (void)cudaFree(device_lease);
    (void)cudaFree(device_checks);
    (void)cudaFree(device_state);
    if (lease_checks[0] != 1 || lease_checks[1] != 1 ||
        lease_checks[2] != 4 || lease_checks[3] != 1 ||
        lease_checks[4] != 2 || lease_checks[5] != 0 ||
        lease_checks[6] != 4 || lease_checks[7] != 2 ||
        lease_checks[8] != 0 || lease_checks[9] != 1 ||
        lease_checks[10] != 0 || lease_checks[11] != 3 ||
        lease.offers != 2 || lease.claims != 0 || lease.rejects != 0 ||
        lease.returns != 0 || lease.revocations != 0 ||
        lease.stale_returns != 0) {
      throw std::runtime_error("QP expansion lease transition mismatch");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
