// Microbenchmark for the persistent-kernel OPQ rotation (query matvec).
//
// One 128-thread CTA rotates a production-sized query, mirroring the
// per-query-CTA prepare phase of process_query:
//   mode 0 - legacy thread-per-row serial matvec (pre-optimization)
//   mode 1 - warp-cooperative coalesced matvec, production helper
//            (apply_opq_rotation: four interleaved rows per warp)
//
// With N CTAs each block rotates its own query against the shared matrix,
// emulating the L2 pressure of concurrent query CTAs.
//
// Usage: gpu_opq_rotation_microbench [dim] [iterations] [ctas]
// Prints per-rotation time for each mode and checks both against a CPU
// reference matvec (relative error < 1e-4) before timing.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/query_traversal.cuh"

namespace {

using gpu_search::f32;
using gpu_search::u32;
using gpu_search::persistent_kernel_detail::apply_opq_rotation;
using gpu_search::persistent_kernel_detail::finite_f32_bits;
using gpu_search::persistent_kernel_detail::saturate_device_component;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                          count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) (void)cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return data_; }
  size_t size() const { return count_; }

 private:
  T* data_{};
  size_t count_{};
};

// Pre-optimization implementation: one thread per output row, serial MAC
// scan over the row.  Warp lanes touch addresses strided by dim floats.
__global__ void opq_rotation_legacy_kernel(
    const f32* __restrict__ opq_matrix, const f32* __restrict__ queries,
    f32* __restrict__ transformed, u32 dim, u32 iterations) {
  const f32* query = queries + static_cast<size_t>(blockIdx.x) * dim;
  f32* output = transformed + static_cast<size_t>(blockIdx.x) * dim;
  for (u32 iteration = 0; iteration < iterations; ++iteration) {
    for (u32 row = threadIdx.x; row < dim; row += blockDim.x) {
      f32 value = 0.0f;
      const f32* matrix_row = opq_matrix + static_cast<size_t>(row) * dim;
      for (u32 column = 0; column < dim; ++column) {
        value += matrix_row[column] * query[column];
      }
      if (!finite_f32_bits(value)) {
        double wide_value = 0.0;
        for (u32 column = 0; column < dim; ++column) {
          wide_value = fma(static_cast<double>(matrix_row[column]),
                           static_cast<double>(query[column]), wide_value);
        }
        value = saturate_device_component(wide_value);
      }
      output[row] = value;
    }
    __syncthreads();
  }
}

__global__ void opq_rotation_coalesced_kernel(
    const f32* __restrict__ opq_matrix, const f32* __restrict__ queries,
    f32* __restrict__ transformed, u32 dim, u32 iterations) {
  const f32* query = queries + static_cast<size_t>(blockIdx.x) * dim;
  f32* output = transformed + static_cast<size_t>(blockIdx.x) * dim;
  for (u32 iteration = 0; iteration < iterations; ++iteration) {
    apply_opq_rotation(opq_matrix, query, output, dim);
    __syncthreads();
  }
}

double time_kernel(void (*launch)(const f32*, const f32*, f32*, u32, u32),
                   const f32* matrix, const f32* queries, f32* transformed,
                   u32 dim, u32 iterations, u32 ctas) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate");
  check_cuda(cudaEventRecord(start), "cudaEventRecord start");
  launch<<<ctas, 128>>>(matrix, queries, transformed, dim, iterations);
  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
  check_cuda(cudaGetLastError(), "kernel launch");
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "cudaEventElapsedTime");
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy");
  return static_cast<double>(elapsed_ms) / iterations;
}

std::vector<double> cpu_reference(const std::vector<f32>& matrix,
                                  const std::vector<f32>& query, u32 dim) {
  std::vector<double> result(dim, 0.0);
  for (u32 row = 0; row < dim; ++row) {
    double value = 0.0;
    for (u32 column = 0; column < dim; ++column) {
      value = std::fma(
        static_cast<double>(matrix[static_cast<size_t>(row) * dim + column]),
        static_cast<double>(query[column]), value);
    }
    result[row] = value;
  }
  return result;
}

// Returns the worst relative error against the double-precision reference.
double check_against_reference(const std::vector<f32>& actual,
                               const std::vector<double>& reference,
                               u32 dim, const char* mode) {
  double worst = 0.0;
  for (u32 row = 0; row < dim; ++row) {
    const double denominator =
      std::max(std::fabs(reference[row]), 1e-6);
    const double error =
      std::fabs(static_cast<double>(actual[row]) - reference[row]) /
      denominator;
    worst = std::max(worst, error);
  }
  if (worst >= 1e-4) {
    std::fprintf(stderr, "mode=%s relative check FAILED: worst=%.3e\n",
                 mode, worst);
    throw std::runtime_error("OPQ rotation output mismatch");
  }
  return worst;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      std::printf("SKIP: no CUDA device available\n");
      return 0;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");

    const u32 dim = argc > 1 ? static_cast<u32>(std::strtoul(argv[1], nullptr, 10)) : 1024;
    const u32 iterations = argc > 2 ? static_cast<u32>(std::strtoul(argv[2], nullptr, 10)) : 100;
    const u32 ctas = argc > 3 ? static_cast<u32>(std::strtoul(argv[3], nullptr, 10)) : 1;
    if (dim == 0 || iterations == 0 || ctas == 0) {
      std::fprintf(stderr, "dim, iterations and ctas must be positive\n");
      return 1;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<f32> uniform(-0.5f, 0.5f);
    std::vector<f32> matrix(static_cast<size_t>(dim) * dim);
    for (auto& value : matrix) value = uniform(rng);
    std::vector<f32> host_queries(static_cast<size_t>(ctas) * dim);
    for (auto& value : host_queries) value = uniform(rng);

    DeviceBuffer<f32> device_matrix(matrix.size());
    DeviceBuffer<f32> device_queries(host_queries.size());
    DeviceBuffer<f32> device_transformed(host_queries.size());
    check_cuda(cudaMemcpy(device_matrix.get(), matrix.data(),
                          matrix.size() * sizeof(f32),
                          cudaMemcpyHostToDevice), "copy matrix");
    check_cuda(cudaMemcpy(device_queries.get(), host_queries.data(),
                          host_queries.size() * sizeof(f32),
                          cudaMemcpyHostToDevice), "copy queries");

    const std::vector<double> reference =
      cpu_reference(matrix, host_queries, dim);
    std::vector<f32> actual(dim);

    struct Mode {
      const char* name;
      void (*kernel)(const f32*, const f32*, f32*, u32, u32);
    };
    const Mode modes[] = {
      {"legacy-thread-per-row", opq_rotation_legacy_kernel},
      {"warp-per-row-coalesced", opq_rotation_coalesced_kernel},
    };

    std::printf("dim=%u iterations=%u ctas=%u\n", dim, iterations, ctas);
    double baseline_ms = 0.0;
    for (const Mode& mode : modes) {
      // Untimed correctness pass on CTA 0.
      mode.kernel<<<1, 128>>>(device_matrix.get(), device_queries.get(),
                              device_transformed.get(), dim, 1);
      check_cuda(cudaGetLastError(), "correctness launch");
      check_cuda(cudaDeviceSynchronize(), "correctness sync");
      check_cuda(cudaMemcpy(actual.data(), device_transformed.get(),
                            dim * sizeof(f32), cudaMemcpyDeviceToHost),
                 "copy result");
      const double worst =
        check_against_reference(actual, reference, dim, mode.name);

      // Warmup, then timed run.
      mode.kernel<<<ctas, 128>>>(device_matrix.get(), device_queries.get(),
                                 device_transformed.get(), dim, 10);
      check_cuda(cudaDeviceSynchronize(), "warmup sync");
      const double per_rotation_ms =
        time_kernel(mode.kernel, device_matrix.get(), device_queries.get(),
                    device_transformed.get(), dim, iterations, ctas);
      if (mode.kernel == opq_rotation_legacy_kernel) {
        baseline_ms = per_rotation_ms;
      }
      std::printf(
        "mode=%-22s per_rotation_ms=%.4f (%.1f us) speedup=%.2fx max_rel_err=%.2e\n",
        mode.name, per_rotation_ms, per_rotation_ms * 1000.0,
        baseline_ms / per_rotation_ms, worst);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
