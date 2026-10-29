// Microbenchmark for the persistent-kernel static PQ scoring paths.
//
// One 128-thread CTA scores a production-sized candidate wave in
// kPersistentMaxMergeCandidates batches, mirroring the per-query-CTA
// execution of approximate_handles_batch:
//   mode 0 - legacy generic per-byte LUT gathers from global memory
//            (the pre-optimization kPersistentRuntimePqSubquantizers path)
//   mode 1 - per-candidate packed scorer over the global LUT
//            (approximate_entry_aligned_packed)
//   mode 2 - shared-memory LUT tile scoring, production helper
//            (score_static_candidates_lut_tiled)
//   mode 3 - tile staging + barriers only (no scoring; cost attribution)
//   mode 4 - tile scoring, register partials + cached ordinals
//   mode 5 - mode 4 plus next-tile register prefetch
//
// With --ctas N the launch uses N CTAs, each with its own LUT slice and
// candidate wave, to emulate the L2 pressure of many concurrent query CTAs.
//
// Usage: gpu_pq_scoring_microbench [subquantizers] [candidates] [iterations]
//        [ctas]
// Prints per-query scoring time for each mode and bitwise-checks modes
// 1/2/4/5 against the scalar reference before timing.

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_search/persistent_kernel/candidate_scoring.cuh"

namespace {

using gpu_search::DeviceShardRegion;
using gpu_search::PersistentKernelParams;
using gpu_search::f32;
using gpu_search::u32;
using gpu_search::u64;
using gpu_search::u8;
using gpu_search::persistent_kernel_detail::approximate_entry;
using gpu_search::persistent_kernel_detail::approximate_entry_wide_fallback;
using gpu_search::persistent_kernel_detail::finite_f32_bits;
using gpu_search::persistent_kernel_detail::score_static_candidates_lut_tiled;
using gpu_search::persistent_kernel_detail::static_ordinal_from_raw;

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

// Diagnostic tile-scoring variants used to attribute the tile path's cost.
// variant 0: stage tiles and synchronize only (no scoring).
// variant 1: register partials + per-slot cached ordinals.
// variant 2: variant 1 plus next-tile prefetch into registers.
template <u32 PqSubquantizers, u32 TileSubquantizers>
__device__ void score_wave_lut_tiled_variant(
    const PersistentKernelParams& params, const f32* query_lut,
    const u64* handles, u32 count, f32* distances, f32* lut_tile,
    u32 variant) {
  static_assert(TileSubquantizers == 4);
  const u32 subquantizers =
    PqSubquantizers == 0 ? params.pq_subquantizers : PqSubquantizers;
  constexpr u32 kMaxSlots = gpu_search::kPersistentMaxMergeCandidates / 128u;
  constexpr u32 kTileWords = TileSubquantizers * 256u / 4u;
  u32 ordinals[kMaxSlots];
  u32 static_mask = 0;
#pragma unroll
  for (u32 slot = 0; slot < kMaxSlots; ++slot) {
    const u32 index = threadIdx.x + slot * blockDim.x;
    ordinals[slot] = UINT32_MAX;
    u32 ordinal = 0;
    if (variant != 0 && index < count &&
        static_ordinal_from_raw(params, handles[index], ordinal) &&
        ordinal < params.num_nodes) {
      ordinals[slot] = ordinal;
      static_mask |= 1u << slot;
    }
  }
  f32 partials[kMaxSlots];
  // Each thread stages the same one or two float4 words of every tile; the
  // second word exists only for 128-thread CTAs (kTileWords == 256).
  const u32 word0 = threadIdx.x;
  const u32 word1 = threadIdx.x + 128u;
  float4 prefetch0;
  float4 prefetch1;
  auto issue_tile_loads = [&](u32 tile_base, u32 tile_rows) {
    const f32* source = query_lut + static_cast<size_t>(tile_base) * 256u;
    if (word0 < tile_rows * 64u) {
      prefetch0 = reinterpret_cast<const float4*>(source)[word0];
    }
    if (word1 < tile_rows * 64u) {
      prefetch1 = reinterpret_cast<const float4*>(source)[word1];
    }
  };
  if (variant == 2) {
    issue_tile_loads(0, min(TileSubquantizers, subquantizers));
  }
  for (u32 tile_base = 0; tile_base < subquantizers;
       tile_base += TileSubquantizers) {
    const u32 tile_rows = min(TileSubquantizers, subquantizers - tile_base);
    const f32* tile_source =
      query_lut + static_cast<size_t>(tile_base) * 256u;
    if (variant == 2) {
      if (tile_base != 0) __syncthreads();
      if (word0 < tile_rows * 64u) {
        reinterpret_cast<float4*>(lut_tile)[word0] = prefetch0;
      }
      if (word1 < tile_rows * 64u) {
        reinterpret_cast<float4*>(lut_tile)[word1] = prefetch1;
      }
      __syncthreads();
      // Kick off the next tile's global loads now so they overlap this
      // tile's scoring; the registers are consumed by the store at the top
      // of the next iteration.
      if (tile_base + TileSubquantizers < subquantizers) {
        issue_tile_loads(tile_base + TileSubquantizers,
                         min(TileSubquantizers,
                             subquantizers - tile_base - TileSubquantizers));
      }
    } else {
      for (u32 word = threadIdx.x; word < tile_rows * 64u;
           word += blockDim.x) {
        reinterpret_cast<float4*>(lut_tile)[word] =
          reinterpret_cast<const float4*>(tile_source)[word];
      }
      __syncthreads();
    }
    if (variant == 0) {
      __syncthreads();
      continue;
    }
#pragma unroll
    for (u32 slot = 0; slot < kMaxSlots; ++slot) {
      if ((static_mask & (1u << slot)) == 0) continue;
      const u8* code = params.pq_codes +
                       static_cast<size_t>(ordinals[slot]) * subquantizers;
      f32 distance = tile_base == 0 ? 0.0f : partials[slot];
      if (tile_rows == TileSubquantizers) {
        u32 packed = 0;
        const u8* at = code + tile_base;
        if ((reinterpret_cast<uintptr_t>(at) & 3u) == 0) {
          packed = *reinterpret_cast<const u32*>(at);
        } else {
          packed = static_cast<u32>(at[0]) |
                   (static_cast<u32>(at[1]) << 8u) |
                   (static_cast<u32>(at[2]) << 16u) |
                   (static_cast<u32>(at[3]) << 24u);
        }
        distance += lut_tile[0u * 256u + (packed & 0xffu)];
        distance += lut_tile[1u * 256u + ((packed >> 8u) & 0xffu)];
        distance += lut_tile[2u * 256u + ((packed >> 16u) & 0xffu)];
        distance += lut_tile[3u * 256u + (packed >> 24u)];
      } else {
        for (u32 row = 0; row < tile_rows; ++row) {
          distance += lut_tile[static_cast<size_t>(row) * 256u +
                               code[tile_base + row]];
        }
      }
      partials[slot] = distance;
    }
    if (variant != 2) __syncthreads();
  }
  if (variant == 0) return;
#pragma unroll
  for (u32 slot = 0; slot < kMaxSlots; ++slot) {
    if ((static_mask & (1u << slot)) == 0) continue;
    const u32 index = threadIdx.x + slot * blockDim.x;
    f32 distance = partials[slot];
    if (!(finite_f32_bits(distance) && distance < FLT_MAX)) {
      distance = approximate_entry_wide_fallback(
        query_lut,
        params.pq_codes + static_cast<size_t>(ordinals[slot]) * subquantizers,
        subquantizers);
    }
    distances[index] = distance;
  }
}

template <u32 PqSubquantizers>
__global__ void score_wave_kernel(PersistentKernelParams params,
                                 const f32* query_lut, const u64* handles,
                                 f32* distances, u32 count, u32 mode) {
  extern __shared__ f32 lut_tile[];
  constexpr u32 kBatch = gpu_search::kPersistentMaxMergeCandidates;
  const u32 subquantizers = params.pq_subquantizers;
  // Each CTA owns a private LUT slice and candidate wave.
  query_lut += static_cast<size_t>(blockIdx.x) * subquantizers * 256u;
  handles += static_cast<size_t>(blockIdx.x) * count;
  distances += static_cast<size_t>(blockIdx.x) * count;
  for (u32 batch = 0; batch * kBatch < count; ++batch) {
    const u32 base = batch * kBatch;
    const u32 batch_count = min(kBatch, count - base);
    if (mode >= 2) {
      for (u32 index = threadIdx.x; index < batch_count; index += blockDim.x) {
        distances[base + index] = FLT_MAX;
      }
      __syncthreads();
      if (mode == 2) {
        score_static_candidates_lut_tiled<PqSubquantizers>(
          params, query_lut, handles + base, batch_count, distances + base,
          lut_tile);
      } else {
        score_wave_lut_tiled_variant<
          PqSubquantizers, gpu_search::kPersistentPqLutTileSubquantizers>(
          params, query_lut, handles + base, batch_count, distances + base,
          lut_tile, mode - 3);
      }
      continue;
    }
    for (u32 index = threadIdx.x; index < batch_count; index += blockDim.x) {
      u32 ordinal = 0;
      f32 distance = FLT_MAX;
      if (static_ordinal_from_raw(params, handles[base + index], ordinal) &&
          ordinal < params.num_nodes) {
        const u8* code = params.pq_codes +
                         static_cast<size_t>(ordinal) * subquantizers;
        if (mode == 0) {
          distance = 0.0f;
          for (u32 subquantizer = 0; subquantizer < subquantizers;
               ++subquantizer) {
            distance += query_lut[static_cast<size_t>(subquantizer) * 256u +
                                  code[subquantizer]];
          }
        } else {
          distance =
            approximate_entry<PqSubquantizers>(params, query_lut, code);
        }
      }
      distances[base + index] = distance;
    }
  }
}

__global__ void compare_distances_kernel(PersistentKernelParams params,
                                         const f32* query_lut,
                                         const u64* handles, u32 count,
                                         const f32* actual,
                                         u32* mismatches) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  f32 expected = FLT_MAX;
  u32 ordinal = 0;
  if (static_ordinal_from_raw(params, handles[index], ordinal) &&
      ordinal < params.num_nodes) {
    expected = 0.0f;
    const u8* code = params.pq_codes +
                     static_cast<size_t>(ordinal) * params.pq_subquantizers;
    for (u32 subquantizer = 0; subquantizer < params.pq_subquantizers;
         ++subquantizer) {
      expected += query_lut[static_cast<size_t>(subquantizer) * 256u +
                            code[subquantizer]];
    }
  }
  if (__float_as_uint(actual[index]) != __float_as_uint(expected)) {
    atomicAdd(mismatches, 1u);
  }
}

double time_mode_ms(const PersistentKernelParams& params,
                    const f32* query_lut, const u64* handles, f32* distances,
                    u32 count, u32 iterations, u32 mode, u32 subquantizers,
                    u32 ctas) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate");
  check_cuda(cudaEventRecord(start), "cudaEventRecord start");
  for (u32 iteration = 0; iteration < iterations; ++iteration) {
    if (subquantizers == 64) {
      score_wave_kernel<gpu_search::kPersistentPq64Subquantizers>
        <<<ctas, 128, gpu_search::kPersistentPqLutTileBytes>>>(
        params, query_lut, handles, distances, count, mode);
    } else if (subquantizers == 256) {
      score_wave_kernel<gpu_search::kPersistentPq256Subquantizers>
        <<<ctas, 128, gpu_search::kPersistentPqLutTileBytes>>>(
        params, query_lut, handles, distances, count, mode);
    } else {
      score_wave_kernel<gpu_search::kPersistentRuntimePqSubquantizers>
        <<<ctas, 128, gpu_search::kPersistentPqLutTileBytes>>>(
        params, query_lut, handles, distances, count, mode);
    }
  }
  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "cudaEventElapsedTime");
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy");
  return static_cast<double>(elapsed_ms) / iterations;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
      std::printf("SKIP: no CUDA device available\n");
      return 0;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");

    const u32 subquantizers =
      argc > 1 ? static_cast<u32>(std::strtoul(argv[1], nullptr, 10)) : 256;
    const u32 default_candidates = subquantizers <= 64 ? 7973u : 5427u;
    const u32 candidates =
      argc > 2 ? static_cast<u32>(std::strtoul(argv[2], nullptr, 10))
               : default_candidates;
    const u32 iterations =
      argc > 3 ? static_cast<u32>(std::strtoul(argv[3], nullptr, 10)) : 200;
    const u32 ctas =
      argc > 4 ? static_cast<u32>(std::strtoul(argv[4], nullptr, 10)) : 1;
    const u32 table_entries =
      argc > 5 ? static_cast<u32>(std::strtoul(argv[5], nullptr, 10))
               : std::max<u32>(2u * candidates, 65536u);

    std::vector<f32> query_lut(static_cast<size_t>(ctas) * subquantizers *
                               256u);
    for (u32 index = 0; index < query_lut.size(); ++index) {
      const u32 offset = index % (subquantizers * 256u);
      query_lut[index] =
        static_cast<f32>(((offset / 256u) * 257u + (offset % 256u)) &
                         0xffffu) /
        1024.0f;
    }
    std::vector<u8> codes(static_cast<size_t>(table_entries) * subquantizers);
    u32 state = 0x9e3779b9u;
    for (u32 index = 0; index < codes.size(); ++index) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      codes[index] = static_cast<u8>(state & 0xffu);
    }
    std::vector<u64> handles(static_cast<size_t>(ctas) * candidates);
    for (u32 index = 0; index < handles.size(); ++index) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      // raw handle = ordinal + 1 under the 16-byte fake node stride.
      handles[index] = static_cast<u64>(state % table_entries) + 1u;
    }

    DeviceShardRegion shard{};
    shard.ordinal_base = 0;
    shard.node_count = table_entries;
    shard.node_base_offset = 16;
    shard.node_stride = 16;

    DeviceBuffer<f32> device_lut(query_lut.size());
    DeviceBuffer<u8> device_codes(codes.size());
    DeviceBuffer<u64> device_handles(handles.size());
    DeviceBuffer<f32> device_distances(static_cast<size_t>(ctas) * candidates);
    DeviceBuffer<u32> device_mismatches(1);
    DeviceBuffer<DeviceShardRegion> device_shards(1);
    check_cuda(cudaMemcpy(device_lut.get(), query_lut.data(),
                          query_lut.size() * sizeof(f32),
                          cudaMemcpyHostToDevice), "cudaMemcpy LUT H2D");
    check_cuda(cudaMemcpy(device_codes.get(), codes.data(),
                          codes.size() * sizeof(u8), cudaMemcpyHostToDevice),
               "cudaMemcpy codes H2D");
    check_cuda(cudaMemcpy(device_handles.get(), handles.data(),
                          handles.size() * sizeof(u64),
                          cudaMemcpyHostToDevice), "cudaMemcpy handles H2D");
    check_cuda(cudaMemcpy(device_shards.get(), &shard, sizeof(shard),
                          cudaMemcpyHostToDevice), "cudaMemcpy shard H2D");

    PersistentKernelParams params{};
    params.shards = device_shards.get();
    params.num_shards = 1;
    params.num_nodes = table_entries;
    params.pq_codes = device_codes.get();
    params.pq_subquantizers = subquantizers;

    std::printf(
      "pq=%u candidates=%u table_entries=%u iterations=%u ctas=%u "
      "lut_kib_per_cta=%zu\n",
      subquantizers, candidates, table_entries, iterations, ctas,
      static_cast<size_t>(subquantizers) * 256u * sizeof(f32) / 1024u);
    const char* mode_names[] = {"legacy-byte-global", "packed-global",
                                "lut-tile-shared", "lut-tile-copy-only",
                                "lut-tile-registers",
                                "lut-tile-reg-prefetch"};
    // Ramp the GPU out of idle clocks before any measurement; untimed.
    time_mode_ms(params, device_lut.get(), device_handles.get(),
                 device_distances.get(), candidates,
                 std::max(iterations, 500u), 1, subquantizers, ctas);
    time_mode_ms(params, device_lut.get(), device_handles.get(),
                 device_distances.get(), candidates,
                 std::max(iterations, 500u), 2, subquantizers, ctas);
    for (u32 mode = 0; mode < 6; ++mode) {
      // One untimed correctness/warmup pass.
      time_mode_ms(params, device_lut.get(), device_handles.get(),
                   device_distances.get(), candidates, 1, mode, subquantizers,
                   ctas);
      if (mode != 0 && mode != 3) {
        check_cuda(cudaMemset(device_mismatches.get(), 0, sizeof(u32)),
                   "cudaMemset mismatches");
        // CTA 0's wave is authoritative for the bitwise check.
        compare_distances_kernel<<<(candidates + 127u) / 128u, 128>>>(
          params, device_lut.get(), device_handles.get(), candidates,
          device_distances.get(), device_mismatches.get());
        check_cuda(cudaGetLastError(), "compare_distances_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "compare completion");
        u32 mismatches = 0;
        check_cuda(cudaMemcpy(&mismatches, device_mismatches.get(),
                              sizeof(u32), cudaMemcpyDeviceToHost),
                   "cudaMemcpy mismatches D2H");
        if (mismatches != 0) {
          std::fprintf(stderr, "mode=%s bitwise check FAILED: %u mismatches\n",
                       mode_names[mode], mismatches);
          return 1;
        }
      }
      const double per_query_ms =
        time_mode_ms(params, device_lut.get(), device_handles.get(),
                     device_distances.get(), candidates, iterations, mode,
                     subquantizers, ctas);
      std::printf("mode=%-22s per_query_ms=%.4f (%.1f us)\n",
                  mode_names[mode], per_query_ms, per_query_ms * 1000.0);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
