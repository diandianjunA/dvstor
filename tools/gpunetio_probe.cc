#include <cuda_runtime.h>

#ifndef IBV_WC_DRIVER1
#define IBV_WC_DRIVER1 135
#define IBV_WC_DRIVER2 136
#define IBV_WC_DRIVER3 137
#endif

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_log.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#include <infiniband/mlx5dv.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void check_cuda(const char* operation, cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_doca(const char* operation, doca_error_t status) {
  if (status != DOCA_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": " + doca_error_get_descr(status));
  }
}

struct ProbeResources {
  ~ProbeResources() {
    for (ibv_mr* memory_region : memory_regions) {
      if (memory_region != nullptr) ibv_dereg_mr(memory_region);
    }
    if (gpu_umem != nullptr) doca_umem_destroy(gpu_umem);
    if (dmabuf_fd >= 0) close(dmabuf_fd);
    if (gpu_memory != nullptr && gpu != nullptr) doca_gpu_mem_free(gpu, gpu_memory);
    if (gpu != nullptr) doca_gpu_destroy(gpu);
    if (device_attributes != nullptr) doca_verbs_device_attr_free(device_attributes);
    if (device != nullptr) doca_dev_close(device);
    if (protection_domain != nullptr) doca_verbs_pd_destroy(protection_domain);
    if (verbs_context != nullptr) doca_verbs_context_destroy(verbs_context);
    if (device_infos != nullptr) doca_devinfo_destroy_list(device_infos);
  }

  doca_devinfo** device_infos{};
  uint32_t device_count{};
  doca_verbs_context* verbs_context{};
  doca_verbs_pd* protection_domain{};
  doca_dev* device{};
  doca_verbs_device_attr* device_attributes{};
  doca_gpu* gpu{};
  void* gpu_memory{};
  doca_umem* gpu_umem{};
  int dmabuf_fd{-1};
  std::vector<ibv_mr*> memory_regions{};
};

doca_devinfo* find_device(ProbeResources& resources, const std::string& requested_name,
                          std::string& selected_name) {
  check_doca("doca_devinfo_create_list",
             doca_devinfo_create_list(&resources.device_infos, &resources.device_count));
  for (uint32_t index = 0; index < resources.device_count; ++index) {
    char name[DOCA_DEVINFO_IBDEV_NAME_SIZE]{};
    if (doca_devinfo_get_ibdev_name(resources.device_infos[index], name,
                                    sizeof(name)) != DOCA_SUCCESS) {
      continue;
    }
    if (requested_name.empty() || requested_name == name) {
      selected_name = name;
      return resources.device_infos[index];
    }
  }
  throw std::runtime_error(requested_name.empty()
    ? "no DOCA InfiniBand device was found"
    : "DOCA device not found: " + requested_name);
}

}  // namespace

int main(int argc, char** argv) {
  doca_log_backend* sdk_log_backend = nullptr;
  (void)doca_log_backend_create_with_file_sdk(stderr, &sdk_log_backend);
  if (sdk_log_backend != nullptr) {
    (void)doca_log_backend_set_sdk_level(sdk_log_backend, DOCA_LOG_LEVEL_TRACE);
  }
  try {
    const int gpu_index = argc > 1 ? std::stoi(argv[1]) : 0;
    const std::string requested_ibdev = argc > 2 ? argv[2] : "";
    const size_t allocation_bytes = argc > 3
      ? static_cast<size_t>(std::stoull(argv[3])) : 64 * 1024;
    const size_t registration_bytes = argc > 4
      ? static_cast<size_t>(std::stoull(argv[4])) : allocation_bytes;
    const std::string registration_mode = argc > 5 ? argv[5] : "dmabuf";
    if (allocation_bytes == 0 || allocation_bytes > std::numeric_limits<size_t>::max() - 65535) {
      throw std::invalid_argument("allocation size must be a positive byte count");
    }
    if (registration_bytes == 0 || registration_bytes % (64 * 1024) != 0) {
      throw std::invalid_argument("registration chunk must be a positive multiple of 65536 bytes");
    }
    if (registration_mode != "dmabuf" && registration_mode != "peer") {
      throw std::invalid_argument("registration mode must be dmabuf or peer");
    }
    ProbeResources resources;

    check_cuda("cudaSetDevice", cudaSetDevice(gpu_index));
    check_cuda("cudaFree(0)", cudaFree(nullptr));
    char gpu_bus_id[32]{};
    check_cuda("cudaDeviceGetPCIBusId",
               cudaDeviceGetPCIBusId(gpu_bus_id, sizeof(gpu_bus_id), gpu_index));

    std::string ibdev_name;
    doca_devinfo* device_info = find_device(resources, requested_ibdev, ibdev_name);
    check_doca("doca_verbs_context_create",
               doca_verbs_context_create(device_info,
                                         DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE,
                                         &resources.verbs_context));
    check_doca("doca_verbs_query_device",
               doca_verbs_query_device(resources.verbs_context,
                                       &resources.device_attributes));
    check_doca("GPU external datapath capability",
               doca_verbs_device_attr_get_is_gpu_external_datapath_supported(
                 resources.device_attributes));
    check_doca("RC QP capability",
               doca_verbs_device_attr_get_is_qp_type_supported(
                 resources.device_attributes, DOCA_VERBS_QP_TYPE_RC));
    check_doca("doca_verbs_pd_create",
               doca_verbs_pd_create(resources.verbs_context,
                                    &resources.protection_domain));
    check_doca("doca_verbs_pd_as_doca_dev",
               doca_verbs_pd_as_doca_dev(resources.protection_domain,
                                         &resources.device));
    check_doca("doca_gpu_create", doca_gpu_create(gpu_bus_id, &resources.gpu));

    std::cout << "Registering " << allocation_bytes << " GPU bytes\n";
    check_doca("doca_gpu_mem_alloc",
               doca_gpu_mem_alloc(resources.gpu, allocation_bytes, 64 * 1024,
                                  DOCA_GPU_MEM_TYPE_GPU, &resources.gpu_memory, nullptr));
    std::cout << "  doca_gpu_mem_alloc: passed" << std::endl;
    if (registration_mode == "dmabuf") {
      check_doca("doca_gpu_dmabuf_fd",
                 doca_gpu_dmabuf_fd(resources.gpu, resources.gpu_memory,
                                    allocation_bytes, &resources.dmabuf_fd));
      std::cout << "  doca_gpu_dmabuf_fd: passed" << std::endl;
    }

    ibv_pd* verbs_pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(resources.protection_domain);
    if (verbs_pd == nullptr) {
      throw std::runtime_error("doca_verbs_bridge_verbs_pd_get_ibv_pd returned null");
    }
    for (size_t offset = 0; offset < allocation_bytes; offset += registration_bytes) {
      const size_t bytes = std::min(registration_bytes, allocation_bytes - offset);
      const int access = IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
      ibv_mr* memory_region = registration_mode == "dmabuf"
        ? mlx5dv_reg_dmabuf_mr(
            verbs_pd, offset, bytes, offset, resources.dmabuf_fd, access, 0)
        : ibv_reg_mr(
            verbs_pd, static_cast<unsigned char*>(resources.gpu_memory) + offset,
            bytes, access);
      if (memory_region == nullptr) {
        throw std::runtime_error(
          registration_mode + " MR registration(offset=" + std::to_string(offset) +
          ", bytes=" + std::to_string(bytes) + "): " + std::strerror(errno));
      }
      resources.memory_regions.push_back(memory_region);
    }
    const size_t registered_mr_count = resources.memory_regions.size();
    const uint32_t first_lkey = resources.memory_regions.front()->lkey;
    std::cout << "  mlx5 GPU MR registration: passed" << std::endl;
    for (ibv_mr* memory_region : resources.memory_regions) {
      if (ibv_dereg_mr(memory_region) != 0) {
        throw std::runtime_error(std::string("ibv_dereg_mr: ") + std::strerror(errno));
      }
    }
    resources.memory_regions.clear();
    std::cout << "  mlx5 GPU MR deregistration: passed" << std::endl;

    if (registration_mode == "dmabuf") {
      ibv_context* verbs_context =
        doca_verbs_bridge_get_ibv_ctx(resources.verbs_context);
      if (verbs_context == nullptr) {
        throw std::runtime_error("doca_verbs_bridge_get_ibv_ctx returned null");
      }
      mlx5dv_devx_umem_in devx_umem_input{};
      devx_umem_input.addr = nullptr;
      devx_umem_input.size = std::min<size_t>(allocation_bytes, 64 * 1024);
      devx_umem_input.access = IBV_ACCESS_LOCAL_WRITE;
      devx_umem_input.comp_mask = MLX5DV_UMEM_MASK_DMABUF;
      devx_umem_input.dmabuf_fd = resources.dmabuf_fd;
      errno = 0;
      mlx5dv_devx_umem* devx_umem =
        mlx5dv_devx_umem_reg_ex(verbs_context, &devx_umem_input);
      if (devx_umem == nullptr) {
        const int saved_errno = errno;
        std::cout << "  mlx5 DevX DMA-BUF UMEM registration: failed: errno="
                  << saved_errno << " (" << std::strerror(saved_errno) << ')'
                  << std::endl;
      } else {
        std::cout << "  mlx5 DevX DMA-BUF UMEM registration: passed" << std::endl;
        if (mlx5dv_devx_umem_dereg(devx_umem) != 0) {
          throw std::runtime_error(std::string("mlx5dv_devx_umem_dereg: ") +
                                   std::strerror(errno));
        }
        std::cout << "  mlx5 DevX DMA-BUF UMEM deregistration: passed" << std::endl;
      }
    }

    // Some DOCA failure paths populate the output pointer before reporting an
    // error. Keep that partial handle out of ProbeResources so stack unwinding
    // does not call doca_umem_destroy() on an invalid object and hide the
    // original registration error with a segmentation fault.
    doca_umem* created_gpu_umem = nullptr;
    check_doca("doca_umem_gpu_create",
               doca_umem_gpu_create(resources.gpu, resources.device,
                                    resources.gpu_memory,
                                    std::min<size_t>(allocation_bytes, 64 * 1024),
                                    DOCA_ACCESS_FLAG_LOCAL_READ_WRITE,
                                    &created_gpu_umem));
    resources.gpu_umem = created_gpu_umem;
    std::cout << "  doca_umem_gpu_create: passed" << std::endl;

    // Relinquish ownership before destruction so an error return cannot cause
    // the resource destructor to attempt a second destroy during unwinding.
    created_gpu_umem = resources.gpu_umem;
    resources.gpu_umem = nullptr;
    check_doca("doca_umem_destroy", doca_umem_destroy(created_gpu_umem));
    std::cout << "  doca_umem_destroy: passed" << std::endl;

    std::cout << "GPUNetIO probe passed\n"
              << "  GPU: " << gpu_index << " (" << gpu_bus_id << ")\n"
              << "  RDMA device: " << ibdev_name << "\n"
              << "  GPU external datapath: supported\n"
              << "  RC QP: supported\n"
              << "  DOCA GPU UMEM registration: passed\n"
              << "  mlx5 GPU MR registration: passed\n"
              << "  registration mode: " << registration_mode << '\n'
              << "  registered bytes: " << allocation_bytes << '\n'
              << "  registration chunk bytes: " << registration_bytes << '\n'
              << "  memory regions: " << registered_mr_count << '\n'
              << "  first local mkey: " << first_lkey << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "GPUNetIO probe failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
