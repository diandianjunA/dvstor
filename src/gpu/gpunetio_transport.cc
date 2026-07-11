#include "gpu/gpunetio_transport.hh"

#include <cuda_runtime.h>

#ifndef IBV_WC_DRIVER1
#define IBV_WC_DRIVER1 135
#define IBV_WC_DRIVER2 136
#define IBV_WC_DRIVER3 137
#endif

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_mmap.h>
#include <doca_rdma_bridge.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include <library/connection_manager.hh>
#include <library/memory_region.hh>
#include <library/queue_pair.hh>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gpu/gpunetio_probe.hh"

namespace gpu {

namespace {

constexpr uint32_t kQueryQueueEntries = 1024;
constexpr size_t kGpuPageSize = 64 * 1024;
constexpr size_t kExternalQueueBytes = 128 * 1024;
constexpr size_t kExternalDbrBytes = 4 * 1024;

size_t align_up(const size_t value, const size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

uint32_t byte_swap32(uint32_t value) {
  return ((value & 0x000000ffU) << 24) | ((value & 0x0000ff00U) << 8) | ((value & 0x00ff0000U) >> 8) |
         ((value & 0xff000000U) >> 24);
}

[[noreturn]] void throw_doca(const char* what, doca_error_t status) {
  throw std::runtime_error(std::string(what) + ": " + doca_error_get_descr(status));
}

void check_doca(const char* what, doca_error_t status) {
  if (status != DOCA_SUCCESS) {
    throw_doca(what, status);
  }
}

void check_cuda(const char* what, cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void initialize_cq_owner_bits(void* cq_buffer, const size_t bytes) {
  std::vector<unsigned char> initial(bytes, 0);
  for (size_t offset = 63; offset < bytes; offset += 64) {
    initial[offset] =
      (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) | MLX5_CQE_OWNER_MASK;
  }
  check_cuda("cudaMemcpy(cq owner init)", cudaMemcpy(cq_buffer, initial.data(), bytes, cudaMemcpyHostToDevice));
}

std::string hex_u64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
}

const char* nic_handler_name(const doca_gpu_dev_verbs_nic_handler handler) {
  switch (handler) {
    case DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO:
      return "AUTO";
    case DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY:
      return "CPU_PROXY";
    case DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB:
      return "GPU_SM_DB";
    case DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF:
      return "GPU_SM_BF";
    default:
      return "UNKNOWN";
  }
}

void exchange_qp_info(Context& channel_context, QueuePair& channel_qp, const QPInfo& local_info, QPInfo& remote_info) {
  LocalMemoryRegion region{channel_context, &remote_info, sizeof(remote_info)};
  channel_qp.post_receive(region);
  channel_qp.post_send_inlined(&local_info, sizeof(local_info), IBV_WR_SEND);
  channel_context.poll_send_cq_until_completion();
  channel_context.receive();
}

char* gpu_pci_address(const uint32_t gpu_device, char (&bus_id)[32]) {
  check_cuda("cudaDeviceGetPCIBusId", cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), static_cast<int>(gpu_device)));
  return bus_id;
}

class DocaDevinfoList {
public:
  DocaDevinfoList() {
    check_doca("doca_devinfo_create_list", doca_devinfo_create_list(&infos_, &count_));
  }

  ~DocaDevinfoList() {
    if (infos_ != nullptr) doca_devinfo_destroy_list(infos_);
  }

  doca_devinfo* find(const char* ibdev_name) const {
    for (uint32_t i = 0; i < count_; ++i) {
      char current_ibdev[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {0};
      if (doca_devinfo_get_ibdev_name(infos_[i], current_ibdev,
                                     sizeof(current_ibdev)) != DOCA_SUCCESS) {
        continue;
      }
      if (std::strcmp(current_ibdev, ibdev_name) == 0) return infos_[i];
    }
    throw std::runtime_error(std::string("failed to find DOCA device for ibdev ") + ibdev_name);
  }

private:
  doca_devinfo** infos_{};
  uint32_t count_{};
};

void qp_modify_to_init(doca_verbs_qp* qp) {
  doca_verbs_qp_attr* attr = nullptr;
  check_doca("doca_verbs_qp_attr_create", doca_verbs_qp_attr_create(&attr));
  check_doca("doca_verbs_qp_attr_set_next_state", doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT));
  check_doca("doca_verbs_qp_attr_set_allow_remote_write", doca_verbs_qp_attr_set_allow_remote_write(attr, 1));
  check_doca("doca_verbs_qp_attr_set_allow_remote_read", doca_verbs_qp_attr_set_allow_remote_read(attr, 1));
  check_doca("doca_verbs_qp_attr_set_atomic_mode", doca_verbs_qp_attr_set_atomic_mode(attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC));
  check_doca("doca_verbs_qp_attr_set_pkey_index", doca_verbs_qp_attr_set_pkey_index(attr, 0));
  check_doca("doca_verbs_qp_attr_set_port_num", doca_verbs_qp_attr_set_port_num(attr, 1));
  check_doca("doca_verbs_qp_modify",
             doca_verbs_qp_modify(qp,
                                  attr,
                                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                    DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_ATOMIC_MODE |
                                    DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM));
  check_doca("doca_verbs_qp_attr_destroy", doca_verbs_qp_attr_destroy(attr));
}

void qp_modify_to_rtr(doca_verbs_context* verbs_context, doca_verbs_qp* qp, const QPInfo& remote_info) {
  doca_verbs_qp_attr* attr = nullptr;
  doca_verbs_ah_attr* ah_attr = nullptr;
  check_doca("doca_verbs_qp_attr_create", doca_verbs_qp_attr_create(&attr));
  check_doca("doca_verbs_ah_attr_create", doca_verbs_ah_attr_create(verbs_context, &ah_attr));
  check_doca("doca_verbs_ah_attr_set_addr_type",
             doca_verbs_ah_attr_set_addr_type(ah_attr, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
  check_doca("doca_verbs_ah_attr_set_dlid", doca_verbs_ah_attr_set_dlid(ah_attr, remote_info.lid));
  check_doca("doca_verbs_ah_attr_set_sl", doca_verbs_ah_attr_set_sl(ah_attr, 0));
  check_doca("doca_verbs_qp_attr_set_next_state", doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR));
  check_doca("doca_verbs_qp_attr_set_path_mtu", doca_verbs_qp_attr_set_path_mtu(attr, DOCA_MTU_SIZE_4K_BYTES));
  check_doca("doca_verbs_qp_attr_set_dest_qp_num", doca_verbs_qp_attr_set_dest_qp_num(attr, remote_info.qp_number));
  check_doca("doca_verbs_qp_attr_set_rq_psn", doca_verbs_qp_attr_set_rq_psn(attr, 0));
  check_doca("doca_verbs_qp_attr_set_max_dest_rd_atomic", doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, 16));
  check_doca("doca_verbs_qp_attr_set_min_rnr_timer", doca_verbs_qp_attr_set_min_rnr_timer(attr, 12));
  check_doca("doca_verbs_qp_attr_set_ah_attr", doca_verbs_qp_attr_set_ah_attr(attr, ah_attr));
  check_doca("doca_verbs_qp_modify",
             doca_verbs_qp_modify(qp,
                                  attr,
                                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                    DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                    DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                    DOCA_VERBS_QP_ATTR_AH_ATTR));
  check_doca("doca_verbs_ah_attr_destroy", doca_verbs_ah_attr_destroy(ah_attr));
  check_doca("doca_verbs_qp_attr_destroy", doca_verbs_qp_attr_destroy(attr));
}

void qp_modify_to_rts(doca_verbs_qp* qp) {
  doca_verbs_qp_attr* attr = nullptr;
  check_doca("doca_verbs_qp_attr_create", doca_verbs_qp_attr_create(&attr));
  check_doca("doca_verbs_qp_attr_set_next_state", doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTS));
  check_doca("doca_verbs_qp_attr_set_sq_psn", doca_verbs_qp_attr_set_sq_psn(attr, 0));
  check_doca("doca_verbs_qp_attr_set_ack_timeout", doca_verbs_qp_attr_set_ack_timeout(attr, 14));
  check_doca("doca_verbs_qp_attr_set_retry_cnt", doca_verbs_qp_attr_set_retry_cnt(attr, 7));
  check_doca("doca_verbs_qp_attr_set_rnr_retry", doca_verbs_qp_attr_set_rnr_retry(attr, 7));
  check_doca("doca_verbs_qp_attr_set_max_rd_atomic", doca_verbs_qp_attr_set_max_rd_atomic(attr, 16));
  check_doca("doca_verbs_qp_modify",
             doca_verbs_qp_modify(qp,
                                  attr,
                                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                    DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                    DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC));
  check_doca("doca_verbs_qp_attr_destroy", doca_verbs_qp_attr_destroy(attr));
}

}  // namespace

struct GpuNetioPersistentTransport::Impl {
  Impl(const configuration::IndexConfiguration& config,
       const size_t data_bytes,
       Context& context,
       ClientConnectionManager& cm,
       const MemoryRegionTokens& remote_regions)
      : qps_per_node(std::max<u32>(1, config.gpu_rdma_qps)),
        remote_region_count(static_cast<uint32_t>(remote_regions.size())) {
    if (data_bytes == 0 || remote_regions.empty()) {
      throw std::invalid_argument("GPUNetIO transport requires non-empty data and remote regions");
    }

    char pci_bus_id[32] = {0};
    const char* ibdev_name = ibv_get_device_name(context.get_raw_context()->device);
    check_cuda("cudaSetDevice", cudaSetDevice(static_cast<int>(config.gpu_device)));
    check_cuda("cudaFree(0)", cudaFree(nullptr));
    const char* gpu_pci = gpu_pci_address(config.gpu_device, pci_bus_id);
    constexpr doca_gpu_dev_verbs_nic_handler nic_handler =
      DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;

    {
      DocaDevinfoList devinfos;
      check_doca("doca_verbs_context_create",
                 doca_verbs_context_create(devinfos.find(ibdev_name),
                                           DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE,
                                           &verbs_context));
    }
    check_doca("doca_verbs_pd_create", doca_verbs_pd_create(verbs_context, &pd));
    ibv_pd* ibv_pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(pd);
    if (ibv_pd == nullptr) {
      throw std::runtime_error("doca_verbs_bridge_verbs_pd_get_ibv_pd returned null");
    }
    check_doca("doca_rdma_bridge_open_dev_from_pd",
               doca_rdma_bridge_open_dev_from_pd(ibv_pd, &dev));
    check_doca("doca_gpu_create", doca_gpu_create(gpu_pci, &gpu));

    for (uint32_t lane = 0; lane < std::max<uint32_t>(1, qps_per_node); ++lane) {
      for (uint32_t server = 0; server < remote_region_count; ++server) {
      doca_verbs_cq_attr* cq_attr = nullptr;
      doca_verbs_qp_init_attr* qp_init = nullptr;
      doca_verbs_cq* send_cq = nullptr;
      doca_verbs_cq* recv_cq = nullptr;
      doca_verbs_qp* qp = nullptr;
      doca_uar* external_uar = nullptr;
      doca_gpu_verbs_qp* gpu_qp = nullptr;
      doca_gpu_dev_verbs_qp* gpu_qp_dev = nullptr;
      void* send_cq_umem_buf = nullptr;
      void* recv_cq_umem_buf = nullptr;
      void* qp_wq_umem_buf = nullptr;
      void* qp_dbr_umem_buf = nullptr;
      doca_umem* send_cq_umem = nullptr;
      doca_umem* recv_cq_umem = nullptr;
      doca_umem* qp_wq_umem = nullptr;
      doca_umem* qp_dbr_umem = nullptr;

      check_doca("doca_verbs_cq_attr_create", doca_verbs_cq_attr_create(&cq_attr));
      check_doca("doca_verbs_cq_attr_set_entry_size",
                 doca_verbs_cq_attr_set_entry_size(cq_attr, DOCA_VERBS_CQ_ENTRY_SIZE_64));
      check_doca("doca_verbs_cq_attr_set_cq_size", doca_verbs_cq_attr_set_cq_size(cq_attr, kQueryQueueEntries));
      check_doca("doca_verbs_cq_attr_set_cq_overrun",
                 doca_verbs_cq_attr_set_cq_overrun(cq_attr, 1));
      check_doca("doca_gpu_mem_alloc(send_cq_umem)",
                 doca_gpu_mem_alloc(
                   gpu, kExternalQueueBytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU, &send_cq_umem_buf, nullptr));
      check_doca("doca_gpu_mem_alloc(recv_cq_umem)",
                 doca_gpu_mem_alloc(
                   gpu, kExternalQueueBytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU, &recv_cq_umem_buf, nullptr));
      initialize_cq_owner_bits(send_cq_umem_buf, kExternalQueueBytes);
      initialize_cq_owner_bits(recv_cq_umem_buf, kExternalQueueBytes);
      check_doca("doca_umem_gpu_create(send_cq)",
                 doca_umem_gpu_create(gpu,
                                      dev,
                                      send_cq_umem_buf,
                                      kExternalQueueBytes,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_READ |
                                        DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                      &send_cq_umem));
      check_doca("doca_umem_gpu_create(recv_cq)",
                 doca_umem_gpu_create(gpu,
                                      dev,
                                      recv_cq_umem_buf,
                                      kExternalQueueBytes,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_READ |
                                        DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                      &recv_cq_umem));
      check_doca("doca_verbs_cq_attr_set_external_datapath_en",
                 doca_verbs_cq_attr_set_external_datapath_en(cq_attr, 1));
      check_doca("doca_verbs_cq_attr_set_external_umem(send)",
                 doca_verbs_cq_attr_set_external_umem(cq_attr, send_cq_umem, 0));
      check_doca("doca_verbs_cq_create(send)", doca_verbs_cq_create(verbs_context, cq_attr, &send_cq));
      check_doca("doca_verbs_cq_attr_set_external_umem(recv)",
                 doca_verbs_cq_attr_set_external_umem(cq_attr, recv_cq_umem, 0));
      check_doca("doca_verbs_cq_create(recv)", doca_verbs_cq_create(verbs_context, cq_attr, &recv_cq));
      check_doca("doca_verbs_qp_init_attr_create", doca_verbs_qp_init_attr_create(&qp_init));
      check_doca("doca_verbs_qp_init_attr_set_pd", doca_verbs_qp_init_attr_set_pd(qp_init, pd));
      check_doca("doca_verbs_qp_init_attr_set_send_cq", doca_verbs_qp_init_attr_set_send_cq(qp_init, send_cq));
      check_doca("doca_verbs_qp_init_attr_set_receive_cq", doca_verbs_qp_init_attr_set_receive_cq(qp_init, recv_cq));
      check_doca("doca_verbs_qp_init_attr_set_sq_wr", doca_verbs_qp_init_attr_set_sq_wr(qp_init, kQueryQueueEntries));
      check_doca("doca_verbs_qp_init_attr_set_rq_wr", doca_verbs_qp_init_attr_set_rq_wr(qp_init, kQueryQueueEntries));
      check_doca("doca_verbs_qp_init_attr_set_send_max_sges",
                 doca_verbs_qp_init_attr_set_send_max_sges(qp_init, 1));
      check_doca("doca_verbs_qp_init_attr_set_receive_max_sges",
                 doca_verbs_qp_init_attr_set_receive_max_sges(qp_init, 1));
      check_doca("doca_verbs_qp_init_attr_set_max_inline_data",
                 doca_verbs_qp_init_attr_set_max_inline_data(qp_init, 0));
      check_doca("doca_verbs_qp_init_attr_set_qp_type",
                 doca_verbs_qp_init_attr_set_qp_type(qp_init, DOCA_VERBS_QP_TYPE_RC));
      check_doca("doca_gpu_mem_alloc(qp_wq_umem)",
                 doca_gpu_mem_alloc(
                   gpu, kExternalQueueBytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU, &qp_wq_umem_buf, nullptr));
      check_doca("doca_gpu_mem_alloc(qp_dbr_umem)",
                 doca_gpu_mem_alloc(
                   gpu, kExternalDbrBytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU, &qp_dbr_umem_buf, nullptr));
      check_cuda("cudaMemset(qp_wq_umem)", cudaMemset(qp_wq_umem_buf, 0, kExternalQueueBytes));
      check_cuda("cudaMemset(qp_dbr_umem)", cudaMemset(qp_dbr_umem_buf, 0, kExternalDbrBytes));
      check_doca("doca_umem_gpu_create(qp_wq)",
                 doca_umem_gpu_create(gpu,
                                      dev,
                                      qp_wq_umem_buf,
                                      kExternalQueueBytes,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_READ |
                                        DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                      &qp_wq_umem));
      check_doca("doca_umem_gpu_create(qp_dbr)",
                 doca_umem_gpu_create(gpu,
                                      dev,
                                      qp_dbr_umem_buf,
                                      kExternalDbrBytes,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_READ |
                                        DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                                      &qp_dbr_umem));
      check_doca("doca_verbs_qp_init_attr_set_external_datapath_en",
                 doca_verbs_qp_init_attr_set_external_datapath_en(qp_init, 1));
      check_doca("doca_verbs_qp_init_attr_set_external_umem",
                 doca_verbs_qp_init_attr_set_external_umem(qp_init, qp_wq_umem, 0));
      check_doca("doca_verbs_qp_init_attr_set_external_dbr_umem",
                 doca_verbs_qp_init_attr_set_external_dbr_umem(qp_init, qp_dbr_umem, 0));
      doca_error_t uar_status = doca_uar_create(
        dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &external_uar);
      if (uar_status != DOCA_SUCCESS) {
        uar_status = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &external_uar);
      }
      check_doca("doca_uar_create(GPU doorbell)", uar_status);
      check_doca("doca_verbs_qp_init_attr_set_external_uar",
                 doca_verbs_qp_init_attr_set_external_uar(qp_init, external_uar));
      check_doca("doca_verbs_qp_create", doca_verbs_qp_create(verbs_context, qp_init, &qp));

      send_cqs.push_back(send_cq);
      recv_cqs.push_back(recv_cq);
      qps.push_back(qp);
      std::cerr << "[STATUS]: exporting GPUNetIO QP resource=" << 0
                << " lane=" << lane
                << " server=" << server << " qpn=" << doca_verbs_qp_get_qpn(qp)
                << " gpu_pci=" << gpu_pci << " ibdev=" << ibdev_name
                << " handler=" << nic_handler_name(nic_handler) << std::endl;
      check_doca("doca_gpu_verbs_export_qp",
                 doca_gpu_verbs_export_qp(gpu,
                                         dev,
                                         qp,
                                         nic_handler,
                                         qp_wq_umem_buf,
                                         send_cq,
                                         recv_cq,
                                         &gpu_qp));
      check_doca("doca_gpu_verbs_get_qp_dev", doca_gpu_verbs_get_qp_dev(gpu_qp, &gpu_qp_dev));
      uint8_t cpu_proxy_enabled = 0;
      check_doca("doca_gpu_verbs_cpu_proxy_enabled",
                 doca_gpu_verbs_cpu_proxy_enabled(gpu_qp, &cpu_proxy_enabled));
      if (cpu_proxy_enabled != 0) {
        throw std::runtime_error(
          "GPUNetIO exported a CPU-proxy QP; the GPU-only query engine requires GPU doorbells");
      }

      qp_modify_to_init(qp);
      const QPInfo local_info{context.get_lid(), doca_verbs_qp_get_qpn(qp)};
      QPInfo remote_info{};
      exchange_qp_info(context, *cm.server_qps[server], local_info, remote_info);
      qp_modify_to_rtr(verbs_context, qp, remote_info);
      qp_modify_to_rts(qp);

      gpu_qps.push_back(gpu_qp);
      gpu_qp_devices_host.push_back(gpu_qp_dev);
      external_uars.push_back(external_uar);
      external_umems.push_back(send_cq_umem);
      external_umems.push_back(recv_cq_umem);
      external_umems.push_back(qp_wq_umem);
      external_umems.push_back(qp_dbr_umem);
      external_umem_buffers.push_back(send_cq_umem_buf);
      external_umem_buffers.push_back(recv_cq_umem_buf);
      external_umem_buffers.push_back(qp_wq_umem_buf);
      external_umem_buffers.push_back(qp_dbr_umem_buf);
      check_doca("doca_verbs_qp_init_attr_destroy", doca_verbs_qp_init_attr_destroy(qp_init));
      check_doca("doca_verbs_cq_attr_destroy", doca_verbs_cq_attr_destroy(cq_attr));
      }
    }


    const size_t control_bytes =
      2 * sizeof(uint64_t) + sizeof(int) +
      kGpuNetioProbeDebugValueCount * sizeof(uint64_t) + 256;
    const size_t registered_bytes =
      align_up(control_bytes + kGpuPageSize, kGpuPageSize) +
      align_up(data_bytes, kGpuPageSize);

    check_doca("doca_gpu_mem_alloc",
               doca_gpu_mem_alloc(
                 gpu, registered_bytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU,
                 &registered_base, nullptr));
    const int mr_access = IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    errno = 0;
    registered_mr = ibv_reg_mr(
      ibv_pd, registered_base, registered_bytes, mr_access);
    const int peer_memory_error = errno;
    if (registered_mr != nullptr) {
      local_iova_base = 0;
      std::cerr << "[STATUS]: GPUNetIO GPU MR registration=peer_memory bytes="
                << registered_bytes << std::endl;
    } else {
      check_doca("doca_gpu_dmabuf_fd",
                 doca_gpu_dmabuf_fd(
                   gpu, registered_base, registered_bytes, &dmabuf_fd));
      registered_mr = mlx5dv_reg_dmabuf_mr(
        ibv_pd, 0, registered_bytes, 0, dmabuf_fd, mr_access, 0);
      local_iova_base = reinterpret_cast<uint64_t>(registered_base);
    }
    if (registered_mr == nullptr) {
      throw std::runtime_error(
        std::string("GPU MR registration failed: peer_memory=") +
        std::strerror(peer_memory_error) + ", dmabuf=" + std::strerror(errno));
    }
    if (local_iova_base != 0) {
      std::cerr << "[STATUS]: GPUNetIO GPU MR registration=dmabuf bytes="
                << registered_bytes << " peer_memory_error="
                << std::strerror(peer_memory_error) << std::endl;
    }
    local_mkey = registered_mr->lkey;
    local_mkey_wqe = byte_swap32(local_mkey);

    size_t offset = 0;
    auto allocate = [&](const size_t bytes, const size_t alignment) -> void* {
      offset = align_up(offset, alignment);
      auto* pointer = static_cast<unsigned char*>(registered_base) + offset;
      offset += bytes;
      return pointer;
    };

    d_probe_value = static_cast<uint64_t*>(
      allocate(sizeof(uint64_t), alignof(uint64_t)));
    d_dump = static_cast<unsigned char*>(
      allocate(sizeof(uint64_t), alignof(uint64_t)));
    d_probe_status = static_cast<int*>(
      allocate(sizeof(int), alignof(int)));
    d_probe_debug = static_cast<uint64_t*>(
      allocate(kGpuNetioProbeDebugValueCount * sizeof(uint64_t),
               alignof(uint64_t)));
    persistent_data = static_cast<unsigned char*>(
      allocate(data_bytes, kGpuPageSize));
    persistent_data_size = data_bytes;
    if (offset > registered_bytes) {
      throw std::logic_error("GPUNetIO registered allocation layout overflow");
    }

    remote_regions_host.resize(remote_region_count);
    for (uint32_t i = 0; i < remote_region_count; ++i) {
      remote_regions_host[i] = {
        .address = remote_regions[i]->address,
        .rkey = byte_swap32(remote_regions[i]->rkey),
        .reserved = remote_regions[i]->rkey,
      };
    }
    check_cuda("cudaMalloc(remote_regions)", cudaMalloc(&d_remote_regions, remote_regions_host.size() * sizeof(GpuNetioRemoteMemoryRegion)));
    check_cuda("cudaMemcpy(remote_regions)",
               cudaMemcpy(d_remote_regions,
                          remote_regions_host.data(),
                          remote_regions_host.size() * sizeof(GpuNetioRemoteMemoryRegion),
                          cudaMemcpyHostToDevice));

    check_cuda("cudaMalloc(qp_array)", cudaMalloc(&d_qp_array, gpu_qp_devices_host.size() * sizeof(void*)));
    check_cuda("cudaMemcpy(qp_array)",
               cudaMemcpy(d_qp_array,
                          gpu_qp_devices_host.data(),
                          gpu_qp_devices_host.size() * sizeof(void*),
                          cudaMemcpyHostToDevice));
    check_cuda("cudaMalloc(qp_locks)",
               cudaMalloc(&d_qp_locks, gpu_qp_devices_host.size() * sizeof(int)));
    check_cuda("cudaMemset(qp_locks)",
               cudaMemset(d_qp_locks, 0, gpu_qp_devices_host.size() * sizeof(int)));

    check_cuda("cudaStreamCreate", cudaStreamCreate(&stream));
    if (data_bytes > 0) {
      for (uint32_t qp_index = 0; qp_index < gpu_qp_devices_host.size(); ++qp_index) {
        check_cuda("cudaMemset(GPUNetIO probe status)",
                   cudaMemset(d_probe_status, 0, sizeof(int)));
        check_cuda("cudaMemset(GPUNetIO probe debug)",
                   cudaMemset(d_probe_debug, 0,
                              kGpuNetioProbeDebugValueCount * sizeof(uint64_t)));
        launch_gpunetio_read_probe(stream, GpuNetioReadProbeParams{
          .local_mkey = local_mkey_wqe,
          .local_iova_base = local_iova_base,
          .remote_regions = d_remote_regions,
          .remote_region_count = remote_region_count,
          .qp_array = d_qp_array,
          .qp_count = static_cast<uint32_t>(gpu_qp_devices_host.size()),
          .qp_index = qp_index,
          .remote_region = qp_index % remote_region_count,
          .destination = reinterpret_cast<unsigned char*>(d_probe_value),
          .dump_ptr = d_dump,
          .status_code = d_probe_status,
          .debug_values = d_probe_debug,
        });
        check_cuda("launch_gpunetio_read_probe", cudaGetLastError());
        check_cuda("cudaStreamSynchronize(GPUNetIO probe)", cudaStreamSynchronize(stream));
        int probe_status = 0;
        uint64_t probe_debug[kGpuNetioProbeDebugValueCount]{};
        check_cuda("cudaMemcpy(GPUNetIO probe status)",
                   cudaMemcpy(&probe_status, d_probe_status,
                              sizeof(probe_status), cudaMemcpyDeviceToHost));
        check_cuda("cudaMemcpy(GPUNetIO probe debug)",
                   cudaMemcpy(probe_debug, d_probe_debug,
                              sizeof(probe_debug), cudaMemcpyDeviceToHost));
        if (probe_status != 0) {
          throw std::runtime_error(
            "GPUNetIO startup RDMA read probe failed: qp=" +
            std::to_string(qp_index) +
            " remote_region=" + std::to_string(qp_index % remote_region_count) +
            " status=" + std::to_string(probe_status) +
            " remote=" + hex_u64(probe_debug[0]) +
            " rkey=" + std::to_string(probe_debug[1]) +
            " local_iova=" + hex_u64(probe_debug[2]) +
            " lkey=" + std::to_string(probe_debug[3]) +
            " ticket=" + std::to_string(probe_debug[4]) +
            " cqe_debug=" + hex_u64(probe_debug[6]) +
            " cqe_ticket_index=" + hex_u64(probe_debug[7]) +
            " cq_consumer=" + std::to_string(probe_debug[8]) +
            " cq_opown=" + hex_u64(probe_debug[9]) +
            " probe_stage=" + std::to_string(probe_debug[38]) +
            " read_dump_opown=" + hex_u64(probe_debug[39]) +
            " qp_flags=" + hex_u64(probe_debug[5]) +
            " sq_pre=" + std::to_string(probe_debug[10]) + "/" +
              std::to_string(probe_debug[11]) + "/" + std::to_string(probe_debug[12]) +
            " sq_post=" + std::to_string(probe_debug[18]) + "/" +
              std::to_string(probe_debug[19]) + "/" + std::to_string(probe_debug[20]) +
            " sq_wqe=" + hex_u64(probe_debug[13]) +
            " sq_dbr=" + hex_u64(probe_debug[14]) +
            " sq_db=" + hex_u64(probe_debug[15]) +
            " cq_addr=" + hex_u64(probe_debug[16]) +
            " queue_shape=" + hex_u64(probe_debug[17]) +
            " dbr_value=" + hex_u64(probe_debug[21]) +
            " read_wqe=" + hex_u64(probe_debug[22]) + "/" + hex_u64(probe_debug[23]) + "/" +
              hex_u64(probe_debug[24]) + "/" + hex_u64(probe_debug[25]) + "/" +
              hex_u64(probe_debug[26]) + "/" + hex_u64(probe_debug[27]) + "/" +
              hex_u64(probe_debug[28]) + "/" + hex_u64(probe_debug[29]) +
            " final_wqe=" + hex_u64(probe_debug[30]) + "/" + hex_u64(probe_debug[31]) + "/" +
              hex_u64(probe_debug[32]) + "/" + hex_u64(probe_debug[33]) + "/" +
              hex_u64(probe_debug[34]) + "/" + hex_u64(probe_debug[35]) + "/" +
              hex_u64(probe_debug[36]) + "/" + hex_u64(probe_debug[37]));
        }
      }
      std::cerr << "[STATUS]: GPUNetIO startup RDMA read probe passed for "
                << gpu_qp_devices_host.size() << " QPs\n";
      std::cerr << "[STATUS]: GPUNetIO RDMA Read implementation=manual_wqe_locked\n";
    }
  }

  ~Impl() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (d_qp_array != nullptr) {
      cudaFree(d_qp_array);
    }
    if (d_qp_locks != nullptr) {
      cudaFree(d_qp_locks);
    }
    if (d_remote_regions != nullptr) {
      cudaFree(d_remote_regions);
    }
    if (registered_mr != nullptr) {
      ibv_dereg_mr(registered_mr);
    }
    if (dmabuf_fd >= 0) {
      close(dmabuf_fd);
    }
    if (registered_base != nullptr && gpu != nullptr) {
      doca_gpu_mem_free(gpu, registered_base);
    }
    for (size_t i = 0; i < gpu_qps.size(); ++i) {
      if (gpu != nullptr && gpu_qps[i] != nullptr) {
        doca_gpu_verbs_unexport_qp(gpu, gpu_qps[i]);
      }
    }
    for (auto* qp : qps) {
      if (qp != nullptr) {
        doca_verbs_qp_destroy(qp);
      }
    }
    for (auto* uar : external_uars) {
      if (uar != nullptr) {
        doca_uar_destroy(uar);
      }
    }
    for (auto* cq : send_cqs) {
      if (cq != nullptr) {
        doca_verbs_cq_destroy(cq);
      }
    }
    for (auto* cq : recv_cqs) {
      if (cq != nullptr) {
        doca_verbs_cq_destroy(cq);
      }
    }
    for (auto* umem : external_umems) {
      if (umem != nullptr) {
        doca_umem_destroy(umem);
      }
    }
    for (auto* buffer : external_umem_buffers) {
      if (gpu != nullptr && buffer != nullptr) {
        doca_gpu_mem_free(gpu, buffer);
      }
    }
    if (pd != nullptr) {
      doca_verbs_pd_destroy(pd);
    }
    if (verbs_context != nullptr) {
      doca_verbs_context_destroy(verbs_context);
    }
    if (gpu != nullptr) {
      doca_gpu_destroy(gpu);
    }
    if (dev != nullptr) {
      doca_dev_close(dev);
    }
  }

  u32 qps_per_node{};
  uint32_t remote_region_count{};
  doca_verbs_context* verbs_context{nullptr};
  doca_verbs_pd* pd{nullptr};
  doca_dev* dev{nullptr};
  doca_gpu* gpu{nullptr};
  vec<doca_verbs_cq*> send_cqs;
  vec<doca_verbs_cq*> recv_cqs;
  vec<doca_verbs_qp*> qps;
  vec<doca_uar*> external_uars;
  vec<doca_gpu_verbs_qp*> gpu_qps;
  vec<doca_umem*> external_umems;
  vec<void*> external_umem_buffers;
  vec<void*> gpu_qp_devices_host;
  void** d_qp_array{nullptr};
  int* d_qp_locks{nullptr};
  ibv_mr* registered_mr{nullptr};
  void* registered_base{nullptr};
  int dmabuf_fd{-1};
  uint32_t local_mkey{};
  uint32_t local_mkey_wqe{};
  uint64_t local_iova_base{};
  vec<GpuNetioRemoteMemoryRegion> remote_regions_host;
  GpuNetioRemoteMemoryRegion* d_remote_regions{nullptr};
  uint64_t* d_probe_value{nullptr};
  unsigned char* d_dump{nullptr};
  int* d_probe_status{nullptr};
  uint64_t* d_probe_debug{nullptr};
  unsigned char* persistent_data{nullptr};
  size_t persistent_data_size{};
  cudaStream_t stream{nullptr};
};

GpuNetioPersistentTransport::GpuNetioPersistentTransport(
    const configuration::IndexConfiguration& config,
    const size_t data_bytes,
    Context& context,
    ClientConnectionManager& cm,
    const MemoryRegionTokens& remote_regions)
    : impl_(std::make_unique<Impl>(
        config, data_bytes, context, cm, remote_regions)) {}

GpuNetioPersistentTransport::~GpuNetioPersistentTransport() = default;

GpuNetioPersistentView GpuNetioPersistentTransport::view() const {
  return {
    .qp_array = impl_->d_qp_array,
    .remote_regions = impl_->d_remote_regions,
    .remote_region_count = impl_->remote_region_count,
    .qps_per_node = impl_->qps_per_node,
    .qp_locks = impl_->d_qp_locks,
    .local_mkey = impl_->local_mkey_wqe,
    .local_iova_base = impl_->local_iova_base,
    .data = impl_->persistent_data,
    .data_bytes = impl_->persistent_data_size,
    .dump = impl_->d_dump,
  };
}

}  // namespace gpu
