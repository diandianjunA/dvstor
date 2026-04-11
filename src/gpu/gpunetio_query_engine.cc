#include "gpu/gpunetio_query_engine.hh"

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
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gpu/gpunetio_query_launcher.hh"
#include "vamana/vamana_node.hh"

namespace gpu {

namespace {

constexpr uint32_t kQueryQueueEntries = 1024;
constexpr size_t kGpuPageSize = 64 * 1024;
constexpr size_t kExternalQueueBytes = 128 * 1024;
constexpr size_t kExternalDbrBytes = 4 * 1024;
constexpr size_t kGpuQpUmemBytes = 64 * 1024;

uint32_t bounded_search_visits(const configuration::IndexConfiguration& config) {
  const uint64_t expanded_nodes = std::max<uint32_t>(config.beam_width, 1);
  const uint64_t discovered_neighbors = expanded_nodes * std::max<uint32_t>(config.R, 1);
  const uint64_t scratch_visits = 1 + expanded_nodes + discovered_neighbors + config.k;
  return static_cast<uint32_t>(std::min<uint64_t>(scratch_visits, config.max_vectors));
}

uint32_t max_rdma_reads_per_query(const uint32_t max_visited,
                                  const uint32_t beam_width,
                                  const uint32_t top_k) {
  const uint64_t reads = 2ULL + static_cast<uint64_t>(max_visited) +
                         static_cast<uint64_t>(beam_width) + static_cast<uint64_t>(top_k);
  return static_cast<uint32_t>(std::min<uint64_t>(reads, std::numeric_limits<uint32_t>::max()));
}

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
    initial[offset] = 1;
  }
  check_cuda("cudaMemcpy(cq owner init)", cudaMemcpy(cq_buffer, initial.data(), bytes, cudaMemcpyHostToDevice));
}

std::string hex_u64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
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

doca_devinfo* find_doca_devinfo(const char* ibdev_name) {
  doca_devinfo** infos = nullptr;
  uint32_t count = 0;
  check_doca("doca_devinfo_create_list", doca_devinfo_create_list(&infos, &count));
  for (uint32_t i = 0; i < count; ++i) {
    char current_ibdev[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {0};
    if (doca_devinfo_get_ibdev_name(infos[i], current_ibdev, sizeof(current_ibdev)) != DOCA_SUCCESS) {
      continue;
    }
    if (std::strcmp(current_ibdev, ibdev_name) == 0) {
      return infos[i];
    }
  }
  throw std::runtime_error(std::string("failed to find DOCA device for ibdev ") + ibdev_name);
}

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

struct GpuNetioQueryPool::Resource {
  explicit Resource(const configuration::IndexConfiguration& config,
                    const uint32_t resource_id,
                    Context& context,
                    ClientConnectionManager& cm,
                    const MemoryRegionTokens& remote_regions)
      : dim(config.dim),
        beam_width(config.beam_width),
        max_results(config.k),
        max_visited(bounded_search_visits(config)),
        max_degree(config.R),
        node_size(static_cast<uint32_t>(VamanaNode::total_size())),
        remote_region_count(static_cast<uint32_t>(remote_regions.size())) {
    char pci_bus_id[32] = {0};
    const char* ibdev_name = ibv_get_device_name(context.get_raw_context()->device);
    check_cuda("cudaSetDevice", cudaSetDevice(static_cast<int>(config.gpu_device)));
    check_cuda("cudaFree(0)", cudaFree(nullptr));
    const char* gpu_pci = gpu_pci_address(config.gpu_device, pci_bus_id);

    doca_devinfo* devinfo = find_doca_devinfo(ibdev_name);
    check_doca("doca_verbs_context_create",
               doca_verbs_context_create(devinfo, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, &verbs_context));
    check_doca("doca_verbs_pd_create", doca_verbs_pd_create(verbs_context, &pd));
    check_doca("doca_verbs_pd_as_doca_dev", doca_verbs_pd_as_doca_dev(pd, &dev));
    check_doca("doca_gpu_create", doca_gpu_create(gpu_pci, &gpu));

    for (uint32_t server = 0; server < remote_region_count; ++server) {
      doca_verbs_cq_attr* cq_attr = nullptr;
      doca_verbs_qp_init_attr* qp_init = nullptr;
      doca_verbs_cq* send_cq = nullptr;
      doca_verbs_cq* recv_cq = nullptr;
      doca_verbs_qp* qp = nullptr;
      doca_gpu_verbs_qp* gpu_qp = nullptr;
      doca_gpu_dev_verbs_qp* gpu_qp_dev = nullptr;
      void* gpu_qp_umem = nullptr;
      void* gpu_qp_umem_cpu = nullptr;
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
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE,
                                      &send_cq_umem));
      check_doca("doca_umem_gpu_create(recv_cq)",
                 doca_umem_gpu_create(gpu,
                                      dev,
                                      recv_cq_umem_buf,
                                      kExternalQueueBytes,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE,
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
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE,
                                      &qp_wq_umem));
      check_doca("doca_umem_gpu_create(qp_dbr)",
                 doca_umem_gpu_create(gpu,
                                      dev,
                                      qp_dbr_umem_buf,
                                      kExternalDbrBytes,
                                      DOCA_ACCESS_FLAG_LOCAL_READ_WRITE,
                                      &qp_dbr_umem));
      check_doca("doca_verbs_qp_init_attr_set_external_datapath_en",
                 doca_verbs_qp_init_attr_set_external_datapath_en(qp_init, 1));
      check_doca("doca_verbs_qp_init_attr_set_external_umem",
                 doca_verbs_qp_init_attr_set_external_umem(qp_init, qp_wq_umem, 0));
      check_doca("doca_verbs_qp_init_attr_set_external_dbr_umem",
                 doca_verbs_qp_init_attr_set_external_dbr_umem(qp_init, qp_dbr_umem, 0));
      check_doca("doca_verbs_qp_create", doca_verbs_qp_create(verbs_context, qp_init, &qp));
      qp_modify_to_init(qp);

      const QPInfo local_info{context.get_lid(), doca_verbs_qp_get_qpn(qp)};
      QPInfo remote_info{};
      exchange_qp_info(context, *cm.server_qps[server], local_info, remote_info);
      qp_modify_to_rtr(verbs_context, qp, remote_info);
      qp_modify_to_rts(qp);

      send_cqs.push_back(send_cq);
      recv_cqs.push_back(recv_cq);
      qps.push_back(qp);
      check_doca("doca_gpu_mem_alloc(qp_umem)",
                 doca_gpu_mem_alloc(
                   gpu, kGpuQpUmemBytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU_CPU, &gpu_qp_umem, &gpu_qp_umem_cpu));
      (void)gpu_qp_umem_cpu;
      check_doca("doca_gpu_verbs_export_qp",
                 doca_gpu_verbs_export_qp(gpu,
                                         dev,
                                         qp,
                                         DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                                         gpu_qp_umem,
                                         send_cq,
                                         recv_cq,
                                         &gpu_qp));
      check_doca("doca_gpu_verbs_get_qp_dev", doca_gpu_verbs_get_qp_dev(gpu_qp, &gpu_qp_dev));
      uint8_t cpu_proxy_enabled = 0;
      check_doca("doca_gpu_verbs_cpu_proxy_enabled",
                 doca_gpu_verbs_cpu_proxy_enabled(gpu_qp, &cpu_proxy_enabled));
      std::cerr << "[STATUS]: GPUNetIO QP " << server << " cpu_proxy="
                << static_cast<unsigned>(cpu_proxy_enabled) << std::endl;
      gpu_qps.push_back(gpu_qp);
      gpu_qp_umems.push_back(gpu_qp_umem);
      gpu_qp_devices_host.push_back(gpu_qp_dev);
      gpu_qp_cpu_proxy_enabled.push_back(cpu_proxy_enabled);
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

    const size_t scratch_bytes =
      dim * sizeof(float) + max_results * sizeof(uint32_t) + beam_width * sizeof(uint64_t) +
      beam_width * sizeof(float) + beam_width * sizeof(uint32_t) + max_visited * sizeof(uint64_t) +
      2 * node_size + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t) + 2 * sizeof(int) +
      10 * sizeof(uint64_t) + 256;
    const size_t scratch_allocation_bytes = align_up(scratch_bytes, kGpuPageSize);

    check_doca("doca_gpu_mem_alloc",
               doca_gpu_mem_alloc(
                 gpu, scratch_allocation_bytes, kGpuPageSize, DOCA_GPU_MEM_TYPE_GPU, &scratch_base, nullptr));
    check_doca("doca_gpu_dmabuf_fd", doca_gpu_dmabuf_fd(gpu, scratch_base, scratch_allocation_bytes, &dmabuf_fd));

    ibv_pd* ibv_pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(pd);
    if (ibv_pd == nullptr) {
      throw std::runtime_error("doca_verbs_bridge_verbs_pd_get_ibv_pd returned null");
    }
    scratch_mr = mlx5dv_reg_dmabuf_mr(ibv_pd,
                                      0,
                                      scratch_allocation_bytes,
                                      0,
                                      dmabuf_fd,
                                      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE,
                                      0);
    if (scratch_mr == nullptr) {
      throw std::runtime_error(std::string("mlx5dv_reg_dmabuf_mr(scratch): ") + std::strerror(errno));
    }
    local_mkey = scratch_mr->lkey;
    local_mkey_wqe = byte_swap32(local_mkey);

    size_t offset = 0;
    auto allocate = [&](const size_t bytes, const size_t alignment) -> void* {
      offset = align_up(offset, alignment);
      auto* ptr = static_cast<unsigned char*>(scratch_base) + offset;
      offset += bytes;
      return ptr;
    };

    d_query = static_cast<float*>(allocate(dim * sizeof(float), alignof(float)));
    d_result_ids = static_cast<uint32_t*>(allocate(max_results * sizeof(uint32_t), alignof(uint32_t)));
    d_beam_ptrs = static_cast<uint64_t*>(allocate(beam_width * sizeof(uint64_t), alignof(uint64_t)));
    d_beam_dists = static_cast<float*>(allocate(beam_width * sizeof(float), alignof(float)));
    d_beam_expanded = static_cast<uint32_t*>(allocate(beam_width * sizeof(uint32_t), alignof(uint32_t)));
    d_visited_ptrs = static_cast<uint64_t*>(allocate(max_visited * sizeof(uint64_t), alignof(uint64_t)));
    d_node_a = static_cast<unsigned char*>(allocate(node_size, alignof(uint64_t)));
    d_node_b = static_cast<unsigned char*>(allocate(node_size, alignof(uint64_t)));
    d_medoid_ptr = static_cast<uint64_t*>(allocate(sizeof(uint64_t), alignof(uint64_t)));
    d_dump_ptr = static_cast<unsigned char*>(allocate(sizeof(uint64_t), alignof(uint64_t)));
    d_result_count = static_cast<uint32_t*>(allocate(sizeof(uint32_t), alignof(uint32_t)));
    d_status_code = static_cast<int*>(allocate(sizeof(int), alignof(int)));
    d_rdma_status_code = static_cast<int*>(allocate(sizeof(int), alignof(int)));
    d_debug_values = static_cast<uint64_t*>(allocate(kGpuNetioDebugValueCount * sizeof(uint64_t), alignof(uint64_t)));

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

    check_cuda("cudaStreamCreate", cudaStreamCreate(&stream));
  }

  ~Resource() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    if (d_qp_array != nullptr) {
      cudaFree(d_qp_array);
    }
    if (d_remote_regions != nullptr) {
      cudaFree(d_remote_regions);
    }
    if (scratch_mr != nullptr) {
      ibv_dereg_mr(scratch_mr);
    }
    if (dmabuf_fd >= 0) {
      close(dmabuf_fd);
    }
    if (scratch_base != nullptr && gpu != nullptr) {
      doca_gpu_mem_free(gpu, scratch_base);
    }
    for (size_t i = 0; i < gpu_qps.size(); ++i) {
      if (gpu != nullptr && gpu_qps[i] != nullptr) {
        doca_gpu_verbs_unexport_qp(gpu, gpu_qps[i]);
      }
    }
    for (auto* gpu_qp_umem : gpu_qp_umems) {
      if (gpu != nullptr && gpu_qp_umem != nullptr) {
        doca_gpu_mem_free(gpu, gpu_qp_umem);
      }
    }
    for (auto* qp : qps) {
      if (qp != nullptr) {
        doca_verbs_qp_destroy(qp);
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
    if (dev != nullptr) {
      doca_dev_close(dev);
    }
    if (gpu != nullptr) {
      doca_gpu_destroy(gpu);
    }
  }

  vec<node_t> search(const vec<element_t>& query, const u32 requested_k) {
    check_cuda("cudaMemcpyAsync(query)",
               cudaMemcpyAsync(d_query, query.data(), dim * sizeof(float), cudaMemcpyHostToDevice, stream));

    GpuNetioExactSearchParams params{
      .query = d_query,
      .dim = dim,
      .beam_width = beam_width,
      .top_k = std::min<uint32_t>(requested_k, max_results),
      .max_results = max_results,
      .max_visited = max_visited,
      .max_degree = max_degree,
      .node_size = node_size,
      .offset_id = static_cast<uint32_t>(VamanaNode::offset_id()),
      .offset_edge_count = static_cast<uint32_t>(VamanaNode::offset_edge_count()),
      .offset_vector = static_cast<uint32_t>(VamanaNode::offset_vector()),
      .offset_neighbors = static_cast<uint32_t>(VamanaNode::offset_neighbors()),
      .max_rdma_reads = max_rdma_reads_per_query(max_visited, beam_width, std::min<uint32_t>(requested_k, max_results)),
      .local_mkey = local_mkey_wqe,
      .local_iova_base = reinterpret_cast<uint64_t>(scratch_base),
      .remote_regions = d_remote_regions,
      .remote_region_count = remote_region_count,
      .qp_array = d_qp_array,
      .beam_ptrs = d_beam_ptrs,
      .beam_dists = d_beam_dists,
      .beam_expanded = d_beam_expanded,
      .visited_ptrs = d_visited_ptrs,
      .result_ids = d_result_ids,
      .result_count = d_result_count,
      .status_code = d_status_code,
      .rdma_status_code = d_rdma_status_code,
      .debug_values = d_debug_values,
      .node_a = d_node_a,
      .node_b = d_node_b,
      .medoid_ptr = d_medoid_ptr,
      .dump_ptr = d_dump_ptr,
    };

    launch_gpunetio_exact_search(stream, params);
    const bool needs_cpu_proxy =
      std::any_of(gpu_qp_cpu_proxy_enabled.begin(), gpu_qp_cpu_proxy_enabled.end(), [](uint8_t enabled) {
        return enabled != 0;
      });
    if (needs_cpu_proxy) {
      while (true) {
        const cudaError_t stream_status = cudaStreamQuery(stream);
        if (stream_status == cudaSuccess) {
          break;
        }
        if (stream_status != cudaErrorNotReady) {
          check_cuda("cudaStreamQuery", stream_status);
        }
        for (size_t i = 0; i < gpu_qps.size(); ++i) {
          if (gpu_qp_cpu_proxy_enabled[i] != 0) {
            check_doca("doca_gpu_verbs_cpu_proxy_progress", doca_gpu_verbs_cpu_proxy_progress(gpu_qps[i]));
          }
        }
        std::this_thread::yield();
      }
    }
    check_cuda("cudaStreamSynchronize", cudaStreamSynchronize(stream));

    uint32_t result_count = 0;
    int status_code = 0;
    int rdma_status_code = 0;
    uint64_t debug_values[kGpuNetioDebugValueCount] = {};
    check_cuda("cudaMemcpy(result_count)",
               cudaMemcpy(&result_count, d_result_count, sizeof(result_count), cudaMemcpyDeviceToHost));
    check_cuda("cudaMemcpy(status_code)",
               cudaMemcpy(&status_code, d_status_code, sizeof(status_code), cudaMemcpyDeviceToHost));
    check_cuda("cudaMemcpy(rdma_status_code)",
               cudaMemcpy(&rdma_status_code, d_rdma_status_code, sizeof(rdma_status_code), cudaMemcpyDeviceToHost));
    check_cuda("cudaMemcpy(debug_values)",
               cudaMemcpy(debug_values, d_debug_values, sizeof(debug_values), cudaMemcpyDeviceToHost));
    if (status_code != 0) {
      throw std::runtime_error("gpunetio_exact_gpu query kernel failed with status " + std::to_string(status_code) +
                               ", rdma_status=" + std::to_string(rdma_status_code) +
                               ", remote_addr=" + hex_u64(debug_values[0]) +
                               ", remote_rkey_wqe=" + std::to_string(debug_values[1]) +
                               ", remote_rkey_native=" + std::to_string(debug_values[5]) +
                               ", local_iova=" + hex_u64(debug_values[2]) +
                               ", local_mkey_wqe=" + std::to_string(debug_values[3]) +
                               ", local_mkey_native=" + std::to_string(local_mkey) +
                               ", medoid_raw=" + hex_u64(debug_values[4]) +
                               ", node_id=" + std::to_string(static_cast<uint32_t>(debug_values[14])) +
                               ", node_edges=" + std::to_string(static_cast<uint32_t>(debug_values[15])) +
                               ", cqe_err=" + hex_u64(debug_values[6]) +
                               ", cqe_wqe=" + hex_u64(debug_values[7]) +
                               ", poll_ci=" + hex_u64(debug_values[8]) +
                               ", poll_opown=" + hex_u64(debug_values[9]) +
                               ", rdma_stage=" + std::to_string(debug_values[10]) +
                               ", rdma_raw=" + hex_u64(debug_values[11]) +
                               ", rdma_remote=" + hex_u64(debug_values[12]) +
                               ", rdma_local_iova=" + hex_u64(debug_values[13]));
    }

    result_count = std::min<uint32_t>(result_count, std::min<uint32_t>(requested_k, max_results));
    vec<node_t> result_ids(result_count);
    if (result_count > 0) {
      check_cuda("cudaMemcpy(result_ids)",
                 cudaMemcpy(result_ids.data(), d_result_ids, result_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    return result_ids;
  }

  uint32_t dim{};
  uint32_t beam_width{};
  uint32_t max_results{};
  uint32_t max_visited{};
  uint32_t max_degree{};
  uint32_t node_size{};
  uint32_t remote_region_count{};
  doca_verbs_context* verbs_context{nullptr};
  doca_verbs_pd* pd{nullptr};
  doca_dev* dev{nullptr};
  doca_gpu* gpu{nullptr};
  vec<doca_verbs_cq*> send_cqs{};
  vec<doca_verbs_cq*> recv_cqs{};
  vec<doca_verbs_qp*> qps{};
  vec<doca_gpu_verbs_qp*> gpu_qps{};
  vec<void*> gpu_qp_umems{};
  vec<uint8_t> gpu_qp_cpu_proxy_enabled{};
  vec<doca_umem*> external_umems{};
  vec<void*> external_umem_buffers{};
  vec<void*> gpu_qp_devices_host{};
  void** d_qp_array{nullptr};
  ibv_mr* scratch_mr{nullptr};
  void* scratch_base{nullptr};
  int dmabuf_fd{-1};
  uint32_t local_mkey{};
  uint32_t local_mkey_wqe{};
  vec<GpuNetioRemoteMemoryRegion> remote_regions_host{};
  GpuNetioRemoteMemoryRegion* d_remote_regions{nullptr};
  float* d_query{nullptr};
  uint32_t* d_result_ids{nullptr};
  uint64_t* d_beam_ptrs{nullptr};
  float* d_beam_dists{nullptr};
  uint32_t* d_beam_expanded{nullptr};
  uint64_t* d_visited_ptrs{nullptr};
  unsigned char* d_node_a{nullptr};
  unsigned char* d_node_b{nullptr};
  uint64_t* d_medoid_ptr{nullptr};
  unsigned char* d_dump_ptr{nullptr};
  uint32_t* d_result_count{nullptr};
  int* d_status_code{nullptr};
  int* d_rdma_status_code{nullptr};
  uint64_t* d_debug_values{nullptr};
  cudaStream_t stream{nullptr};
};

GpuNetioQueryPool::GpuNetioQueryPool(const configuration::IndexConfiguration& config,
                                     const u32 resource_count,
                                     Context& context,
                                     ClientConnectionManager& cm,
                                     const MemoryRegionTokens& remote_regions)
    : config_(config),
      busy_(resource_count, false) {
  resources_.reserve(resource_count);
  for (uint32_t i = 0; i < resource_count; ++i) {
    resources_.push_back(std::make_unique<Resource>(config, i, context, cm, remote_regions));
  }
}

GpuNetioQueryPool::~GpuNetioQueryPool() = default;

vec<node_t> GpuNetioQueryPool::search(const vec<element_t>& query, const u32 k, service::breakdown::Sample* sample) {
  statistics::ThreadStatistics counters_before{};
  if (sample != nullptr) {
    const auto now = service::breakdown::Clock::now();
    sample->mark_started(now, now, counters_before);
  }

  size_t idx = 0;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() {
      for (size_t scan = 0; scan < busy_.size(); ++scan) {
        const size_t i = (next_resource_ + scan) % busy_.size();
        if (!busy_[i]) {
          idx = i;
          busy_[i] = true;
          next_resource_ = (i + 1) % busy_.size();
          return true;
        }
      }
      return false;
    });
  }

  try {
    vec<node_t> result = resources_[idx]->search(query, k);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      busy_[idx] = false;
    }
    cv_.notify_one();

    if (sample != nullptr) {
      sample->mark_finished(service::breakdown::Clock::now(), counters_before);
    }
    return result;
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      busy_[idx] = false;
    }
    cv_.notify_one();
    throw;
  }
}

}  // namespace gpu
