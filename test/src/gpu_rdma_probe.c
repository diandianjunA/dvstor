#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#ifndef IBV_WC_DRIVER1
#define IBV_WC_DRIVER1 135
#define IBV_WC_DRIVER2 136
#define IBV_WC_DRIVER3 137
#endif

#include <infiniband/mlx5dv.h>

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_verbs.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#define PROBE_BYTES (64 * 1024)
#define CQ_ENTRIES 128
#define QP_ENTRIES 128

struct probe_args {
  const char *gpu_pci;
  const char *nic_pci;
  const char *ibdev_name;
};

static void failf(const char *fmt, const char *detail) {
  fprintf(stderr, "probe_result=FAIL error=\"");
  fprintf(stderr, fmt, detail);
  fprintf(stderr, "\"\n");
  exit(1);
}

static void fail_errno(const char *what) {
  failf("%s: %s", strerror(errno));
}

static void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess)
    failf("%s: %s", cudaGetErrorString(status));
}

static void check_doca(doca_error_t status, const char *what) {
  if (status != DOCA_SUCCESS)
    failf("%s: %s", doca_error_get_descr(status));
}

static struct probe_args parse_args(int argc, char **argv) {
  struct probe_args args = {0};
  int i;

  for (i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--gpu-pci") == 0 && i + 1 < argc) {
      args.gpu_pci = argv[++i];
    } else if (strcmp(argv[i], "--nic-pci") == 0 && i + 1 < argc) {
      args.nic_pci = argv[++i];
    } else if (strcmp(argv[i], "--ibdev") == 0 && i + 1 < argc) {
      args.ibdev_name = argv[++i];
    } else {
      failf("unknown or incomplete argument: %s", argv[i]);
    }
  }

  if (args.gpu_pci == NULL || args.nic_pci == NULL || args.ibdev_name == NULL) {
    failf("%s", "usage: GpuRdmaProbe --gpu-pci <0000:BB:DD.F> --nic-pci <0000:BB:DD.F> --ibdev <mlx5_0>");
  }
  return args;
}

static int find_cuda_device_by_pci(const char *gpu_pci) {
  int count = 0;
  int i;
  char normalized_input[32] = {0};
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");

  {
    const char *src = gpu_pci;
    unsigned int domain = 0, bus = 0, dev = 0, func = 0;
    if (sscanf(src, "%x:%x:%x.%x", &domain, &bus, &dev, &func) == 4) {
      snprintf(normalized_input, sizeof(normalized_input), "%04x:%02x:%02x.%01x", domain & 0xffff, bus & 0xff,
               dev & 0xff, func & 0xf);
    } else {
      strncpy(normalized_input, gpu_pci, sizeof(normalized_input) - 1);
    }
  }

  for (i = 0; i < count; ++i) {
    char bus_id[32] = {0};
    check_cuda(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), i), "cudaDeviceGetPCIBusId");
    if (strcasecmp(normalized_input, bus_id) == 0)
      return i;
  }
  failf("failed to map GPU PCI address to a CUDA device: %s", normalized_input);
  return -1;
}

static struct ibv_device *find_ibv_device(const char *ibdev_name, struct ibv_device **list, int count) {
  int i;
  for (i = 0; i < count; ++i) {
    if (strcmp(ibdev_name, ibv_get_device_name(list[i])) == 0)
      return list[i];
  }
  return NULL;
}

static struct doca_devinfo *find_doca_device(const struct probe_args *args) {
  struct doca_devinfo **dev_list = NULL;
  uint32_t nb_devs = 0;
  uint32_t i;

  check_doca(doca_devinfo_create_list(&dev_list, &nb_devs), "doca_devinfo_create_list");
  for (i = 0; i < nb_devs; ++i) {
    char pci[DOCA_DEVINFO_PCI_ADDR_SIZE] = {0};
    char ibdev[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {0};
    if (doca_devinfo_get_pci_addr_str(dev_list[i], pci) != DOCA_SUCCESS)
      continue;
    if (doca_devinfo_get_ibdev_name(dev_list[i], ibdev, sizeof(ibdev)) != DOCA_SUCCESS)
      continue;
    if (strcmp(args->nic_pci, pci) == 0 || strcmp(args->ibdev_name, ibdev) == 0)
      return dev_list[i];
  }

  failf("failed to find a DOCA device matching NIC PCI or IB device: %s", args->nic_pci);
  return NULL;
}

static void run_gpudirect_rdma_probe(const struct probe_args *args) {
  struct doca_gpu *gpu = NULL;
  void *gpu_mem = NULL;
  int dmabuf_fd = -1;
  int ibv_count = 0;
  struct ibv_device **ibv_list = NULL;
  struct ibv_device *ibv_dev = NULL;
  struct ibv_context *ibv_ctx = NULL;
  struct ibv_pd *pd = NULL;
  struct ibv_mr *mr = NULL;
  int cuda_device;

  printf("[phase 1] GPUDirect RDMA DMABUF registration\n");

  cuda_device = find_cuda_device_by_pci(args->gpu_pci);
  check_cuda(cudaSetDevice(cuda_device), "cudaSetDevice");

  check_doca(doca_gpu_create(args->gpu_pci, &gpu), "doca_gpu_create");
  check_doca(doca_gpu_mem_alloc(gpu, PROBE_BYTES, 64 * 1024, DOCA_GPU_MEM_TYPE_GPU, &gpu_mem, NULL),
             "doca_gpu_mem_alloc");
  check_doca(doca_gpu_dmabuf_fd(gpu, gpu_mem, PROBE_BYTES, &dmabuf_fd), "doca_gpu_dmabuf_fd");

  ibv_list = ibv_get_device_list(&ibv_count);
  if (ibv_list == NULL)
    fail_errno("ibv_get_device_list");
  ibv_dev = find_ibv_device(args->ibdev_name, ibv_list, ibv_count);
  if (ibv_dev == NULL)
    failf("failed to find IB verbs device: %s", args->ibdev_name);

  ibv_ctx = ibv_open_device(ibv_dev);
  if (ibv_ctx == NULL)
    fail_errno("ibv_open_device");
  pd = ibv_alloc_pd(ibv_ctx);
  if (pd == NULL)
    fail_errno("ibv_alloc_pd");
  mr = mlx5dv_reg_dmabuf_mr(pd,
                            0,
                            PROBE_BYTES,
                            0,
                            dmabuf_fd,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE,
                            0);
  if (mr == NULL)
    fail_errno("ibv_reg_dmabuf_mr");

  printf("  gpu_pci=%s ibdev=%s lkey=%u rkey=%u\n", args->gpu_pci, args->ibdev_name, mr->lkey, mr->rkey);

  ibv_dereg_mr(mr);
  ibv_dealloc_pd(pd);
  ibv_close_device(ibv_ctx);
  ibv_free_device_list(ibv_list);
  close(dmabuf_fd);
  check_doca(doca_gpu_mem_free(gpu, gpu_mem), "doca_gpu_mem_free");
  check_doca(doca_gpu_destroy(gpu), "doca_gpu_destroy");
}

static void run_gpunetio_export_probe(const struct probe_args *args) {
  struct doca_devinfo *match = NULL;
  struct doca_verbs_context *verbs_context = NULL;
  struct doca_verbs_pd *pd = NULL;
  struct doca_dev *dev = NULL;
  struct doca_verbs_cq_attr *cq_attr = NULL;
  struct doca_verbs_cq *send_cq = NULL;
  struct doca_verbs_cq *recv_cq = NULL;
  struct doca_verbs_qp_init_attr *qp_attr = NULL;
  struct doca_verbs_qp *qp = NULL;
  struct doca_gpu *gpu = NULL;
  void *gpu_proxy_buf = NULL;
  struct doca_gpu_verbs_qp *gpu_qp = NULL;
  struct doca_gpu_dev_verbs_qp *gpu_qp_dev = NULL;
  uint8_t cpu_proxy = 0;

  printf("[phase 2] GPUNetIO local QP export\n");

  match = find_doca_device(args);
  check_doca(doca_verbs_context_create(match, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, &verbs_context),
             "doca_verbs_context_create");
  check_doca(doca_verbs_pd_create(verbs_context, &pd), "doca_verbs_pd_create");
  check_doca(doca_verbs_pd_as_doca_dev(pd, &dev), "doca_verbs_pd_as_doca_dev");

  check_doca(doca_verbs_cq_attr_create(&cq_attr), "doca_verbs_cq_attr_create");
  check_doca(doca_verbs_cq_attr_set_entry_size(cq_attr, DOCA_VERBS_CQ_ENTRY_SIZE_64),
             "doca_verbs_cq_attr_set_entry_size");
  check_doca(doca_verbs_cq_attr_set_cq_size(cq_attr, CQ_ENTRIES), "doca_verbs_cq_attr_set_cq_size");
  check_doca(doca_verbs_cq_create(verbs_context, cq_attr, &send_cq), "doca_verbs_cq_create(send)");
  check_doca(doca_verbs_cq_create(verbs_context, cq_attr, &recv_cq), "doca_verbs_cq_create(recv)");

  check_doca(doca_verbs_qp_init_attr_create(&qp_attr), "doca_verbs_qp_init_attr_create");
  check_doca(doca_verbs_qp_init_attr_set_pd(qp_attr, pd), "doca_verbs_qp_init_attr_set_pd");
  check_doca(doca_verbs_qp_init_attr_set_send_cq(qp_attr, send_cq), "doca_verbs_qp_init_attr_set_send_cq");
  check_doca(doca_verbs_qp_init_attr_set_receive_cq(qp_attr, recv_cq),
             "doca_verbs_qp_init_attr_set_receive_cq");
  check_doca(doca_verbs_qp_init_attr_set_sq_wr(qp_attr, QP_ENTRIES), "doca_verbs_qp_init_attr_set_sq_wr");
  check_doca(doca_verbs_qp_init_attr_set_rq_wr(qp_attr, QP_ENTRIES), "doca_verbs_qp_init_attr_set_rq_wr");
  check_doca(doca_verbs_qp_init_attr_set_send_max_sges(qp_attr, 1),
             "doca_verbs_qp_init_attr_set_send_max_sges");
  check_doca(doca_verbs_qp_init_attr_set_receive_max_sges(qp_attr, 1),
             "doca_verbs_qp_init_attr_set_receive_max_sges");
  check_doca(doca_verbs_qp_init_attr_set_max_inline_data(qp_attr, 0),
             "doca_verbs_qp_init_attr_set_max_inline_data");
  check_doca(doca_verbs_qp_init_attr_set_qp_type(qp_attr, DOCA_VERBS_QP_TYPE_RC),
             "doca_verbs_qp_init_attr_set_qp_type");
  check_doca(doca_verbs_qp_create(verbs_context, qp_attr, &qp), "doca_verbs_qp_create");

  check_doca(doca_gpu_create(args->gpu_pci, &gpu), "doca_gpu_create");
  check_doca(doca_gpu_mem_alloc(gpu, 4096, 64 * 1024, DOCA_GPU_MEM_TYPE_GPU, &gpu_proxy_buf, NULL),
             "doca_gpu_mem_alloc(proxy)");
  check_doca(doca_gpu_verbs_export_qp(gpu,
                                      dev,
                                      qp,
                                      DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO,
                                      gpu_proxy_buf,
                                      send_cq,
                                      recv_cq,
                                      &gpu_qp),
             "doca_gpu_verbs_export_qp");
  check_doca(doca_gpu_verbs_get_qp_dev(gpu_qp, &gpu_qp_dev), "doca_gpu_verbs_get_qp_dev");
  check_doca(doca_gpu_verbs_cpu_proxy_enabled(gpu_qp, &cpu_proxy), "doca_gpu_verbs_cpu_proxy_enabled");

  printf("  qpn=%u sq_cqn=%u rq_cqn=%u cpu_proxy=%u gpu_qp_dev=%p\n",
         doca_verbs_qp_get_qpn(qp),
         doca_verbs_cq_get_cqn(send_cq),
         doca_verbs_cq_get_cqn(recv_cq),
         (unsigned)cpu_proxy,
         (void *)gpu_qp_dev);

  check_doca(doca_gpu_verbs_unexport_qp(gpu, gpu_qp), "doca_gpu_verbs_unexport_qp");
  check_doca(doca_gpu_mem_free(gpu, gpu_proxy_buf), "doca_gpu_mem_free(proxy)");
  check_doca(doca_gpu_destroy(gpu), "doca_gpu_destroy");
  check_doca(doca_verbs_qp_destroy(qp), "doca_verbs_qp_destroy");
  check_doca(doca_verbs_qp_init_attr_destroy(qp_attr), "doca_verbs_qp_init_attr_destroy");
  check_doca(doca_verbs_cq_destroy(send_cq), "doca_verbs_cq_destroy(send)");
  check_doca(doca_verbs_cq_destroy(recv_cq), "doca_verbs_cq_destroy(recv)");
  check_doca(doca_verbs_cq_attr_destroy(cq_attr), "doca_verbs_cq_attr_destroy");
  check_doca(doca_dev_close(dev), "doca_dev_close");
  check_doca(doca_verbs_pd_destroy(pd), "doca_verbs_pd_destroy");
  check_doca(doca_verbs_context_destroy(verbs_context), "doca_verbs_context_destroy");
}

int main(int argc, char **argv) {
  struct probe_args args = parse_args(argc, argv);

  printf("gpu_pci=%s nic_pci=%s ibdev=%s\n", args.gpu_pci, args.nic_pci, args.ibdev_name);
  run_gpudirect_rdma_probe(&args);
  run_gpunetio_export_probe(&args);
  printf("probe_result=PASS\n");
  return 0;
}
