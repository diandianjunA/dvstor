# Live-Extent RDMA

## 目标

Live-Extent RDMA 解耦远端图记录的两个尺度：

```text
存储分配：固定大小，保留原地更新余量
查询传输：一次 one-sided RDMA READ，只传当前邻接长度档覆盖的前缀
```

它不改变图记录格式、搜索父节点、Beam/visited、PQ、精排或存储 CPU 查询路径。
同一批次仍产生与 fixed 模式相同数量的 graph READ WQE、descriptor、doorbell 和
最终 CQE；唯一变化是每条 READ WQE 可以具有不同的 payload length。

## 索引契约

离线工具为每个 immutable base-node physical ordinal 写一个 u8 class：

```text
class = ceil((stable_count + provisional_count) / 8)
bytes = min(graph_entry_bytes, 16 + class * 8 * sizeof(RemotePtr))
```

输出为 `<prefix>.gextent8`：

- 128 B header；
- `num_nodes` 个 u8 class，按 metadata shard 顺序和 shard-local slot 顺序排列；
- header 绑定 format version、build fingerprint、nodes、shards、record
  bytes/capacity 和 pointer width；
- header 与 payload 分别校验 checksum；
- 生成器全量校验每个 schema-16 graph record、active pointer 和零 suffix；
- 同目录临时文件完成后原子 rename，已有文件必须显式 `--overwrite`。

生成命令必须在持有 metadata 和全部 `.dat` shard 的机器执行：

```bash
cmake --build build --target vamana_graph_extent_indexer -j
./build/vamana_graph_extent_indexer \
  --index-prefix /path/to/index/prefix
```

分布式部署时，只需把生成的 `.gextent8` 复制到计算节点相同的 index prefix。
SIFT100M 的 payload 为 100,000,000 B；它不包含任何边、handle 或向量。

## GPU 数据路径

`fixed` 是默认策略：

```text
--gpu-query-graph-read-policy=fixed
```

启用：

```text
--gpu-query-graph-read-policy=live-extent
```

启动时 live-extent 模式严格加载并校验 sidecar，分配：

- `num_nodes * sizeof(u8)` 的只读 device class table；
- `query_slots * kPersistentMaxPrefetch * sizeof(u32)` 的 request-length
  scratch。

fixed 模式不加载 sidecar，也不分配这两个数组。

每个 graph batch 的执行顺序为：

1. 解析 handle。static base node 通过确定性 physical ordinal 读取 class；
   dynamic node 使用完整记录长度。
2. 在原有固定大小 graph scratch 中，只清零短读请求的未读 suffix。
3. 按 shard 形成原有 descriptor，同时传入可选的 per-request length 数组。
4. QP owner 为每条 WQE 使用自己的长度；聚合、SQ credit、doorbell、最终 signaled
   WQE 和 CQ ownership 均保持不变。
5. completion 后先仅使用已到达 header 计算 exact required prefix。
6. 若 required prefix 超出本次 transfer，或补零后的完整结构/checksum 无效，
   下一 snapshot attempt 将该请求升级为 full read。
7. full record 仍无效时沿用原有有界重读/fail-stop；动态 slot incarnation 已变化
   时沿用原有 read-committed stale discard。

因此 extent class 只是性能 hint，不是正确性 authority。并发更新使 class 过期只会
产生显式 full-read fallback，不会截断邻接表或绕过 checksum/incarnation/tombstone
语义。

## Telemetry

query completion 和 benchmark JSON 新增：

```text
graph_read_bytes
graph_live_extent_reads
graph_full_record_reads
graph_extent_fallback_reads
```

其中 `graph_read_bytes` 是成功同步执行或成功进入 owner queue 的实际 graph payload
总和，包含 snapshot retry 和 fallback；不能用逻辑 `remote_pages * record_bytes`
替代。`short + full` 是物理 graph READ WQE 数，`remote_pages` 仍是逻辑正式父节点
读取数。

RDMA trace schema 3 的 shard-batch event 记录：

```text
parent_count
payload_bytes
minimum_bytes_per_parent
maximum_bytes_per_parent
```

completion 粒度仍是 owner submission group 的最终 CQE 边界，不伪造 per-WQE
completion timestamp。

## A/B

固定和 live-extent 配置只改变 graph read policy：

```bash
./motivation/run_live_extent_ab.sh
```

默认扫描 concurrency `1 8 64 256`，每点 3 次，并在相邻重复中交换策略顺序。
如每次 service run 前需要重新启动远端 storage session，可设置：

```bash
LIVE_EXTENT_BEFORE_CASE_HOOK=/path/to/restart-storage-hook \
  ./motivation/run_live_extent_ab.sh
```

报告必须同时比较 Recall、top-k、logical graph reads、physical graph WQE、
actual graph/RDMA bytes、fallback、QPS、mean/P95/P99 和 RDMA wait。字节减少但
QPS 不提升也应如实报告。

## 已有验证与尚需硬件验证

CPU/C++ 测试覆盖 sidecar checksum/绑定/overwrite、global ordinal/class、
header-only 与 R128 class、短读补零后的 full checksum、under-sized stale class
检测和 benchmark telemetry 派生。

生产 CUDA kernel 已用 ptxas 构建：query entry kernel 158 registers/thread、
owner kernel 130 registers/thread，二者 entry kernel 均无 spill；实现没有新增
shared per-candidate array。

真实 mixed-length GPUNetIO 内容等价性、Recall/QPS/P99 和更新下 fallback rate
必须在 `.gextent8` 部署到计算节点并连接真实 storage nodes 后完成，不能由
transport microbenchmark 或 CPU reconstruction test 代替。
