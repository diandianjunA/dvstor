# GPU-Centric Query Engine

## 设计边界

查询数据面只有 GPU：持久化 kernel 保存 beam、visited set、PQ lookup table、
缓存状态和 RDMA 请求状态。CPU 不执行图导航，不逐层提交 RDMA，也不计算候选
距离。存储节点只暴露注册内存并执行动态更新协议。

支持的索引契约固定为：

- schema 15；
- L2 距离；
- fixed record 为 `header + id + generation + exact vector`；
- 每个紧凑图记录不超过 512 字节，指针宽度为 5 字节；
- OPQ + 可配置数量的 8-bit PQ 子空间，默认 PQ32 即每向量 32 字节；
- 图和精确向量由存储节点持有，PQ code 常驻计算 GPU。

不满足契约的索引在建立任何查询资源前被拒绝。

## 启动

1. 计算服务读取 metadata、PQ 模型与 anchors，合成内存中的分片布局。
2. 每个存储节点加载自身 `.dat`，再把 `.pq32.codes` 放入 metadata 指定的
   注册内存区间。
3. 计算节点按远端区间直接将 PQ code 批量 RDMA 到最终 GPU 数组。
4. 对每个分片抽样首、中、尾 code，与远端权威码流比较。
5. GPUNetIO 为每个存储节点建立 GPU 可见 QP，并运行真实 RDMA read probe。
6. 只有布局、码流、QP 和显存预算全部验证成功后才启动持久化 kernel。

启动阶段的 CPU-posted GPUDirect RDMA 只用于连续码流导入，不是稳态查询回退；
payload 不经过计算节点主机内存。

## 请求调度

提交环和完成环是有界结构。查询 CTA、更新 control CTA 和 QP-owner warp 是隔离的
长期资源；查询 CTA 不消费更新命令，更新编码也不占用查询 CTA。CPU admission
thread 只做以下工作：

- 从空闲槽池领取 slot；
- 将原始 query 拷入预分配 GPU 输入区；
- 立即发布 ring descriptor，不等待人为凑批。

每个请求只分配一次 slot，不在热路径调用 `cudaMalloc`、创建 stream 或创建 QP。
持久化 block 从提交 ring 获取 slot，完成后写 completion ring；CPU completion thread
只组装最终 ID 列表并归还 slot。常驻查询 CTA 数量同时受 GPU 上限、query slot 和
服务并发线程数约束；空闲 CTA 不会按 query slot 数量过度常驻并与 QP-owner warp
争抢 SM 调度资源。

## GPU 图遍历

一个查询 block 的执行阶段如下：

1. 将 query 转为 float，应用 OPQ 矩阵；
2. 为每个 PQ 子空间构建 256 项距离表；
3. 对连续存放的 anchor PQ 路由层和最新动态入口打分；
4. 从 beam 选择未展开候选；
5. 以 `gpu-graph-prefetch-depth` 并行发出远端图读取；
6. 解码 5-byte RemotePtr，并用常驻 PQ code 评分；
7. 去重、合并并裁剪到 traversal beam；
8. 达到收敛或 `gpu-max-expansions` 后进入精排。

anchor PQ code 在启动时从完整 code 表聚集成连续的小型路由层。这只是相同 code
的无损重排，既不引入新的量化误差，也避免每次入口选择在大规模 PQ 表中随机读。

图读取 miss 直接落入每查询 scratch，只有重复访问才准入有界 GPU adjacency
cache；因此冷态正确路径和吞吐不依赖缓存命中。基础图视为不可变快照；在线 mutation
通过独立 GPU delta overlay、override epoch 和动态 anchor 桶生效，不逐写清空
基础图热点。四路组相联缓存使用 reader pin，避免替换仍被查询读取的 cache line。

## 精确重排

近似导航只决定候选覆盖，最终结果始终使用原始 L2 距离：

1. 选择最多 `gpu-final-rerank-width` 个候选；
2. 从 GPU exact cache 命中，或通过 GPUNetIO 拉取远端 fixed record；
3. 按 metadata dtype 解码 uint8、int8 或 float32；
4. 计算精确 L2，合并动态 delta，过滤 delete/旧 generation；
5. 按距离返回 top-k。

因此 PQ 误差不会直接污染最终距离，但会影响候选是否进入精排。召回率主要由
entry point、traversal beam、最大展开数和精排宽度共同决定。

## 动态一致性

更新采用 storage-owner commit + GPU epoch publish：

1. storage owner 完成 fixed record、紧凑图、idmap 和反向边更新；
2. 计算服务按可见性窗口合并 commit 结果；
3. CPU 把原始 dtype 向量和记录描述符写入 mapped pinned staging，不启动 side kernel，
   也不执行同步 H2D 拷贝；
4. 专用常驻 control CTA 批量完成 OPQ/PQ 编码，并更新 delta 哈希、bucket 和 override；
5. 原子发布 delta count 和 snapshot epoch。

查询在 admission 时绑定 snapshot epoch。upsert/delete 的 base 版本由 override
epoch 屏蔽，动态版本只在对应 epoch 可见。发布、压缩或 kernel 失败会把引擎
标记为 unhealthy；后续查询立即失败，而不是继续使用可能陈旧的状态。

动态层不是单调增长日志。storage maintenance 完成新节点 stitch 和旧节点反向边
清理后推进 durable sequence；计算节点再等待可见性窗口和所有旧 query ticket
退出，随后退休 GPU L0 记录，并把该 sequence 写入自己的远端 ACK 槽。存储节点
只在所有计算节点 ACK 的最小值覆盖删除记录后复用动态物理槽。因此 upsert/delete
产生的死记录有界回收，旧查询不会读到已复用地址，多计算节点也不会互相越过 RCU
屏障。前台持续写入时至少保留一个维护 worker，避免 durable watermark 被永久饿死。

这里的“退休”不是伪装成静态索引重写：静态 PQ/图 generation 在线保持不变，
已 stitch 的存活动态节点继续由存储节点持有并按需 RDMA 访问。净新增数据必然占用
新增存储容量；只有更新和删除产生的死槽可以复用。若数据规模超过预留容量，需要
构建下一静态 generation 并切换，而不能覆盖仍存活的远端记录。

## 内存预算

显式预算由 `gpu-memory-limit-gb - gpu-memory-reserve-gb` 决定，启动前统一核算：

- `N * M` 字节的常驻 PQ code，默认 `M=32`；
- adjacency cache payload、tag、state、reader pin 和 victim；
- exact cache payload 与并发控制；
- 原始 dtype delta vector、delta PQ code、hash table 和 bucket；
- query、OPQ 输出、LUT、beam、visited set、anchors 和结果；
- DOCA/CUDA 外部状态的固定安全余量。

默认 SIFT1B：

| 项目 | 上限 |
| --- | ---: |
| PQ32 base codes | 32,000,000,000 B |
| adjacency cache | 官方冷态基线为 0，可选缓存仅接纳重复访问 |
| exact cache | 默认关闭 |
| mutable L0 | 默认 256 MiB |
| 所有显式分配 | 36 GiB |
| CUDA/DOCA reserve | 4 GiB |

若任一数组溢出、cache set 少于并发 slot、delta 容量为零或总预算超限，启动直接
失败。计算节点主机内存不保存全量 code，磁盘也不保存远端图副本。

## 性能原则

- 连续 PQ code 只在启动时传输一次；
- 稳态只按需读取 512 B 图记录和最终精确向量；
- 每个 storage node 使用多条长期存活的 GPU QP；
- 查询状态、LUT、beam 和缓存均预分配；
- 多查询并发隐藏远端延迟，而不是同步执行单查询 RDMA 往返；
- exact cache 与 graph cache 分离，避免大向量挤占导航工作集；
- telemetry 分别记录图读、精确读、cache hit、GPU phase cycle、direct-path failure、
  mutation publication queue/prepare/command、可见性延迟、L0 退休量和存储回收 ACK。

要判断是否达到目标吞吐，必须同时检查 QPS、recall、GPU utilization、QP 错误、
direct-path failure、每查询图读取数和精确读取数。单独看到 GPUNetIO probe 成功
并不代表查询执行已经高效。

## 故障策略

以下情况均为 fail-stop：

- GPUNetIO QP probe 或稳态 read 失败；
- 持久化 kernel 返回非零状态；
- delta publish、cache invalidation 或 consolidation 失败；
- ring/slot 状态不一致；
- 索引 checksum、ordinal 或远端 offset 校验失败。

系统没有 CPU 查询 fallback。这样 benchmark 不会在硬件路径失效后悄悄测量
另一套慢路径，也不会以低召回结果继续运行。
