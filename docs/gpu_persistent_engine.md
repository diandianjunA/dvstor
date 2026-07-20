# GPU-Centric Query Engine

## 设计边界

查询数据面只有 GPU：持久化 kernel 保存 beam、visited set、PQ lookup table、
路由记录状态和 RDMA 请求状态。CPU 不执行图导航，不逐层提交 RDMA，也不计算候选
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
3. 对连续存放的静态启动入口和固定容量在线动态入口共同打分；
4. 从 beam 选择未展开候选；
5. 以 `gpu-graph-prefetch-depth` 并行发出远端图读取；
6. 解码 5-byte RemotePtr，并用常驻 PQ code 评分；
7. 去重、合并并裁剪到 traversal beam；
8. 达到收敛或 `gpu-max-expansions` 后进入精排。

静态入口 PQ code 在启动时从完整 code 表聚集成连续的小型兜底层。这只是相同 code
的无损重排，既不引入新的量化误差，也避免每次入口选择在大规模 PQ 表中随机读。
每个分片另有 8 个 storage-owner 权威固定槽位。owner 用在线 mutation 持续更新
代表节点，并在既有 4 KiB control page 的保留区发布带 checksum/seqlock 的快照；
所有计算节点低频拉取同一快照，再由 control CTA 按 epoch 更新 GPU 槽位。每槽同时
携带最多 32B 的权威 PQ code，所以其他计算节点写入的动态节点也能直接参与入口
评分，不需要逐 mutation 广播或每查询额外 RDMA。动态表内存固定，不会随运行时间
或 mutation 数量增长。

离线 anchors 对应的固定数量路由图记录在启动时批量放入 GPU 常驻路由区，供初始
扩展使用；在线 mutation 修改这些记录时，刷新流程等待活跃 reader 退出后再发布
新版本。除此之外，每次图扩展都通过 GPUNetIO/RDMA 将存储节点上的紧凑图记录
直接读入每查询 scratch，不在 GPU 上保留普通图记录供后续查询复用。基础图视为
不可变快照；在线 mutation 通过独立 GPU delta overlay、override epoch 和动态
候选桶生效。

## 精确重排

近似导航只决定候选覆盖，最终结果始终使用原始 L2 距离：

1. 选择最多 `gpu-final-rerank-width` 个候选；
2. 通过 GPUNetIO/RDMA 从存储节点直接拉取远端 fixed record；
3. 按 metadata dtype 解码 uint8、int8 或 float32；
4. 计算精确 L2，合并动态 delta，过滤 delete/旧 generation；
5. 按距离返回 top-k。

因此 PQ 误差不会直接污染最终距离，但会影响候选是否进入精排。召回率主要由
entry point、traversal beam、最大展开数和精排宽度共同决定。

## 动态一致性

更新采用 storage-owner commit + GPU epoch publish：

1. stage1 在 storage owner 发布 fixed record、临时出边、idmap 和 maintenance intent，
   但不等待 GPU 发布，也不写权威反向边；
2. owner-memory ACK 返回后，计算服务按可见性窗口合并 commit 结果；
3. CPU 把原始 dtype 向量和记录描述符写入 mapped pinned staging，不启动 side kernel，
   也不执行同步 H2D 拷贝；
4. 专用常驻 control CTA 批量完成 OPQ/PQ 编码，并更新 delta 哈希、bucket 和 override；
5. 原子发布 delta count 和 snapshot epoch；
6. stage2 完成所有分片的在线构建搜索、统一 RobustPrune 和所选邻居的权威反向边，
   再写最终出边并推进 durable sequence。

查询在 admission 时绑定 snapshot epoch。upsert/delete 的 base 版本由 override
epoch 屏蔽，动态版本只在对应 epoch 可见。发布、压缩或 kernel 失败会把引擎
标记为 unhealthy；后续查询立即失败，而不是继续使用可能陈旧的状态。

本 CN 的 mutation delta 仍在 stage1 ACK 后立即异步发布，保证短期可见性；查询
路由则只接受 storage owner 的 canonical 8-slot 快照，避免多计算节点分别观察局部
写流后产生不同槽位。route-only command 先写该槽的 PQ code，再用 device-scope
seqlock 发布 `{epoch, pointer, id, generation}`，最后推进查询 snapshot epoch；若
正好与写入冲突，查询跳过该槽并继续使用静态启动入口。storage control snapshot
若被 RDMA 读到撕裂，checksum 校验失败后保留旧 GPU 快照并在下一周期重试。

动态 GPU 元数据不是单调增长日志。storage maintenance 完成任务已声明的 stage2
操作后推进 durable sequence；计算节点再等待可见性窗口和所有旧 query ticket
退出，随后退休 GPU L0 记录，并把该 sequence 写入自己的远端 ACK 槽。这个 RCU
流程只保证 GPU 可变元数据和被淘汰的常驻 PQ 槽可安全回收，不表示存储端物理节点
地址可复用。前台持续写入时至少保留一个 maintenance worker，避免 durable
watermark 被永久饿死。

这里的“退休”不是伪装成静态索引重写：静态 PQ/图 generation 在线保持不变，
已 stitch 的存活动态节点继续由存储节点持有并按需 RDMA 访问。schema-15 反向边
请求只有物理指针、没有 generation；为使迟到重试不可能命中另一个节点，动态物理
地址不会复用。净新增和每次 upsert 的新版本会分配新地址，delete 也不会释放旧
地址；它们都会持续占用预留的存储节点容量。若容量不足，需要协议升级或构建下一
静态 generation 并切换。

当前实现也没有完整入边索引。insert 的 stage2 会等待它最终选择的全部反向边操作
完成，但 delete/upsert 只能处理协议已知的边，不能证明所有历史未知入边已经从全图
移除；查询会按 tombstone/generation 过滤失效版本，残留指针仍可能增加遍历开销和
影响长期质量。因而“两阶段最终等价”仅指同一逻辑快照下：每个分片执行相同宽度
`L` 的在线构建搜索、合并所有 beam，并执行一次相同 RobustPrune 的分片 reference；
它不等价于离线 builder 的全候选构图，也不覆盖 delete/upsert 的全图整理语义。

## 内存预算

显式预算由 `gpu-memory-limit-gb - gpu-memory-reserve-gb` 决定，启动前统一核算：

- `N * M` 字节的常驻 PQ code，默认 `M=32`；
- anchors 对应的固定数量常驻路由图记录及其状态；
- 原始 dtype delta vector、delta PQ code、hash table 和 bucket；
- query、OPQ 输出、LUT、beam、visited set、静态/动态路由入口和结果；
- DOCA/CUDA 外部状态的固定安全余量。

默认 SIFT1B：

| 项目 | 上限 |
| --- | ---: |
| PQ32 base codes | 32,000,000,000 B |
| anchor 路由图记录 | 由 anchors 数量和单条图记录大小确定，固定常驻 |
| mutable L0 | 默认 256 MiB |
| 所有显式分配 | 36 GiB |
| CUDA/DOCA reserve | 4 GiB |

若任一数组溢出、delta 容量为零或总预算超限，启动直接
失败。计算节点主机内存不保存全量 code，磁盘也不保存远端图副本。

## 性能原则

- 连续 PQ code 只在启动时传输一次；
- 稳态只按需读取 512 B 图记录和最终精确向量；
- 每个 storage node 使用多条长期存活的 GPU QP；
- 查询状态、LUT、beam 和 anchor 路由图记录均预分配；
- 多查询并发隐藏远端延迟，而不是同步执行单查询 RDMA 往返；
- telemetry 分别记录图读、精确读、GPU phase cycle、direct-path failure、
  mutation publication queue/prepare/command、可见性延迟、L0 回收批次、退休量和存储回收 ACK。

要判断是否达到目标吞吐，必须同时检查 QPS、recall、GPU utilization、QP 错误、
direct-path failure、每查询图读取数和精确读取数。单独看到 GPUNetIO probe 成功
并不代表查询执行已经高效。

## 故障策略

以下情况均为 fail-stop：

- GPUNetIO QP probe 或稳态 read 失败；
- 持久化 kernel 返回非零状态；
- delta publish、anchor 路由记录刷新或安全退休失败；
- ring/slot 状态不一致；
- 索引 checksum、ordinal 或远端 offset 校验失败。

系统没有 CPU 查询 fallback。这样 benchmark 不会在硬件路径失效后悄悄测量
另一套慢路径，也不会以低召回结果继续运行。
