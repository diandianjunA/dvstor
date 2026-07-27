# GPU-Centric Query Engine

## 设计边界

查询数据面只有 GPU：持久化 kernel 保存 beam、visited set、PQ lookup table、
路由记录状态和 RDMA 请求状态。CPU 不执行图导航，不逐层提交 RDMA，也不计算候选
距离。存储节点只暴露注册内存并执行动态更新协议。

支持的索引契约固定为：

- schema 16；
- L2 距离；
- fixed record 带 generation、slot incarnation 和原始 dtype 精确向量；
- 每个紧凑图记录不超过 2048 字节，使用 8 字节 incarnation-tagged RemotePtr；
- OPQ + 可配置数量的 8-bit PQ 子空间，默认 PQ32 即每向量 32 字节；
- 图和精确向量由存储节点持有，PQ code 常驻计算 GPU。

不满足契约的索引在建立任何查询资源前被拒绝。

## 启动

1. 计算服务读取 metadata 与 PQ 模型，仅从不可变尺寸、offset 和格式字段合成并校验
   内存中的分片布局；medoid 与离线采样 entry-point 不属于在线查询契约。
   当 `gpu-query-graph-read-policy=live-extent` 时，同时加载
   `<prefix>.gextent8`，并校验 build fingerprint、节点/分片数、记录容量、指针宽度
   及 header/payload checksum；任何不匹配都在创建查询资源前失败。
2. 每个存储节点加载自身 `.dat` 和 `.centroid`。后者包含该物理分片的补偿式 FP64
   sum/count 与 1--4 个真实存活入口，用于恢复 `CentroidRouter` 的首个版本。
3. 存储节点把 `.pq32.codes` 放入 metadata 指定的注册内存区间，并在 control block
   中公布 variable-length centroid publication 的 descriptor。码流 header 的 dtype、
   build fingerprint 与 shard fingerprint 必须同时匹配 metadata 和对应 `.dat` 分片。
4. 计算节点按远端区间直接将 PQ code 批量 RDMA 到最终 GPU 数组。
5. 对每个分片抽样首、中、尾 code，与远端权威码流比较。
6. GPUNetIO 为每个存储节点建立 GPU 可见 QP，并运行真实 RDMA read probe。
7. 持久化 kernel 启动后，计算节点必须先读取、校验并安装所有物理分片的首个完整
   centroid publication；只有布局、码流、QP、路由状态和显存预算全部就绪后才开放
   查询准入。

启动阶段的 CPU-posted GPUDirect RDMA 只用于连续码流导入，不是稳态查询回退；
payload 不经过计算节点主机内存。

## 请求调度

提交环和完成环是有界结构。查询 CTA、路由 control CTA 和 QP-owner warp 是隔离的
长期资源；查询 CTA 不消费路由命令。CPU admission
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
3. 在同一完整 publication 中读取每个物理分片的 centroid 和实时入口，按 centroid
   距离选择最近的一个物理分片并对其入口打分；跨分片遍历只由真实图边引入；
4. 从 beam 选择未展开候选；
5. 以 `gpu-graph-prefetch-depth` 并行发出远端图读取。默认 `fixed` 为每条 WQE
   使用完整物理记录长度；`live-extent` 在不增加 WQE/descriptor/CQE 的前提下，
   为同一 descriptor 中每条静态记录选择独立的 8-edge 分档长度；
6. 解码 8-byte incarnation-tagged RemotePtr，并用对应 PQ code 评分；
7. 去重、合并并裁剪到 traversal beam；
8. 达到收敛或 `gpu-max-expansions` 后进入精排。

在线查询完全不读取 metadata medoid 或离线静态 entry points。查询准入依赖完整、
可路由的 storage-canonical centroid 状态；若一次查询在两次有界快照尝试后仍得不到
任何有效实时入口，则返回错误，不从不可变入口表恢复。

每个物理分片的 `CentroidRouter` 用补偿式 FP64 sum/count 维护稳定中心，入口集合由调用方
从真实存活图节点中选出，容量固定为 1--4。mutation 只修改权威状态和版本；维护
边界显式调用 `publish()`，使 CPU reader 原子观察上一个或下一个完整 immutable
snapshot。存储节点随后发布
`{shard version, vector count, canonical FP32 centroid, live entries}`。FP64 sum/count
只用于稳定地维护中心；发布时量化一次，CPU 插入选 home 与 GPU 查询都按相同的
FP32 逐维 FMA 和物理分片 tie-break 计算，避免两条路径因精度不同选择不同分片。
publication 是
variable-length 布局，由 control block descriptor 描述，并使用 sequence bracket、
checksum、magic/version 和 shard identity 防止 RDMA 撕裂或混合部署。

计算节点低频拉取所有分片的 publication。一次同步将相同内容装入 CPU physical-home
selector，并由常驻 control CTA 在一个命令事务内更新 GPU 的 centroid、entry
和 per-shard seqlock；全表 publication epoch 还保证一次排名不会混合不同分片的
发布代际。若读取恰逢写窗，
维护线程保留上一份完整状态并重试，查询线程不会等待；若启动时始终没有完整状态，
引擎直接失败。

路由状态只常驻 centroid、版本和少量 live pointer/generation；实时入口若指向基础
节点，其 PQ code 从常驻 base PQ 表解析；若指向动态节点，权威 PQ code 随存储记录通过 GPUNetIO/RDMA
读入每查询 scratch。在线 mutation 直接修改存储侧权威图，不存在独立 GPU 更新
overlay 或需要广播到所有计算节点的动态索引副本。

## 精确重排

近似导航只决定候选覆盖，最终结果始终使用原始 L2 距离：

1. 选择最多 `gpu-final-rerank-width` 个候选；
2. 通过 GPUNetIO/RDMA 从存储节点直接拉取远端 fixed record；
3. 按 metadata dtype 解码 uint8、int8 或 float32；
4. 计算精确 L2，并按权威记录中的 tombstone/generation 过滤失效版本；
5. 按距离返回 top-k。

因此 PQ 误差不会直接污染最终距离，但会影响候选是否进入精排。召回率主要由
centroid publication 中的实时入口、traversal beam、最大展开数和精排宽度共同决定。

## 动态一致性

更新采用 storage-owner 权威图提交：

1. 所有计算节点按 `ID % N` 选择同一个逻辑 authority；计算节点把输入先编码成索引
   的 canonical dtype，再用已完整安装的 storage-canonical centroid 快照选择物理
   Stage1 home；authority 按 ID/代际串行化并验证该选择，随后在本地执行或通过 peer
   RPC 转发；
2. 物理 home 发布 fixed record、临时出边和本地反向边，authority 提交 idmap 与
   maintenance intent；Stage1 ACK 后节点已能被普通查询发现；
3. owner-memory ACK 返回后，计算服务只提交响应，不缓存全量或增量 ID 目录，也不
   复制 mutation 向量、PQ code 或图状态；
4. 查询直接读取存储侧权威动态图，动态 PQ code 和精确向量均保持原始 dtype 语义；
5. 存储节点以完整 publication 发布 centroid 路由；常驻 control CTA 仅原子安装
   该路由事务；
6. durable watermark 决定旧存储记录何时可复用；复用递增 slot incarnation，且新记录
   在正文完成后最后发布 header，使持有旧 tagged pointer 的查询拒绝新对象；
7. Stage2 在 Stage1 home 延续已保存的 beam/visited/frontier，沿跨分片图边执行
   one-sided RDMA，不进行逐分片重启搜索；收敛后统一 RobustPrune，选择跨分片边
   最少的物理 placement，协调最终反向边，再更新新旧 shard 的 centroid membership
   并推进 durable sequence。

Stage1 的权威本地反向边提供短期可见性；查询和 Stage1 home selection 接受
storage node 的 canonical centroid publication，避免多个计算节点分别从局部写流
推导出不同路由。centroid route command 在一个短暂的 device-scope seqlock 写窗中
同时发布 `{version, count, centroid, entries}`；
查询遇到写窗会有界重读，不会退回离线静态入口。storage publication 若被 RDMA
读到撕裂或 checksum 不匹配，维护线程保留上一份完整 GPU transaction，并在下一
周期重试。

storage maintenance 完成任务已声明的 Stage2 操作后推进 durable sequence；达到该
sequence 的旧记录即可进入地址复用队列。系统明确采用 incarnation-tagged
read-committed 查询语义，而不是跨计算节点 snapshot-RCU：同一次查询在不同节点展开
时可以观察到不同的已提交版本，但每次动态记录解引用都会核对 `RemotePtr` tag、记录
header 和 slot incarnation。复用时先递增 incarnation，填充新记录正文，再以 release
语义最后发布 header，因此旧指针只会被判 stale，不会把新对象误认为旧对象。控制块
不包含计算客户端数量或 ACK 槽，计算节点数量不受持久化格式中的固定数组限制。前台
持续写入时至少保留一个 maintenance worker，避免 durable watermark 被永久饿死。

GPU 图读取也遵守同一规则：checksum 完整但 incarnation 与请求 handle 不同的记录
只丢弃当前 stale 展开，不关闭 direct path；传输失败或在有界重读后仍不满足结构/
checksum 契约的记录才触发 fail-stop。这样槽复用是正常并发现象，而不是伪装成硬件
故障。

Live-Extent 把存储分配大小和网络传输大小解耦，但不改变存储格式。静态记录发起
one-sided READ 前，从 1 byte/base-node 的只读 sidecar 取得能够覆盖构建时 live
邻接前缀的长度档；目标 scratch 仍是完整物理记录大小，并在 READ 前协作清零未读
suffix，因此继续使用原有的 full-record checksum。返回 header 声明的 live prefix
超出已读范围，或重构后的结构/checksum 无效时，该请求在下一 snapshot attempt
升级为 full read。动态记录没有静态 ordinal，始终 full read。由此，过期 hint
最多增加一次有显式统计的 fallback，不能造成邻接截断。

这里的“退休”不是伪装成静态索引重写：静态 PQ/图 generation 在线保持不变，
已 Stage2 完成的存活动态节点继续由存储节点持有并按需 RDMA 访问。schema-16 的
8-byte `RemotePtr` 同时携带 shard、16-byte aligned offset 和 24-bit incarnation；
所有读取、反向边修改和回收都验证记录中的同一 incarnation。旧地址只有在 durable
watermark 后才可复用，复用时递增 incarnation；计数耗尽的
槽位永久退休而不回绕。跨分片迁移的目标分配由无超时驱逐的 bounded receipt 保护，
源记录和精确目标身份都进入终态后才能结算，因此任意迟到重试不会分配第二个目标或
命中新对象。

当前实现也没有完整入边索引。insert 的 stage2 会等待它最终选择的全部反向边操作
完成，但 delete/upsert 只能处理协议已知的边，不能证明所有历史未知入边已经从全图
移除；查询会按 tombstone/generation 过滤失效版本，残留指针仍可能增加遍历开销和
影响长期质量。因而“两阶段最终等价”仅指同一逻辑快照下：Stage2 从 Stage1 的
宽度 `L` 搜索状态继续沿图扩展并执行一次相同 RobustPrune；它不等价于离线 builder
的全候选构图，也不覆盖 delete/upsert 的全图整理语义。

文中的 durable sequence 是进程内维护完成边界，不是崩溃一致的 WAL。当前启动会
拒绝带有未完成内存态 maintenance 的恢复；进程/主机故障恢复需要上层重新部署一致
快照，不属于本实现的在线更新语义。

## 内存预算

显式预算由 `gpu-memory-limit-gb - gpu-memory-reserve-gb` 决定，启动前统一核算：

- `N * M` 字节的常驻 PQ code，默认 `M=32`；
- 每物理分片一个 centroid、版本和最多 4 个 live entry 的固定上界路由状态；
- query、OPQ 输出、LUT、beam、visited set、centroid 路由入口和结果；
- 可选 Live-Extent 的 `N` byte class table，以及每 query slot
  `kPersistentMaxPrefetch * sizeof(u32)` 的 request-length scratch；
- DOCA/CUDA 外部状态的固定安全余量。

默认 SIFT1B：

| 项目 | 上限 |
| --- | ---: |
| PQ32 base codes | 32,000,000,000 B |
| centroid route | `shards * (dim + 4 entries)` 的固定上界 |
| 所有显式分配 | 36 GiB |
| CUDA/DOCA reserve | 4 GiB |

若任一数组溢出或总预算超限，启动直接
失败。计算节点主机内存不保存全量 code，磁盘也不保存远端图副本。

## 性能原则

- 连续 PQ code 只在启动时传输一次；
- 稳态只按需读取由 `R` 决定且不超过 2048 B 的紧凑图记录和最终精确向量；
- Live-Extent 保留固定远端记录的更新余量，但只把当前长度档覆盖的邻接前缀搬过网络；
- 每个 storage node 使用多条长期存活的 GPU QP；
- 查询状态、LUT、beam 和 centroid route 状态均预分配；
- 多查询并发隐藏远端延迟，而不是同步执行单查询 RDMA 往返；
- telemetry 分别记录图读、精确读、GPU phase cycle、direct-path failure 和
  centroid route probe/body/publication；图读额外报告真实 payload bytes、
  short/full WQE 及 stale-hint fallback，不能再用物理记录大小估算流量。

要判断是否达到目标吞吐，必须同时检查 QPS、recall、GPU utilization、QP 错误、
direct-path failure、每查询图读取数和精确读取数。单独看到 GPUNetIO probe 成功
并不代表查询执行已经高效。

## 故障策略

以下情况均为 fail-stop：

- GPUNetIO QP probe 或稳态 read 失败；
- 持久化 kernel 返回非零状态；
- centroid publication 同步或存储记录安全退休失败；
- ring/slot 状态不一致；
- 索引 checksum、ordinal 或远端 offset 校验失败。

系统没有 CPU 查询 fallback。这样 benchmark 不会在硬件路径失效后悄悄测量
另一套慢路径，也不会以低召回结果继续运行。
