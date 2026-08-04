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
4. 从已完整发布的 authoritative Beam 冻结精确 commit frontier，再等待保证这些
   节点推进的 core wave；tail wave 只做非阻塞完成轮询。冻结点是本轮逻辑 commit
   决策，`gpu-graph-commit-width` 只决定本轮允许产生搜索副作用的节点数；shadow
   tail 永远不能进入这一步。
5. commit 节点若命中已验证 ROB payload 则直接解码，否则进入原有
   Live-Extent/full-snapshot retry。随后才产生邻居、更新 visited、计算 PQ
   distance，并在本轮权威发布时设置 expanded；RDMA 到达本身不能触发这些状态
   变化。ordered completion pipeline 先无阻塞收割当前可见的 CQ backlog；若下一
   CQ 尚未 READY，则立即按精确父节点顺序执行已缓存前缀，绝不拿着可做工作进入
   网络轮询。parent completion 与 candidate scoring 使用独立进度游标：decode 和
   visited 按父节点及时追加，不足一个 Stable-Run 叶（128 threads × 4 items =
   512 candidates）的候选保留到下一 CQ 小组，PQ 只在完整叶（最终 residue 除外）
   上启动。512 来自后续 COSSF 可立即消费的最小结构单元，不是数据集参数；这样小
   shard completion 仍能用 decode/visited 覆盖 RDMA interval，也不会为了一个尚
   不能 merge 的窄前缀额外进入一次完整 PQ batch。若 ordered progress 已完成本轮
   全部父节点，common final 直接复用三条进度游标，只清理 graph-slot/ROB ownership；
   visited suffix 或 PQ residue 为空时不再执行对应的空 CTA barrier 框架。
6. Stable-Run 保持原排序与 tie 语义，但把通信、候选评分、稳定合并和 Beam 发布拆成
   可重叠阶段。COSSF（Commit-Ordered Streaming Stable Fold）在 commit frontier
   冻结后把 old Beam 复制到既有 `CandidateWorkspace` 的私有 ping-pong
   accumulator，并只在私有副本上覆盖本轮 selected ranks 的 expanded 位。ordered
   CQ progress 按原 parent/neighbor 顺序产生不可再变化的连续候选区间；当后续
   critical CQ 仍在 flight 时，已封闭区间立即执行必要的 stable radix sort，并按
   原始流顺序 left-fold 到私有 top-K accumulator：
   `topK(topK(A) ⊕ B) = topK(A ⊕ B)`。CUB 叶内稳定排序和 co-rank 的左输入
   tie 优先共同保留 `old Beam < run0 < run1 ...` 的精确总顺序，分段边界可以来自
   任意运行时 completion grouping，不依赖数据集。
   最后一个区间完成后，从精确 accumulator 压缩出 unexpanded Issue Frontier，先
   发布下一 critical RDMA wave，再用一次 coalesced copy 发布完整 authoritative
   Beam。此前 Beam、termination 和 authoritative expanded 均不可见；任意 critical
   失败直接丢弃私有 fold。没有新增 per-query global buffer、全局队列或 CPU
   scheduler。高图度导致一次 commit 无法放入 ordered scoring workspace，或
   issue/commit 未解耦时，仍走 SRFC（Sort-Once Reusable Frontier Certificate）
   的同一 Stable-Run 精确前缀/后缀路径，而不是改变搜索语义。
7. `gpu-graph-issue-width` 是运行时上限。CTA-local controller 使用真实
   promotion/retention、stale、owner-ring rejection 和下一 commit prefix 的
   exact coverage，在
   `[commit-width, issue-cap]` 内调整；它不改变 Beam 宽度、commit 宽度或终止
   条件。controller 在 persistent query CTA 内跨查询保留学习状态，按已验证
   shadow 的净收益做有界乘法增长；stale 和 SQ rejection 分别计算比例收缩并只
   应用其中最大值。宽度收缩到 commit width 后不再让每个短查询重复 bootstrap，
   而是每 `commit-width` 个有效查询开放一个 shadow slot 重新采样，以运行时参数
   决定恢复频率而不引入数据集阈值。
   未完成 tail 既不能 refill，也不能取得 exact coverage，因此 CQ backlog 会自然
   阻止增长；controller 不再把一次 nonblocking poll 的函数耗时误当 issue-to-CQ
   latency。每个物理请求的 retention/promotion utility 最多计一次。
8. preview 的前 `commit-width` 个请求组成 guaranteed core wave，其余请求组成
   可丢弃 tail。二者作为一个 split descriptor 只进入 critical owner ring。
   QP owner 先收集服务边界上所有 critical descriptor；仅当已形成的 SQ train
   仍有空闲 WQE 时，才用一个 doorbell 发布
   `[all critical READ][critical CQ fence][admitted tail READ][tail CQ fence]`。
   第一 CQ fence 立即发布 core completion，第二 fence 只发布 tail。若 critical
   backlog、descriptor 饱和或 SQ credit 不足，tail 以 `-EAGAIN` fail-soft 丢弃，
   不进入另一个 QP/doorbell。该调度是 work-conserving non-preemptive priority：
   服务边界上已可见的 critical 永远优先；一个已经 doorbell 的有界 tail 仍会占用
   当前 QP interval，之后到达的 critical 在下一 train 服务，不能宣称硬件级抢占。
9. core/tail 到达后只在复用的 32 个 coalesced graph scratch slot 中验证
   checksum/incarnation，状态按 `ARRIVED -> VALIDATED` 或 `STALE` 迁移，绝不会提前
   设置 `expanded`、visited 或修改 authoritative Beam。过期 tail payload 可直接
   丢弃；Stable-Run materialize 继续作为唯一权威 Beam 写入点。
10. 解码 8-byte incarnation-tagged RemotePtr，并用对应 PQ code 评分；Live-Extent
   继续按每条静态记录独立的 8-edge 分档长度读取，更新撑档仍走 full fallback。
11. 去重、合并并裁剪到 traversal beam；达到收敛或 `gpu-max-expansions` 后进入精排。

query slot 是所有 speculative DMA 的生命周期边界。正常精排、graph/dynamic-code
失败以及 final frontier drain 失败在发布 completion 前都必须先收割 core、tail 和
terminal-exact descriptor；这也包括下一 epoch 开始处的 authoritative graph fetch
失败，因为上一 epoch 发出的 terminal train 此时仍可能 active。已知 `graph_failed`
的 epoch 禁止再发起 terminal wave。
因此 host 观察到 completion 并归还 slot 时，不可能仍有旧 generation 的 RDMA 写入
该 slot 的 graph/exact scratch。异常路径上的结果只用于 quiesce，不参与 Beam、visited
或返回结果。

### Adaptive Speculative Frontier

Frontier reorder buffer 位于 query CTA 的 shared memory。每个 24-byte 条目物理保存
node handle、issue epoch、issue 时 beam rank、scratch slot、传输字节数、priority、
completion state 和 validation state；query id 是 CTA descriptor 标量，request id
由 `epoch * capacity + slot` 派生，record location 保持在 coalesced request SoA，
避免逐条重复存储。ROB 生命周期只由 CTA 维护，因此没有 global queue 或 CPU
scheduler。

shadow 历史只参与宽度控制，不写 visited/expanded，也不会在完整 merge barrier
前成为下一轮的 authoritative Beam。它把 shadow handle 分为“下一次精确对账时
进入 commit prefix”和“仍在 tail 或被替换”两类；前者才计 promotion/retention，
后者计浪费。物理请求失败会被标为 stale 并释放，随后同一 handle 仍可走 critical
重试路径。同一个物理 speculative read 的 controller utility 在首次
retention 或 promotion 时只计一次（promotion telemetry 仍按真实命中统计）；
累计 useful/waste 比例可被后续运行状态反转，不需要离线训练或数据集阈值。

commit 匹配、ROB 回收和 issue allocation 由一个 warp 分别以 commit position、
ROB slot 和 preview position 为 lane 粒度执行，ballot 后直接压缩 critical miss；
不存在 lane-0 的 32x32 ROB 扫描。core issue 同样以一个 request/lane 组织，用
warp match-any 按目标 shard 聚合一次 batch，而不再执行
`num_shards × request_count` 扫描；batch timestamp 数组也由 warp/CTA 协作清零。
COSSF 的 accumulator 使用 `CandidateWorkspace` 已有的两个 Beam-sized segment，
当前 sorted interval 复用既有 rerank scratch 的首段；每个区间 fold 完后即可覆盖
scratch，不保留按数据集扩张的候选缓存。selected expanded overlay 是唯一 rank 一次
写入，没有 `K × commit-width` 扫描或全局原子。最终 Issue 输出使用四个 warp 的
ballot/compaction。SRFC fallback 则在已准备的 Stable-Run leaves 与只读 old Beam
上构造 dependency-closed 精确前缀，split RDMA 发布后由原 authoritative
materializer 复用同一棵 merge tree 的结果。两条路径都只使用 CTA-local shared
metadata，且不会重新实现或绕开 Stable-Run 的排序语义。

DEEC（Dominance-Envelope Exact Certificate）及 PBEC 仍作为精确证明边界与微基准
保留在 CUDA 测试中，但不再位于生产热路径。实测 workload 的 DEEC fit 不足时，
其 raw-candidate 扫描和 fallback 会重复 Stable-Run 的必要工作；COSSF/SRFC 用
“必要工作只做一次、并在通信区间内推进”的结构性约束替代该数据分布相关分支。

请求完成不能直接改变搜索语义。只有 commit 阶段在冻结的 rank 顺序下把 `COMMITTED`
条目映射到 `graph_record_slots`，随后才执行邻居生成、visited 插入、PQ distance
和 Stable-Run merge。这样 speculative read 可以隐藏 RDMA 延迟，但不会因为提前收到
邻居而影响 beam 排序或 termination。

并发更新下沿用现有逐记录一致性契约：一次 graph record 读取在线性化于该查询区间内
的 RDMA snapshot，incarnation/checksum 验证排除 torn 或已失效记录；查询不承诺所有
记录来自同一全局版本。ASFE 只把这个合法 snapshot 提前到 issue 时刻，静态或无并发
写入时与 coupled baseline 位级等价。若未来接口要求“commit 时刻的最新记录”而不是
“查询区间内的一致记录”，必须增加版本重验证；当前系统与 baseline 都不提供该更强的
全局快照语义。

completion/report 输出 `logical_expansions`、critical/core/tail reads/bytes、
core-ready wave ratio、tail promotion/waste ratio、tail wasted bytes、owner
WQE submission utilization、critical/speculative completion latency、
issue-capacity utilization、commit/issue width、reusable-certificate count
及其实际发布数、每 query 的 reusable prefix ranks、覆盖完整
`traversal_capacity` 的 certificate 数、CQ ordered/completion PQ 分段，
以及后续 core CQ 尚未完成时真正提前密封并 fold 的 candidate-run 数。`issued`
只在 critical core
batch 确实进入 active 状态后计数；`full-prefix` 只表示
`fused_tree_prefix == traversal_capacity`，不能由 shadow 输出恰好填满 preview
误判。prefix-rank 和 streamed-run 都按实际完成工作累加，因此可用于验证重叠工作
是否存在而不是仅验证代码分支被访问。PQ 分段计数严格定义如下：

- `ordered_score_batches/candidates` 只统计 COSSF 已建立私有 accumulator、候选数
  非零且调用 `approximate_handles_batch` 时仍有 critical core CQ 未完成的调用；
- `completion_score_batches/candidates` 只统计 critical core 已全部完成后执行的
  非零 PQ 调用，包括此前保留的 candidate-tile residue 和最后到达/critical retry
  父节点产生的连续候选 suffix；
- 两类 batch 都不把零候选控制路径计为调用，候选数是传给评分函数的实际 `count`。

三个 PQ 计数复用已退出生产热路径的 DEEC completion 槽；独立的 terminal exact
cache 计数将 completion descriptor 从 528 bytes 增至 552 bytes，但仍直接写入
已有 CTA-local shared completion 对象，因此没有新增 per-query global allocation
或 cache buffer。报告还包含 exact snapshot train attempts/fallback ratio。
Live-Extent 的物理字节仍
以实际 admitted WQE 计数；owner 因 SQ slack 不足而拒绝的 tail 会一次性回退
producer 的乐观 reads/bytes/batch 记账，只进入 queue-reject 压力，不算 wasted bytes。
已读取但未晋升的 tail 才进入 wasted-byte 统计，且两者都不会被算作 logical
expansion。这里的 `remote_batches` 是每 shard、每 priority 的逻辑 batch，不是
doorbell 或 CQE 数；物理 train 利用率应使用 owner submitted-WQE/capacity 及
critical/speculative owner-batch 指标。

冻结 A/B 汇总器把上述 COSSF 原始计数字段作为报告 schema 合约；任何旧服务进程或
旧 benchmark binary 生成、缺少这些字段的报告都会直接拒绝汇总，而不会与新结果
混合成一个看似有效的性能结论。它还要求 coupled baseline 的 certificate/ordered
计数为零，并要求 candidate 至少完成一个 reusable certificate、实际发布一个
下一轮 core wave 且执行一次非空 ordered/completion score batch，防止“配置写着
adaptive、运行时却没有进入解耦路径”的假 A/B。

公平性能验收使用固定工作量、配对交错顺序和相同 Stable-Run + Live-Extent 栈：

```bash
REPETITIONS=10 CONCURRENCIES="1 8 64 256" \
  ./experiment/run_asfe_ab.sh
```

baseline 唯一差异是 `issue-width == commit-width`；candidate 允许 controller 在
commit width 与 workspace cap 之间选择 issue width。脚本报告每个并发度的配对 QPS
几何均值、mean/P99 latency ratio、recall delta，并用单侧 95% 置信界验证 30% 吞吐
目标，避免依赖一次短时运行。

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
2. 每个有候选的物理分片发布一个 mandatory fenced snapshot train。请求 SoA 为
   `[完整 fixed records][同序第二 header snapshots]`；QP owner 把 descriptor
   作为不可分割的 critical train，第一条 trailer READ 带 mlx5 initiator fence，
   因而只有在全部完整记录 READ 完成后才会启动第二次 header 读取；
3. 最终 trailer READ（或硬件需要时的唯一 final dump WQE）产生一次 CQE。查询 CTA
   只执行一次 `issue -> wait`，但仍用前后 header、tombstone、generation 和 slot
   incarnation 验证更新一致性；它没有缓存远端向量，也没有改变存储记录格式；
4. 若 train 在发布前因 SQ 容量或兼容性原因不能执行，该分片回退到既有的
   `完整记录 issue/wait -> header issue/wait` 协议。只有 fallback 也发生最终
   transport failure 时整条查询以 `exact_fetch` 失败，绝不静默发布其他分片组成的
   partial rerank；合法 tombstone/incarnation/snapshot visibility reject 仍只过滤
   对应候选；
5. 按 metadata dtype 解码 uint8、int8 或 float32，计算精确 L2，并按距离返回 top-k。

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
one-sided READ 前，从 1 byte/base-node 离线 sidecar 初始化的 device high-water
表取得长度档；目标 scratch 仍是完整物理记录大小，但不再写入未传输 suffix。
校验器只扫描 header 声明且已传输的有效前缀，并以模 `2^32` 的 FNV prime 幂把逻辑
零 suffix 延续到完整记录长度，因此结果与 canonical full-record checksum 完全等价。
返回 header 声明的 live prefix 超出已读范围，或逻辑补零后的结构/checksum 无效时，
该请求在下一 snapshot attempt
升级为 full read。若原因是明确的 extent underhint，且随后 full snapshot 通过
checksum，CTA 用 packed-u32 CAS 单调提升该节点的 u8 class；checksum/torn failure
和 dynamic handle 不更新表。动态记录没有静态 ordinal，始终 full read。由此，每个
已经读到过期 hint 的在途查询至多升级一次；首个成功 CAS 发布新 class 后，后续查询
不再为同一档位重复 fallback。竞争查询可能同时进入 full fallback，但不能造成邻接
截断。

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
- 可选 Live-Extent 的 `align_up(N,4)` byte packed class high-water table，以及每 query slot
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
- Live-Extent 保留固定远端记录的更新余量，但只把当前 high-water 长度档覆盖的邻接
  前缀搬过网络；更新触发的首次 underhint 由完整校验后的 GPU 自学习消除重复 fallback；
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
