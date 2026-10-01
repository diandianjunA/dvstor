# 查询与动态更新

## 后端与索引契约

`gpu-rdma-search-progression-mode=coupled` 选择 HostOrchestrated；`decoupled` 和
`manual` 选择 Persistent。前者由 CPU 维护导航状态和 RDMA 波次，调用 CUDA 距离 kernel；
后者在持久化 GPU kernel 中完成导航和 GPUNetIO 通信。本文的 CTA、ring 和提前读取流程
针对 Persistent，路由、存储格式和逐记录一致性由两种后端共用。

索引要求 schema 16、L2、原始 dtype 精确向量、OPQ 与 8-bit PQ 子空间，以及不超过
2048 B 的紧凑图记录。`RemotePtr` 占 8 B，包含 shard、16 B 对齐 offset 和 24-bit
slot incarnation。文件组成见 [根 README](../README.md#索引与部署)。

## 启动与路由

1. 计算节点校验 metadata、PQ 模型、分片布局及所需 extent sidecar。在线路由不读取
   metadata 的 medoid 或离线采样入口。
2. 存储节点加载 `.dat`、idmap 和 `.centroid`，注册基础 PQ 码流并公布 centroid
   publication descriptor。sidecar 的 dtype、构建指纹、分片指纹和 checksum 必须匹配。
3. Persistent 引擎通过启动期 CPU-posted GPUDirect RDMA 将基础 PQ code 导入最终 GPU
   数组，并抽样核对码流；payload 不经过计算节点主机缓冲。
4. 建立长期存活的 GPUNetIO QP，执行 RDMA probe，启动持久化 kernel。
5. 安装全部分片的首份完整 centroid publication 后才开放查询准入。启动失败不退回静态入口。

每个物理分片的 `CentroidRouter` 用补偿式 FP64 sum/count 维护中心，保留最多 4 个真实
存活图入口。维护线程批量发布不可变快照，再通过包含 sequence bracket、checksum、
magic/version 和 shard identity 的变长 publication 暴露给计算节点。

计算侧将同一份 canonical FP32 centroid 同时装入 CPU home selector 和 GPU 路由表。
两者使用相同的 FP32 逐维 FMA 与分片 tie-break。GPU control CTA 原子安装路由事务；
它只处理中心、版本和入口，不参与 mutation 编码。运行中读到撕裂 publication 时保留
上一份完整状态并重试。查询选择最近的一个可用物理分片，以其 live entries 初始化 beam；
后续跨分片遍历由真实图边引入。

## Persistent 请求生命周期

CPU admission thread 领取空闲 query slot，将输入写入预分配区域并立即发布提交描述符。
查询 CTA、路由 control CTA 和 QP-owner warp 使用各自的长期资源。提交环和完成环均有界，
热路径不创建 stream、QP 或 GPU 分配；CPU completion thread 组装结果并归还 slot。

每个查询按以下顺序执行：

1. 将 query 转为 float，应用 OPQ，构建每个 PQ 子空间的 256 项距离表。
2. 从完整 centroid route 选择入口，初始化 beam 和 visited。
3. 冻结本轮 commit frontier，并发读取其父节点。CQ 可乱序完成，但完成事件只更新
   查询私有状态，不能直接修改 beam、visited、expanded 或终止条件。
4. 所有 mandatory dependency 通过结构、checksum、incarnation 验证后，按 parent rank
   和 neighbor ordinal 形成规范候选流，统一去重和 PQ 评分。
5. Stable-Run 合并候选与旧 beam，裁剪到 traversal beam。达到收敛或最大展开数后进入精排。

基础节点使用常驻 PQ code；动态节点按需读取存储侧的 tag、PQ code 和 checksum，
并按物理槽位缓存。缓存命中在评分前后双采样 incarnation/state；复用交叠时丢弃该距离，
转入同批的存储读取。动态缓存不会成为 mutation authority，也不需要向计算节点广播更新。

## 提前读取与推测前沿

提前读取使用 Stable-Run 的不可变输入，生成下一轮 unexpanded frontier 的精确前缀
certificate。在完整 beam materialization 前发出该前沿的 critical RDMA，使网络读取
与剩余 merge/copy 重叠。完整 beam、expanded 和 termination 仍只在合并后发布一次。

`commit-width` 决定本轮允许扩展的节点数；`issue-width` 是物理预取上限。超过 commit
前缀的 shadow tail 只参与预取，不能成为本轮搜索副作用。CTA-local controller 根据
promotion/retention、stale 和 owner queue rejection 在两者之间调整实际宽度。
`manual` 模式可令两者相等以关闭 tail，并单独切换 exact-frontier early issue。

QP owner 在一个服务边界优先收集 critical 请求，仅在 SQ train 尚有余量时接纳 tail。
core/tail 使用各自的 CQ fence；tail 容量不足时可被丢弃，不算传输失败。已发出的 tail
不能被硬件抢占，后来的 critical 请求在下一 train 服务。ROB 和 graph scratch 均为
查询私有的有界资源，乱序 CQ 收割不会拆成多个小批候选评分。

query slot 是全部 DMA 的生命周期边界。正常结束及任何错误路径发布 completion 前，
都必须收割尚在途的 core、tail 和 terminal-exact 描述符，确保 slot 复用后不会收到上一
请求的 RDMA 写入。失败路径中的收割只用于等待资源静止，不参与结果生成。

## 精确重排与故障处理

最终候选按原始 dtype（uint8、int8 或 float32）计算 L2，再返回 top-k。PQ 误差影响
候选覆盖，最终距离由精确向量决定；召回率还受实时入口、beam、展开预算和精排宽度影响。

Persistent 为每个有候选的分片提交 fenced snapshot train，先读完整精确记录，再读同序
header trailer。前后 header、tombstone、generation 与 incarnation 共同检查记录可见性。
train 在提交前因容量或兼容性原因不可用时，使用完整记录和第二次 header 的两阶段读取。
任一分片发生最终传输失败时整条查询失败，不能返回剩余分片组成的部分精排结果。

合法的 tombstone、代际变化和 stale pointer 会过滤对应候选。传输错误、重试后仍无效的
结构/checksum、ring/slot 不一致或持久化 kernel 错误会终止查询或引擎。系统不做 CPU
查询回退，也不会把低召回结果当作成功继续统计。

## 更新、authority 与物理放置

逻辑 authority 由 `ID % N` 决定，负责串行化 ID 代际和 idmap；物理 home 则由数据分布
决定，可以与 authority 不同。计算和存储两侧在创建事务前校验
`[0, vector-id-namespace-size)` ID 命名空间；未显式配置时才由 `max-vectors` 派生。
基础目录保存紧凑的 `ID -> RemotePtr`，只有参与更新的 ID 分配完整事务状态。

更新完成模式与查询后端分别配置：

- `storage-owner-update-completion-mode=coupled`：只接受 fresh insert，由逻辑 owner
  完成全局搜索、剪枝、稳定记录、质心发布和反向边后，才返回 mutation 成功完成；
  upsert/delete 在副作用前拒绝。协议的准入 ACK 先于独立的完成消息，不能作为完成 fence。
- `decoupled`：支持 insert、upsert 和 delete，使用 Stage1/Stage2 维护流程。

两阶段插入过程：

1. 计算节点先把向量编码为 canonical dtype，从已安装的 centroid 快照选择 Stage1 home。
   authority 校验 ID、代际和 placement，再在本地执行或转发到物理 home。
2. home 从当前 live entries 执行宽度 `L` 的本地搜索，写入节点和 provisional 出边。
   本地临时反向桥至少获得一个 ACK 后提交 Stage1；普通查询同时遍历 stable/provisional
   邻接，因此 Stage1 ACK 后新节点可由图遍历发现。
3. Stage2 延续保存的 beam、visited 和 remote frontier，只沿图边通过 one-sided RDMA
   补充远端候选，不重启逐分片搜索。
4. 收敛后统一 RobustPrune；仅在严格减少跨分片边时迁移物理 placement。先将一个临时桥
   提升为 `R` 有界的 stable 边，再协调其余反向边、清除临时边，更新 centroid membership
   并推进维护完成水位。

临时桥使用独立 provisional 邻接区，Stage1 最多安装两个。迁移、反向边和重试按
ID/generation 协调；迁移目标受有界 receipt 保护，避免迟到请求重复分配或命中新对象。
Stage2 日志报告远端 frontier、展开与评分量、迁移、跨分片边变化和预算耗尽。

## 一致性与恢复边界

查询采用 incarnation-tagged read-committed：不同记录可以来自查询期间不同的已提交
版本，不提供跨计算节点全局快照。durable watermark 后旧地址才可复用；复用时递增
incarnation，先写正文，最后发布 header。旧 tagged pointer 只会被判为 stale；incarnation
耗尽的槽永久退休，不回绕。

当前没有完整入边索引。delete/upsert 可以处理协议已知的边，但不能证明历史未知入边
已从全图移除；查询靠 tombstone/generation 过滤失效版本，残留指针仍可能影响长期遍历
开销与质量。Stage2 continuation 的等价范围是同一搜索状态和剪枝协议，不等价于离线
全候选构图，也不包含全图清理承诺。

durable sequence 表示进程内维护完成边界，不是崩溃一致的 WAL。带未完成内存态
maintenance 的恢复会被拒绝；进程或主机故障后需要重新部署一致快照。

## 预算与观测

Persistent 启动前按 `gpu-memory-limit-gb - gpu-memory-reserve-gb` 核算基础 PQ、路由、
查询状态、动态 PQ arena 和可选 extent 数组；溢出或超限直接失败。基础 PQ 占 `N * M` B，
PQ32 的 100M/1B 规模分别为 3.2/32 GB。默认 40 GiB 显存上限中留出 4 GiB 给 CUDA/DOCA。

验证提前读取时查看 reusable certificate、实际 issued certificate、completion score
batch、critical/tail reads、promotion/waste 和 owner WQE 利用率。逻辑展开量与物理
重读量分别统计，`remote_batches` 不是 doorbell/CQE 数。图读取字节和 extent 回退的口径见
[Live-Extent RDMA](live_extent_rdma.md)。性能评估同时检查 QPS、延迟、Recall、GPU 资源、
RDMA/路由失败和 Stage2 积压，实验步骤见 [实验说明](../experiment/README.md)。
