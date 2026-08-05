# DVStor 面向 FAST 2027 的四项核心贡献提炼

> 状态：基于 2026-08-05 的代码与原始实验报告整理。本文把“已有严格证据”“只有运行机制证据”和“尚待实验”分开书写；没有把设计文档中的预期结果当成事实。

## 1. 论文应该讲的总故事

DVStor 要解决的不是普通的单机动态图索引问题，而是一个更尖锐的矛盾：**在计算与内存解耦的动态图 ANN 系统中，GPU 查询、远端图访问和在线更新彼此依赖，传统实现会在错误的边界上同步，或为更新预留/搬运大量当前根本无用的数据。**

四项贡献可以由同一个原则串起来：

> **跨越 GPU、网络和分片边界时，只同步正确性真正需要的状态，只传当前有效的数据，并在信息自然变得充分的时刻作决定。**

对应到四项贡献：

| 贡献 | 传统路径等待或付出的多余成本 | DVStor 的核心洞见 | 当前最强证据 |
|---|---|---|---|
| C1：精确前缀归并与提前 RDMA | 必须先生成并写完 128 项新候选表，才能读取下一轮最靠前的 16 个节点 | 输入已经是几张不再变化的有序表，因此可以只算出下一轮确实要访问的前 16 个节点并立即发 RDMA，再在传输期间完成其余归并 | 每查询 13.262 轮成功在完整归并前发出下一轮核心读取，相对 14.369 个可比较轮次的保守计数比为 92.30%；端到端因果 A/B 待补 |
| C2：可达性优先的两阶段插入 | 插入 ACK 同步承担完整跨分片搜索、向量读取和反向边维护 | 前台只需先建立可查询的最小可达性；后台从保存的 local Beam 与 remote frontier 续跑并在存储 owner 侧融合 | 每个 Stage2 平均承接 647.20 个远端 frontier item；独立远程 vector-access item 是 logical resolved score 的 56.89%，具体来源归因待补；120 秒混合负载下后台债务可排空 |
| C3：代际安全的自愈 Live/DynaExtent | 为支持原地更新而固定分配 832 B，查询也被迫每次搬 832 B | 物理容量不等于网络传输量；extent 只需是可验证、可自愈、与 incarnation 绑定的性能 hint | 静态受控单次配对中 graph bytes/query -49.64%、QPS +8.90%；动态节点目标路径平均 431.05 B，相比 832 B 减少 48.19% |
| C4：更新边界局部出边修复 | centroid 无法为在线节点直接给出最终图邻域最合适的分片；周期性重跑 METIS 又太贵 | 最终邻居已在 Stage2 自然产生，此时只需统计邻居所在分片并仅在严格获益时迁移 | 在本次 finalized 节点的固定 final adjacency 下，仅迁移 10.69% 的节点便使其 outgoing cross edges 减少 23.12%；长期与端到端效果待补 |

这四项贡献分别回答四个不同问题：C1 解决查询轮次之间“能否在完整候选表生成前就发出下一批远端读”；C2 解决更新“何时可以 ACK、剩余构图如何继续”；C3 解决“一次远端图读到底需要搬多少字节”；C4 解决“更新完成后节点物理上应该放在哪里”。它们共享执行基础，但不是四个 CUDA/网络小优化。

## 2. 叙事方式：借鉴什么，不照搬什么

本文采用 [FlowANN](https://www.usenix.org/conference/osdi26/presentation/zhao)、[DistVS](https://www.usenix.org/conference/nsdi26/presentation/yin) 和 [OdinANN](https://www.usenix.org/conference/fast26/presentation/guo) 的共同组织方法：

1. 先用 trace、profile 或负结果量化用户能感知的矛盾；
2. 再指出表面瓶颈背后被设得过强的依赖或不匹配的物理布局；
3. 用一句反直觉但容易复述的洞见解释核心方案；
4. 只保留为实现该洞见所必需的两三个机制；
5. 用“动机实验、机制指标、消融、等召回端到端”四层证据闭环。

三篇论文只是叙事参照，不足以完成 novelty 判定。下文中的“创新主张”是当前最有希望的论文 claim，正式投稿前仍需对动态图 ANN、disaggregated ANN、GPU graph search、online graph partitioning 和 update offloading 做完整 related-work audit。

## 3. C1：精确前缀归并——提前发下一轮 RDMA，用网络时间隐藏剩余 Beam 合并

### 3.1 先用一句话讲清核心思想

你的理解基本正确：**完整合并的目标是生成下一轮整张候选表，但发起下一轮 RDMA 只需要知道其中最靠前、尚未访问的 16 个节点。只要这 16 个节点已经精确确定，就可以立刻读取它们的邻接表，同时继续完成候选表剩余部分的合并和写回。**

可以把它理解成合并几份已经按成绩排序的名单：最终要抄出前 128 名总榜，但只需通知其中最靠前、尚未处理的 16 人提交材料。传统方法先抄完整张榜再通知；DVStor 在前 16 人确定后立即通知，同时继续抄第 17–128 名。已经封存的有序名单中，后面的名字不可能再插到这 16 人前面，所以提前通知不是预测。

旧路径：

```text
给新候选打分并排序 -> 生成并写完 128 项新 Beam -> 取出前 16 个未访问节点 -> RDMA 读取
```

DVStor：

```text
给新候选打分并排序 -> 只精确算出前 16 个未访问节点 -> 立即发 RDMA
                                                    └-> 同时生成并写完 128 项新 Beam
```

需要强调一个边界：新候选的打分和分段排序仍然必须完成；本方案重叠的是**完整多路合并及 Beam 写回的剩余工作**与下一轮 RDMA，而不是把整个候选生成阶段都隐藏掉。

### 3.2 最少术语表

| 术语 | 通俗含义 |
|---|---|
| Beam（候选表） | 当前搜索保留的 K 个最有希望节点，按与查询的近似距离排序；当前 K=128。它是搜索过程中的工作表，不是最终返回结果。 |
| expanded（已访问） | 已经读取过该节点的邻接表、并用它发现过新候选。下一轮只选择尚未访问的 Beam 节点。 |
| candidate run（有序候选子表） | 本轮新发现节点经过去重和距离计算后形成的若干有序短表。run 只是“排好序的一段数组”。 |
| Stable-Run | 用固定平局规则合并这些有序短表：距离相同时保持原输入顺序，并让旧 Beam 排在新候选之前。`stable` 只表示结果确定且与原搜索规则一致。 |
| Beam materialization（完整生成） | 计算并写出合并后的全部 K 项 Beam；当前就是生成并写出完整的 128 项候选表。 |
| Beam publication（正式提交） | GPU 线程块把完整 Beam 写好并经过同步屏障后，后续搜索阶段才允许使用它。“原子”描述的是这种逻辑可见性：只能看到完整旧表或完整新表，不能看到写到一半的混合表；它不是一次硬件原子指令。“规范”是指排序和平局规则固定，不受 RDMA 返回顺序影响。它不是生成 candidate run，也不是等待其他查询结束。 |
| mandatory dependency（本轮必读数据） | 当前轮已经决定要访问的节点邻接表；这些数据必须全部读回并验证，才能生成本轮新候选。后文直接称“本轮必读节点”。 |
| ROB（乱序收件箱） | 每个查询私有的一张小表，记录提前发出的节点、结果存放位置和是否返回。RDMA 响应可以乱序放入表中，但不能凭到达顺序修改 Beam。 |

### 3.3 实验观察：完整 Beam 合并位于下一轮读取之前

在 SIFT100M、concurrency=256、graph prefetch depth=16 的旧路径中：

| 指标 | legacy merge | Stable-Run | 变化 |
|---|---:|---:|---:|
| Beam merge/query | 2.403 ms | 1.089 ms | -54.69% |
| RDMA wait/query | 0.798 ms | 1.035 ms | 吞吐提高后的瓶颈迁移 |
| QPS | 48,846 | 58,681 | +20.14% |
| mean latency | 5.234 ms | 4.357 ms | -16.75% |
| P99 latency | 6.434 ms | 5.370 ms | -16.54% |
| Recall@10 | 0.935 | 0.935 | 不变 |

Stable-Run 已经证明复用旧 Beam 的有序性非常有效，但它也暴露了下一层问题：旧路径每查询仍累计花费约 1.09 ms materialize Beam，而且代码依赖强制每一轮的下一批 RDMA 位于该轮 materialization 之后。这里的 1.09 ms 是**每查询累计 merge 时间，不是每轮都有 1.09 ms 的连续空洞**；实际能隐藏多少仍需 overlap timestamp A/B 测量。进一步优化某个排序 kernel 只能缩短这些串行段，不能消除阶段边界本身。

这组 Stable-Run 结果是同一冻结 build 的单次、5 秒、非随机 policy-major 对照；吞吐不同还使两侧消费了不同 query prefix。它适合证明现象与机制潜力，但不是具有置信区间的最终论文结果。

### 3.4 candidate run 到底从哪里来

它不是为了“提前读取”额外生成的一份预测结果，而是完整 Beam 合并本来就需要的中间结果。以当前配置为例，每轮按下面的过程产生它：

1. 当前查询先等齐**本轮已经选中的节点**的邻接表。它只等自己的这一批读取，不等待其他查询结束；这些响应可以按任意顺序到达。
2. GPU 一次性解析这些邻接表，对新发现的节点去重并计算近似距离。本轮最多产生 2,048 个新候选。
3. GPU 将新候选分成最多四段，每段最多 512 项，并行排序每一段。每一段排好序后就是一个 candidate run，也就是“一张有序的新候选子表”。
4. 每段只需保留前 128 项，因为某个候选若在自己这段中已经排在第 128 项之后，它前面至少有 128 个不差于它的候选，无论其他段是什么，它都不可能进入全局前 128。

与此同时，上一轮的 128 项 Beam 本来就是有序表。因此，完整的新 Beam 最终只是把“旧 Beam + 最多四张新候选子表”按同一距离规则归并，取全局前 128。candidate run 的排序不能省略，但也不是提前读取额外付出的工作；新方案只是复用这些已经排好的输入，避免在发 RDMA 前先把 128 项归并结果全部写出来。

### 3.5 如何不做完整 merge，就精确找到下一轮的 16 个节点

先看一个缩小后的例子。括号中的 `已访问` 表示该节点已经展开过，不能在下一轮再次选择：

```text
旧 Beam：       [1(已访问), 4, 8]
新候选子表 A： [2, 6]
新候选子表 B： [3, 5]

完整归并结果： [1(已访问), 2, 3, 4, 5, 6, 8]
下轮前三个：                 [2, 3, 4]
```

最直观的做法是只比较各表当前最前面的元素，连续取出三个最小值，然后停止；显然没有必要继续生成后四项。论文的核心到这里其实已经讲完：**只做前缀归并，不做完整归并。** `co-rank` 只是让 GPU 并行计算这个前缀的实现方法，不是另一个设计思想。

#### co-rank 先解决一个什么问题

假设现在只合并两张有序表 `A` 和 `B`，要找合并结果中下标为 `r` 的元素，下标从 0 开始。`co-rank(r)` 不直接返回这个元素，而是回答：

> 排在第 `r` 项之前的 `r` 个元素中，有多少个来自 A？

若答案是 `a`，那么来自 B 的数量必然是 `b=r-a`。这相当于分别在 A 的第 `a` 项之前和 B 的第 `b` 项之前各切一刀：

```text
A：[前 a 项 | A[a], ...]
B：[前 b 项 | B[b], ...]       a + b = r
```

如果这两刀切对了，左边合起来就恰好是完整归并的前 `r` 项。此时第 `r` 项只可能是两刀右边的第一个元素中较小的那个，即 `min(A[a], B[b])`。

怎样判断刀切对了？只需检查分界两侧的四个数：

```text
A[a-1] <= B[b]    A 已选部分的最后一项，不应跑到 B 的下一项之后
B[b-1] <  A[a]    B 已选部分的最后一项，不应跑到 A 的下一项之后
```

第二个条件使用严格小于，是因为距离相同时规定 A 排在 B 前面。若 `A[a-1] > B[b]`，说明从 A 取多了，令 `a` 变小；若 `B[b-1] >= A[a]`，说明从 A 取少了，令 `a` 变大。`a` 的范围是一个连续区间，所以可以二分查找。

#### 手算一次二分过程

```text
A = [1, 2, 3, 4, 9]
B = [5, 6, 7, 8, 10]
```

现在求合并结果中下标 `r=4` 的元素，也就是第 5 项。它前面必须恰好有 4 项，因此始终保持 `a+b=4`：

| 尝试 | 从 A 取 a 项 | 从 B 取 b 项 | 这次假定的前 4 项 | 判断 |
|---|---:|---:|---|---|
| 1 | 2 | 2 | A 的 `[1,2]` 加 B 的 `[5,6]` | 错。B 已取部分的末尾 6 大于 A 的下一项 3，说明 3、4 被漏掉了；从 A 取少了 |
| 2 | 3 | 1 | A 的 `[1,2,3]` 加 B 的 `[5]` | 仍错。5 大于 A 的下一项 4；从 A 取少了 |
| 3 | 4 | 0 | A 的 `[1,2,3,4]` | 正确。这正是完整归并的前 4 项 |

分界找到后，两张表的下一项分别是 `A[4]=9` 和 `B[0]=5`，所以合并结果的第 5 项是 5。整个过程没有生成完整的：

```text
[1,2,3,4,5,6,7,8,9,10]
```

只找到了第 5 项所需的分界。

#### 从两张表扩展到当前系统的最多五张表

当前实现按一棵固定的小归并树处理这些表：

```text
左半边  = merge(未访问的旧 Beam, candidate run 0)
右半边  = merge(candidate run 1, candidate run 2)
前三段  = merge(左半边, 右半边)
最终结果 = merge(前三段, candidate run 3)   // run 3 存在时
```

下一轮真正承诺展开的核心宽度记为 `C`，当前通常 `C=16`。实现还允许计算更宽的预览 `W`，其中 `C <= W <= 32`；第 `C` 项之后只是可丢弃的预测性额外读取，不属于本方案的核心。解释核心算法时可以直接令 `W=C=16`。系统只在每一层计算前 `W` 个位置。这样做是安全的，因为一个归并结果的前 `W` 项不可能依赖任一输入表中下标大于等于 `W` 的元素。

GPU 给输出位置 0 到 `W-1` 各分配一个执行 lane。lane 0 查第 1 项的分界，lane 1 同时查第 2 项，依此类推；它们彼此没有“必须等前一项算完”的依赖。当前通常 `W=16`、上限为 32，而每张表最多 128 项，因此一次两表 co-rank 只需很少几轮二分。系统只生成各层所需的 16 项短前缀，不先写出完整的 128 项 Beam。

#### 它仍有串行层次，为什么不直接逐个比较所有表头

有，而且必须准确表述。当前 co-rank 树只是把串行依赖从“输出项之间”改成了“归并树层次之间”，并没有消灭所有串行过程：

```text
第1层：merge(旧Beam未访问项, run0) 与 merge(run1, run2)
第2层：merge(第1层的两个短前缀)
第3层：merge(第2层结果, run3)          // run3存在时
```

第 2 层必须等待第 1 层，第 3 层必须等待第 2 层，每层之间都有同步。只是同一层中的输出位置 0～15 同时计算，因此依赖深度通常是 2～3 层，而不是 16 个输出依次产生。严格按当前代码看，第1层的左右两个 co-rank 由同一个lane先后执行，并没有把它们再拆给两组lane；算上旧 Beam 压缩和最终排名恢复，整个 preview 函数共有 9 次 CTA 级 `__syncthreads()`，所以这里确实存在不可忽略的常数开销。

用户提出的方法是标准的“五路表头归并”，当前代码也保留了完全等价的实现：维护五个 head，每次比较五个 head，输出最小者并只推进获胜表的 head。它的优点是总比较数少、几乎不需要中间数组；缺点是第 `i+1` 项必须等第 `i` 项决定哪一个 head 被推进，所以得到 16 项至少经过 16 个串行选择轮次。即使用五个 lane 并行比较表头，每轮的 winner reduction、head 更新和下一轮之间仍有依赖。

两者的真实取舍是“总工作量”与“关键路径”之间的取舍：

| 方法 | 总工作量 | 单个查询块的串行依赖 | 更可能占优的情况 |
|---|---|---|---|
| 逐个比较表头 | 小；约为输出数 × 输入表数 | 最多 W 个连续 winner 选择 | W 很小、run 很少，或高并发下更看重总指令数 |
| co-rank 前缀树 | 二分、短前缀写入和层间同步使总工作量更大 | 2～3 个树层；每层的 W 个rank并行 | W 较大，或更看重单查询尽早发出 RDMA |

当前仓库的单查询块诊断测试正好同时实现了这两条路径。在本机 GPU、128-thread CTA、K=128 的五次重复启动中，平均 cycle 如下：

| W | 逐个比较表头 | co-rank 树 | co-rank 相对变化 |
|---:|---:|---:|---:|
| 1 | 2,866 | 3,911 | +36.5%，更慢 |
| 8 | 9,123 | 9,372 | +2.7%，基本持平 |
| 16 | 16,421 | 12,484 | -24.0%，更快 |
| 32 | 31,453 | 15,510 | -50.7%，更快 |

这只能作为实现诊断：测试使用合成距离、单个 CTA，而且总是先运行串行版本再运行 co-rank，尚未排除缓存顺序和多 CTA 吞吐效应。更重要的是，当前串行参考只把五个 head 的下标保存在寄存器中，每轮仍重新读取各 head 的值；更强的表头归并应把五个当前值也缓存到寄存器，只重新加载获胜 run 的新 head。因此这组数字还不是对用户所提方法的最佳实现比较。它说明 `W=16` 时当前 co-rank 相对当前串行参考有初步优势，却不能证明 co-rank 对所有宽度、run 数和并发度都更优。最新端到端运行的平均 issue-capacity 为 16.009、最大实际 issue width 为 17，恰好主要位于当前微基准中 co-rank 开始占优的区间；但仍需正式 A/B。

更稳妥的实现应保留两条精确等价路径，并根据 `(W, candidate-run count)` 选择：例如先验证 `W<=8` 使用寄存器缓存的表头归并、`W>=16` 使用 co-rank 树是否在端到端仍成立。还应比较现有的 warp 表头 reduction 版本，以及只用 warp shuffle 的平衡树版本，后者可能去掉 CTA 级 barrier 和中间 shared-memory 写入。测试必须随机化顺序，并使用真实trace和多CTA负载。这个选择是工程实现，不是论文创新本身；论文贡献仍然是“只生成精确前缀并提前发 RDMA”。

最后还有一个与 co-rank 本身无关的排名修正：前缀归并先得到每个结果在“未访问旧节点 + 新候选”中的临时排名，随后把本应排在它前面的“已访问旧节点”数量补回去。对于旧节点可直接使用保存的原位置；对于新候选，先在完整有序的旧 Beam 中二分找到所有“距离小于等于该候选”的旧节点边界——距离相同时旧 Beam 优先——再读取这个边界之前预先统计好的已访问节点数。这样得到它在完整归并中的真实排名。真实排名超过 127 的项被丢弃，留下的前 16 个才是下一轮真正要读取的节点。

这次排名修正看似绕了一下，却是保证正确性所必需的：已经访问的旧节点虽然不能再次被选中，但它们仍占据完整 Beam 的前 128 个位置，不能简单从排名中抹掉。

这条快速路径的工作量约为 `O(K + W log K)`：一次并行扫描跳过旧 Beam 中已访问项，再由 `W` 个 lane 做很短的二分查找。这里 `K=128`、通常 `W=16`；相比生成并写回完整的 128 项多路归并结果，前缀查找很小，而且能更早得到 RDMA 所需的节点 ID。

它之所以是**精确名单而非预测预取**，有三个原因：

- 所有输入表已经完成打分和排序，此后不会再有新候选插入；
- 前缀查找与完整 merge 使用完全相同的距离比较和平局顺序——距离相同时旧 Beam 优先，其后依次为候选子表 0、1、2、3；
- 每段丢弃的第 129 项及以后不可能进入全局前 128，已访问旧节点对完整排名的影响也已被补回。

因此，提前得到的 16 个节点与“先完整生成 Beam、再从头挑出 16 个未访问节点”的结果逐项相同。

实现只有在精确前缀完整覆盖本轮承诺的 `C` 个核心节点时才提前发 RDMA；如果不足 `C` 项，就放弃早发并回到完整 Beam 生成后的普通路径，而不是拿一张不完整名单冒险执行。

### 3.6 新的执行时间线：什么必须等，什么可以重叠

对单个查询而言，一轮搜索现在按以下顺序进行：

1. 上一轮选出的节点正在接受 RDMA 读取。响应可以乱序到达，先回来的结果先放进该查询自己的“乱序收件箱”（ROB）。
2. 等该查询本轮选中的节点全部读回并验证后，GPU 才统一解析邻居、去重、打分和排序。这里仍然保持“一轮一次批处理”，不会按响应到达顺序改变搜索结果。
3. 旧 Beam 和新候选子表被冻结后，GPU 只计算新 Beam 中下一轮要访问的前 16 个未访问节点。
4. 这 16 个节点一确定，系统立即发出下一轮 RDMA。
5. 网络传输进行的同时，GPU 继续完成完整的前 128 项归并及 Beam 写回。
6. 完整 Beam 写完后，它才成为下一轮正式的搜索状态；如果提前发出的数据已经返回，也只会先停在该查询的私有收件箱中，等正式状态提交后再使用。

所以，这里的“乱序”只表示 **RDMA 响应可以乱序返回和存放**，不表示按返回顺序展开节点。真正的创新也不是“随便乱序执行”，而是：**在不改变搜索顺序的前提下，把下一轮网络读取从完整 Beam 合并之后，移动到精确前缀确定之后。**

这里也不存在“等待所有查询结束再生成 candidate run”。每个查询由自己的 GPU 执行块独立推进；它只需等齐自己本轮已选择的邻接表。所谓 **Beam publication**，只是某一个查询把完整的 128 项新 Beam 写完并设为自己的正式下一轮状态。

### 3.7 已有证据，以及现在还不能声称什么

已有两类证据，但用途不同：

1. 前面的 Stable-Run 对照表说明完整 Beam 处理确实是可观开销，也说明复用有序输入可以在不损失 Recall 的情况下提升性能。不过其中 **QPS +20.14% 是旧版 merge 优化的收益，不是本次“提前发 RDMA”的收益**。
2. 最新 120 秒混合运行 `20260805_093639` 中，每查询平均有 14.369 个可比较的搜索轮次，其中 13.262 轮在完整 Beam 合并前成功发出了下一轮核心读取，保守覆盖率为 **92.30%**。分母还包含没有下一轮节点的终止轮和预算末轮，因此剩余 7.70% 不能全部解释为机制失败。

运行计数还显示，提前读取的数据在需要时已有 **96.73%** 位于查询私有收件箱中；每查询平均有 **25.28** 个较晚排名节点的响应先于较早排名节点返回，说明乱序收包路径确实被使用。同时，候选仍然是一轮一次统一打分，未退化为许多按响应到达顺序触发的小批处理。

这些结果证明“精确前缀早发”已经工作且覆盖了大部分可用轮次，但还没有隔离出它对 QPS 和延迟的因果提升。当前可选的额外预测读取几乎全部浪费（99.9923%），因此不应把“自动寻找任意环境的最优扩展宽度”列为贡献。论文应聚焦无需预测、结果严格不变的前 16 个核心读取；自适应额外读取最多作为资源保护策略。

### 3.8 建议的论文表述与待补实验

中文的一句话贡献可以写成：

> **DVStor 发现，下一轮确实要访问的节点会早于完整候选表生成而被精确确定。它只归并出有序输入中最靠前的未访问节点并立即发起 RDMA，让网络传输与其余候选表归并重叠，同时保持与完整归并完全相同的搜索顺序。**

对应的英文表述：

> DVStor observes that the exact nodes required by the next expansion become known before the full search beam is materialized. It merges only the required prefix of immutable sorted inputs, immediately fetches those nodes, and completes the remaining beam merge under network transfer without changing the canonical search order.

与 FlowANN 的区别应落在这个具体边界上：DVStor 不是允许尚未确定的节点提前参与搜索，而是从已冻结的有序输入中**精确算出**下一轮节点，把“写完完整 Beam 才能发 RDMA”这一软件屏障移除。

- `[待补实验：________________]` 同一只读快照、固定查询序列、其他配置完全相同：完整归并后发 RDMA，对比精确前缀确定后发 RDMA；采用 AB/BA 或 Latin 顺序至少 10 次。
- `[待补实验：________________]` 记录“前缀确定并发出 RDMA”“完整 Beam 写完”“数据返回”三个时间戳，直接量化每轮隐藏了多少 merge 时间。
- `[待补实验：________________]` 报告 QPS、mean/P99 latency、RDMA wait 和 GPU merge time，给出等 Recall 的端到端因果提升与置信区间。
- `[待补实验：________________]` 对固定查询逐轮比较所选节点序列、graph-read hash 和最终 top-k hash，证明提前读取与传统顺序路径逐项等价。
- `[待补实验：________________]` 二维扫描前缀宽度和 candidate-run 数，随机化执行顺序，对比单lane表头归并、warp表头reduction、co-rank树的单CTA延迟与多CTA吞吐；再验证测得的混合选择阈值是否改善端到端。
- `[待补实验：________________]` 扫描 SIFT/Deep/SPACEV、并发度、Beam 大小、每轮读取宽度、GPU/NIC 组合和人工网络延迟，说明收益在何种 merge/网络比例下成立。

## 4. C2：可达性优先的两阶段插入——先可查询，再从原状态续跑完整构图

### 4.1 问题：更新卸载之后，瓶颈从计算变成了跨分片依赖链

把插入卸载到存储节点并不自动消除开销。若存储端仍在 ACK 前完成一次全局 Vamana 构图，它必须沿图跨多个分片读取动态图记录和原始向量，反复经历“取图—发现邻居—取向量—决定下一步”的依赖链。计算节点不忙了，但用户看到的插入延迟和存储侧 RDMA 压力并没有消失。

最新混合运行量化了 Stage2 尾部的规模：每个 continuation 平均包含 **647.20 个 remote-frontier item、21.72 次 remote expansion 和 1,215.78 次 logical score**。它证明待完成的跨分片工作并不小，但它不是同步 one-stage baseline，尚不能单独证明这些工作已经构成前台 ACK 瓶颈；决定性的 ACK latency breakdown 仍是待补实验。

### 4.2 核心洞见

> **一个新节点“已有查询可见路径”和“其全局邻接已经完全优化”不是同一个提交条件。前台只需建立最小但受保护的可达性；其余跨分片搜索应从前台留下的 local Beam 与 remote frontier 继续，而不是在后台重新搜索。**

这项贡献建议命名为 **Reachability-First Continuation Insertion（RFCI，可达性优先的续跑式插入）**。

### 4.3 Stage1：只完成 ACK 真正需要的工作

1. 计算节点根据 storage-canonical centroid 选择一个物理 home，逻辑 authority 仍由 `ID % N` 唯一确定；
2. 该 home 只在本地分片执行宽度 L 的完整搜索。遇到跨分片指针时不在前台追逐，而是把它保存为去重的 remote frontier；
3. 写入 query-visible provisional 节点和 provisional 出边，并安装最多两个受保护的本地反向桥；普通查询同时遍历 stable/provisional 邻接，因此 ACK 后图中已有通向新节点的正常查询路径。当前代码建立了这条结构性可达性，但 ACK 后立即可返回动态节点的概率和 time-to-quality 仍待动态 ground truth 实验；
4. 保存足以续跑的 local Beam、remote frontier 和需要协调的 backlink 状态；Stage2 由这些信息重建私有 visited，而不是保存 Stage1 的完整 visited set；
5. 只有 Stage2 已拿到有界队列 permit、后台任务确实 runnable 后，authority 才允许发布 Stage1 ACK。因而它不是“做一半后把不可偿还的债务丢到后台”。

### 4.4 Stage2 到底由谁搜索：一个全局 Beam，远端分片只执行选中的局部操作

先直接回答：**当前 Stage2 不是让每个分片各自维护一个 Beam、独立跑到局部收敛；Stage1 所在的物理分片始终持有这个插入唯一的 continuation Beam 和 visited set，并决定全局下一步展开哪个节点。** 但它也不是把远端节点的图和所有向量都拉回 Stage1 分片再计算，而是把一次已经决定好的展开交给该节点的物理 owner 执行。

```text
Stage1 所在分片：唯一 Beam + visited
        |
        | 从全局 Beam 选出当前距离最近、尚未展开的节点 x
        v
x 的物理分片：只展开 x，并顺手给与 x 同分片的邻居计算距离
        |
        | 返回邻居 ID；同分片邻居同时返回距离，跨分片邻居暂未评分
        v
Stage1 所在分片：把结果放回同一个 Beam，再决定下一次展开谁
```

因此，远端分片执行的是一个无状态的 **expand-and-score 操作**，不是递归搜索：它不保存这个插入的 Beam，不决定下一步访问哪个节点，也不会从收到的节点开始在本分片自行跑到收敛。最终候选选择和 RobustPrune 仍由 Stage1 owner 上这一个逻辑搜索状态驱动。

这样设计是为了同时避免两个极端：

- 若所有工作都在 Stage1 owner 做，每次远端展开后还要逐个读取邻居向量，形成“远端图读取—返回邻居—远端向量读取”的额外往返；
- 若每个分片各自维护一个局部 Beam 并搜索到收敛，一个插入会变成多个独立搜索。要么所有分片过度探索大量最终进不了全局 top-L 的节点，要么分片之间需要频繁交换 Beam 才能维持全局顺序，重新引入同步和网络往返，而且结果不再自然等价于一个宽度为 L 的连续搜索。

Stage2 首先把 Stage1 保存的本地 Beam 恢复为“已经展开”，再从保存的跨分片 frontier 继续；它不会重新走一遍 Stage1 home。多个插入可以在各自的打分/展开依赖边界交错推进，但每个插入仍有自己的唯一 Beam。

### 4.5 在这种执行方式下，METIS 局部性为什么仍然重要

局部性分片优化的是**一次展开之后，后续工作有多大比例仍能留在同一个物理 owner**，而不是要求搜索控制逻辑也必须分散到每个 owner：

1. **Stage1 直接受益。** Stage1 只访问本地分片并跑到自然收敛。若 METIS 把相关图区域聚在一起，更多有用路径能在 ACK 前用本地内存完成，暴露给 Stage2 的跨分片 frontier 更小；若图被随机切分，Stage1 很快碰到大量边界边，前台本地搜索能完成的工作就很少。
2. **Stage2 的 owner-side fusion 直接受益。** 当远端 owner 展开节点 `x` 时，若 `x` 的大部分邻居也在这个 owner，它可以在同一次 RPC 中读取邻接并计算这些邻居到插入向量的距离。若邻居落在别的分片，只能先返回未评分的 ID，再由协调者向另一个 owner 发起评分读取或 RPC。分片越局部，额外评分 wave 和跨 owner 跳转越少。
3. **后台请求更容易按 owner 合并。** 当前实现会把多个插入同时就绪的展开和评分按物理 owner 成批发送。局部性不能消除每一轮的全局 Beam 决策，但能让更多邻居在一次 owner 请求中被消费，而不是散到许多分片。
4. **后续查询继续受益。** Stage2 最终形成的邻接若主要落在节点 home 附近，查询和未来插入沿图遍历时也会减少跨分片读取；C4 的低开销放置修复正是为了在长期更新后维持这一性质。

所以更准确的概括不是“每个分片本地搜索”，而是：

> **搜索决策集中在一个连续 Beam 中，数据相关操作在物理 owner 就地执行；METIS 局部性决定一次 owner 操作能吸收多少后续工作。**

必须诚实说明这个边界：当前系统利用的是**数据访问与距离计算的 owner 局部性**，并没有让远端 owner 沿本地图连续推进一个局部 Beam。因此它没有利用“分片内自主搜索”的全部潜在局部性。这样做换来了唯一全局 Beam、受控工作量和简单的一致语义；是否值得，需要由减少了多少后续评分访问和网络 wave 来回答，而不能仅凭使用了 METIS 就声称成立。

当前已有数据证明 Stage2 continuation、owner RPC 和跨任务 batching 能持续推进，但还没有把 METIS 对上述三项收益单独隔离出来。投稿前必须增加 METIS、hash/random 分片和逐渐扰乱 METIS 布局的对照，分别报告 Stage1 remote-frontier size、Stage2 cross-owner transitions、home-scored-neighbor ratio、RPC/wave/bytes 以及最终 Recall 和吞吐；否则“局部性减少 Stage2 网络开销”仍只是由实现机制支持的合理推断。

为了减少更新卸载后的网络往返，当前具体数据路径是：不可变基础向量可以直接 one-sided RDMA READ；远端图节点由它的物理 owner 展开，并顺便给同 owner 邻居评分；跨 owner 邻居再进入相应评分 wave。共享物理候选会跨同时活跃的插入去重，等待 RPC/RDMA 的 continuation 会释放执行 lane。主思想仍是“visibility/refinement 两阶段 + resume-not-restart”；owner-side expand-and-score 是减少剩余跨分片往返的关键机制，跨插入 batching 是执行优化。

### 4.6 已有证据与证据边界

`20260805_093639` 的 120 秒、256 query threads + 256 insert threads 混合运行中：

- 完成 8,294,279 次查询和 164,448 次测量期插入，即 **69,051 query/s + 1,369 insert/s**；
- warmup 与测量期共 189,098 次插入，其五个 shard 的 target sequence 与 durable sequence 完全相等；
- Stage2 failure=0，Stage1/Stage2 search-budget exhaustion 均为 0；
- 任一 shard 观测到的最大 backlog 为 59；每 shard completion/admission 上限为 128，五 shard 总 window 为 640。聚合 backlog slope 为 -0.458 item/s，停止负载后 maintenance watermark drain 为 96.0 ms；
- 201,119,775 次 logical resolved score request 对应 114,417,709 次独立远程 vector READ/score-RPC item，即 **56.89%**。其余 43.11% 混合包含本地直接评分、terminal/deleted 候选无需读向量、跨任务去重和 owner expand-score fusion，不能由现有 JSON 拆分归因；
- 165,424 个 batched item 进入 76,609 个 context，平均 2.159 item/context；每个 vector wave 平均聚合 36.76 次物理向量访问。

这些数据证明当前机制在该工作点能持续处理并排空后台债务，也表明独立远程访问数显著小于 logical resolved score 数。但必须保留两条边界：

1. 43.11% 不是相对现有 one-stage baseline 的 A/B，也不能写成 batching 或 home-side scoring 的独立收益；普通 batching 主要减少 wave/RTT，不必然减少 item。投稿前需导出 local、terminal/deleted、deduplicated 和 `home_scored_neighbors` 分项或做消融；
2. 当前没有同步 one-stage baseline，因而不能由 1,369 insert/s 宣称“两阶段比同步插入快多少”。当前 insert mean 186.87 ms、P99 约 1.00 s 还混入 256 个 closed-loop writer 的排队和背压，不能作为纯 Stage1 ACK 延迟。

语义上也不要声称 Stage2 等价于离线 builder 或全局瞬时快照。它在当前 read-committed、incarnation-checked 语义下，从 Stage1 的宽度 L 状态继续并最终做一次 RobustPrune；系统尚无完整入边索引和 crash-consistent WAL。

### 4.7 建议的论文创新主张

> DVStor separates query visibility from global graph refinement: Stage1 acknowledges only after installing a protected local reachability bridge and a runnable maintenance intent, while Stage2 resumes from the saved local Beam and remote frontier instead of restarting, then amortizes residual remote work through owner-local expand-and-score.

与 OdinANN 的区别应强调环境和依赖：OdinANN 的核心是避免批量 merge 对查询造成周期性干扰；DVStor 面对的是 disaggregated storage 上被更新卸载放大的跨分片访问链，并用“受保护可达性 + 可执行债务 + 原状态续跑 + owner-side expand-score fusion”切开 ACK 边界。

### 4.8 待补实验

- `[待实现/待补实验：________________]` 同一快照、相同 insert ID/向量、固定完成数：同步 full search+prune before ACK；two-stage 但 Stage2 restart；continuation；continuation + owner fusion；continuation + owner fusion + batching。
- `[待补遥测：________________]` 单独输出 Stage1 authority ACK service latency，排除 client queue；同时输出 Stage2 durable delay、wire request/response bytes、WQE/CQE/RTT、home-scored neighbors。
- `[待补实验：________________]` 在等 query Recall 和等 query QPS 下比较 insert throughput、ACK P50/P99/P999、query P99 干扰、后台 backlog 与 drain time。
- `[待补实验：________________]` 完成 Stage2 durable fence 后，对包含动态 ID 的 ground truth 测 Recall；扫描更新率、跨分片率、分片数和不同数据集。
- `[待补实验：________________]` 故障/重试、queue saturation 和长时间 churn；若论文不实现 WAL，明确把 crash recovery 排除在语义之外。

## 5. C3：代际安全、自愈的 Live/DynaExtent——只搬当前有效的图前缀

### 5.1 观察一：固定容量记录让查询长期搬运空白更新余量

DVStor 为了允许邻接原地增长，为每个图节点固定保留 832 B；但 SIFT100M 两份独立查询 trace 中，实际扩展父节点平均只有 46.84–47.21 条 live edge。按 8 条边一档读取，平均只需 417.72–420.48 B，即理论上可减少 49.46%–49.79% 的 graph payload。

最直觉的“先读 16 B header，再根据长度追读 body”却是负优化。在 160 个 active QP 下，16+400 B 和 16+448 B 的 dependent 两阶段读取吞吐分别只有对应 one-shot 短读的 **0.454 倍和 0.497 倍**，P50 分别增加 129.024 us 和 109.568 us。问题不只是字节数，而是额外 WQE 和串行 RTT。

因此真正需要的是：**在发 WQE 之前就得到一个安全的长度 hint，并让常见路径仍然只发一次 READ。**

### 5.2 观察二：只为静态节点保存 build-time 长度会在更新后失效

第一版 immutable extent class 的历史单点结果在 40K query/s + 1K insert/s 下虽然仍少传 46.96% graph bytes，却发生 4.6395 次 fallback/query，graph WQE 增加 2.42%、shard batch 增加 11.78%，mean/P99 延迟反而增加 0.77%/2.45%。原因是更新让基础节点的反向边增长，同一个过期 class 被反复命中，每次都支付 `short READ + full READ`。它是设计动机，不是当前同 build 的三组消融。

这正是为什么本方案不能写成“只对静态节点有效的 metadata cache”。它必须同时解决：基础节点的边会更新、新节点没有离线 sidecar、slot 还会删除和复用。

### 5.3 核心洞见

> **Extent 不需要成为强一致 metadata authority；它只需要是一个与节点 incarnation 绑定、可能过期但可由现有 header/checksum 验证，并能在访问时自愈的性能 hint。**

这项贡献建议命名为 **Incarnation-Safe Self-Healing Live/DynaExtent（代际安全的自愈有效前缀 RDMA）**。这里不宣称 extent hint 与每次 graph publication 线性一致；安全性来自 hint 可错、读取结果必验证、错误后回退，以及旧 incarnation 的 hint 不能用于复用后的新节点。

### 5.4 统一处理基础节点和动态节点

基础节点路径：

1. 离线 `.gextent8` 只保存每个基础节点 1 B 的 8-edge extent class；
2. GPU 按 class 发一次短 READ，并用原有 header、结构和完整记录 checksum 的逻辑零后缀延续来验证；
3. 若明确 underhint，则对本次请求 full-read；只有 full snapshot 通过 checksum 后才 CAS 提升 device high-water；
4. 同一增长 class 发布后，后续查询不再重复支付同一 fallback。

动态节点路径 DynaExtent：

1. 动态 PQ tag 的低 24 bit 保存 slot incarnation，高 8 bit 搭载 extent class；
2. 邻接更新把 `832 B graph + 4 B tag` 连续 co-publish 为一次 836 B copy/WRITE，payload 只增加 **0.48%**，不增加 WQE、CQE 或 RTT。这里应写“single-WQE contiguous co-publication”，不能写成硬件原子写；
3. 新节点第一次本来就需要读取 PQ32，当前用一次 40 B READ 同时取得 `tag + PQ32 + checksum`，不增加一次 metadata READ；
4. GPU 将 `{incarnation, class}` 与 PQ payload 一起发布进现有 physical-slot arena。只有 incarnation 精确匹配且状态非 BUSY 才允许短读；cold、unknown 和 recycled slot 直接 full-read；
5. underhint 经 checksum-valid full snapshot 后提升；收缩至少相差两档才降到 `observed+1`，保留一档 hysteresis。arena 在评分前后双采样状态，避免地址复用把旧 hint 用到新节点。

因此边更新不要求广播 metadata，也不要求为每个动态节点增加一张独立 GPU 表。基础节点靠访问时 high-water 修复；动态 cold/new incarnation 通过本来就发生的 update write 和首次 PQ miss 安装 class，而已驻留 hot 节点的增长靠 `short underhint -> validated full -> CAS` 自愈，收缩靠有效 snapshot 和 hysteresis。32-bit checksum 用于发现 mixed/torn payload，仍保留有限的 checksum-collision 边界。

### 5.5 已有静态端到端证据

SIFT100M、concurrency=256、120 秒，除 graph read policy 外配置相同的一组 fixed/live 配对结果：

| 指标 | fixed 832 B | Live-Extent | 变化 |
|---|---:|---:|---:|
| graph bytes/query | 162,595.77 B | 81,878.40 B | **-49.64%** |
| tracked RDMA bytes/query | 182,051.77 B | 101,334.40 B | **-44.34%** |
| physical graph WQE/query | 195.4276 | 195.4297 | +0.0011% |
| GPU RDMA wait/query | 1,029.97 us | 846.14 us | **-17.85%** |
| QPS | 57,912.54 | 63,065.85 | **+8.90%** |
| mean latency | 4,419.04 us | 4,057.86 us | **-8.17%** |
| P99 latency | 5,425.46 us | 4,997.48 us | **-7.89%** |
| Recall@10 | 0.9401 | 0.9401 | 不变 |

它证明了“少搬字节、基本不增加请求数”能转化为端到端收益。不过这仍是一组非随机顺序的 closed-loop、单次 120 秒配对：快侧完成更多请求并消费不同的 single-pass query 前缀，报告没有冻结 binary hash，相同 aggregate Recall 也不等于逐查询 top-k identity。投稿前需要固定序列和多次交错复测。

### 5.6 已有动态节点证据

最新混合运行 `20260805_093639` 中：

- 35,208,264 次动态图 snapshot attempt 中，35,205,635 次为 short read，只有 2,629 次 full fallback；
- 平均物理读取 **431.05 B**，相对固定 832 B 减少 **48.19%**；同次运行的机械全长反事实下共少传 **14.1168 GB**，约 1,701.99 B/query；
- short physical attempt 占 **99.9925%**；`fallback/attempt` 为 **0.00747%**，发生 2,614 次 promotion 和 11 次带 hysteresis 的 demotion；
- 873,944,331 个动态 PQ candidate 中只有 179,321 次 storage miss，每次恰为 40 B，arena hit ratio 为 **99.9795%**，incarnation reject 为 0。

这组数据直接证明 DynaExtent 的动态目标路径少传 48.19% 且自愈机制没有形成重复 fallback 风暴。但动态图 attempt 只占全部 graph read 的约 **2.21%**。若保持静态 Live-Extent 的实际流量不变、只把这些动态 attempt 机械替换为 832 B，全图 graph-byte 反事实收益约 **2.08%**；它不是 Dyna-off 的实测 A/B。因此目前不能声称“DynaExtent 单独提高了端到端 QPS”，更不能把静态 Live-Extent 的 +8.90% 直接归给动态扩展。

### 5.7 建议的论文创新主张

> DVStor decouples update-friendly physical capacity from query transfer size using an incarnation-scoped, self-healing extent hint. The hint piggybacks on writes and PQ misses that dynamic execution already performs, preserving a one-READ common path without metadata invalidation broadcasts.

与 DistVS 的区别在于：DistVS 按候选重要性和存储层级选择不同精度表示；DVStor 处理的是**可原地更新的变长图逻辑内容与固定物理记录之间的错配**，重点是 single-READ、并发更新、slot reuse 和无广播自愈。

### 5.8 待补实验

- `[待补实验：________________]` 同一 build、同一可恢复 snapshot、相同 operation trace 的 fixed / static-only Live-Extent / full DynaExtent 三组 3×3 Latin 顺序实验；当前 runner 已有，但还没有合格结果。
- `[待补实验：________________]` 预老化使动态节点占查询 graph access 的 5%、10%、20%，验证 DynaExtent 能否把目标字节收益转化为 QPS/P99。
- `[待补实验：________________]` insert、upsert、delete、slot reuse 和邻接频繁增减；分别报告 promotion/demotion、fallback、torn/incarnation reject。
- `[待补实验：________________]` 冷启动与 warmed steady state、dynamic-GT Recall、dynamic PQ arena capacity/load-factor sweep。
- `[待补实验：________________]` 多数据集、不同 R/记录容量、不同 GPU/NIC；同时测 NIC wire bytes，而不只测应用层 admitted RDMA bytes。

## 6. C4：更新边界局部出边修复——抑制在线节点侵蚀 METIS 局部性

### 6.1 问题：离线 METIS 只能决定初始图，不能安置后来的节点

当前 SIFT100M 基图由 METIS 分为 5 个物理 shard，初始跨分片边比例为 15.891%。在线插入到来时，Stage1 尚未完成全图邻居搜索，只能用向量到 storage centroid 的距离选择一个临时 home。向量 centroid 是便宜的几何近似，但最终图邻居才真正决定查询会沿多少跨分片边访问远端。

如果之后什么也不做，centroid 的近似误差会被写入在线节点的物理布局；若定期重跑 METIS、扫描并搬迁全图，又会引入额外后台流量、暂停和版本协调。全图重分片的实际成本以及整体局部性是否随更新持续恶化，目前仍需长期实验量化。

最新运行给出了当前能够证明的关键观察：在 165,479 个 Stage2 finalized node 中，**89.31% 的 centroid home 已经与最终邻居多数所在 shard 一致，10.69% 存在可严格改善其 outgoing cut 的误放。** 也就是说，本次更新节点上的局部修复机会是稀疏的；这不能推出长期全图漂移只集中在这些节点，也不能证明全局 repartition 永远没有必要。

### 6.2 核心洞见

> **先不引入周期性全图修复；在一次更新本来就得到最终邻居的自然边界上，顺手计算该节点的最优局部 home，并且只在跨分片出边严格减少时迁移一次。**

这项贡献建议命名为 **Mutation-Boundary Locality Repair（MBLR，更新边界局部性修复）**。

它有一个非常容易解释的精确性质。固定一个节点最终邻居集合 N，把节点放到 shard s 后：

```text
cross_edges(s) = |N| - number_of_neighbors_on_shard_s
```

所以选择“最终邻居最多的 shard”就精确最小化了该节点在固定 N 下的 outgoing cut，不需要重跑 METIS。**placement decision 本身**不需要再访问图中其他节点；此前 Stage2 为得到 N 所做的图/向量访问仍属于 C2 的构图成本。若与 Stage1 home 打平，则留在原地；只有严格多出本地边时才迁移。

### 6.3 机制

1. Stage1 仍用动态 centroid 选择便宜的 provisional home，保证前台路径不做全局 placement 搜索；
2. Stage2 continuation 完成并基于权威 snapshot 做最终 RobustPrune 后，本来就已经拥有最多 R 个 final neighbors；
3. 对这些邻居做 O(R) 计数，再在最多 64 个 touched shard 的常数上界内选择邻居数最多的 shard；tie 保留 Stage1 home；
4. 相对 C4 新增的 placement 工作，89.31% 的节点只付出这次有界计数并原地发布；只有严格改善的 10.69% 分配一个目标 slot、复制单节点 record 并协调 placement。它们仍然支付 C2 的 Stage2 search/prune/reconciliation；
5. 迁移由 authority 的 ID/generation/placement version 串行化，目标分配使用有界 receipt；最终反向边、目录和 centroid membership 发布完成后才退休旧 incarnation。

它不是“高开销长期重放置”：不扫描静态图、不周期运行、不重跑 METIS、不搬迁未触发更新的老节点，也没有一个持续驻留的全图优化器。更准确的定位是 **mutation-triggered、single-node、one-shot locality conservation**。

### 6.4 已有结构性收益

`20260805_093639` 对每个 finalized node 的同一组 final outgoing neighbors，同时计算“留在 Stage1 home”和“采用最终 home”两个局部 counterfactual：

| 指标 | 数值 |
|---|---:|
| Stage2 finalized nodes | 165,479 |
| 实际迁移节点 | 17,684（10.687%） |
| 保持原地 | 89.313% |
| final outgoing edges | 7,901,551 |
| 若全部留在 Stage1 home 的跨分片边 | 1,386,547（17.548%） |
| MBLR 后的跨分片边 | 1,066,037（13.491%） |
| 减少的跨分片边 | 320,510（**23.116%**） |

同一运行中 Stage2 failure=0，任一 shard 最大 backlog 为 59（每 shard 上限 128；五 shard 总 window 640），负载停止后 watermark drain 为 96 ms。这只证明“集成了 MBLR 的完整 Stage2 流水线”在该工作点能够推进，不是 relocation 本身低成本的归因证据。最有力的已证结论是：**对完全相同的 final outgoing neighbors，只迁移约十分之一的 finalized 节点即可减少这些节点 23.12% 的跨分片出边。**

目前仍不能把 QPS、query RDMA 或 insert latency 的变化归因于 MBLR，因为系统尚无关闭 relocation 的正式 A/B，且没有单独统计每次 migration 的复制字节和 CPU/Stage2 时间。当前策略也只有“严格改善至少一条边”这一门槛，没有迁移预算、容量约束收益函数或热度权重，文中不能虚构这些机制。

### 6.5 与 C2 的边界

C2 回答“怎样尽快而正确地让插入可查询，以及后台怎样低 RDMA 地完成构图”；C4 回答“构图完成后物理记录放在哪个 shard”。C4 复用 C2 已经计算出的 final adjacency，因此 placement decision 不增加一次搜索；真正迁移仍有单节点 copy 和 reconciliation 成本。论文必须分别做 continuation/fusion 消融和 relocation 消融，否则 reviewer 很容易认为 C4 只是 C2 的一个 placement heuristic。

如果 relocation on/off 的长期实验不能显示独立的 remote-access 或端到端收益，C4 应降为 C2 的 supporting technique，而不应为了凑足四项贡献强行独立。

### 6.6 建议的论文创新主张

> DVStor locally repairs online-node outgoing cuts without periodic graph scans: it piggybacks a single-node placement calculation on each mutation's final adjacency and migrates only when the same neighbor set yields a strict locality gain.

这里不要声称 global graph-cut optimal、恢复原 METIS cut 或完整长期 re-placement。它只对当前节点、固定 outgoing adjacency 局部最优；不计未知 incoming cut、query heat 和 shard load，也不会主动修复未被更新的静态节点。

### 6.7 待补实验

- `[待实现/待补实验：________________]` 增加 no-relocation ablation；固定 snapshot、mutation trace/order、完成数和最终 graph/placement hash，比较跨分片 graph batch、query RDMA bytes、QPS/P99、insert ACK 与 Stage2 durable latency。
- `[待补遥测：________________]` 单次 migration 的 graph/vector/PQ copy bytes、RPC 数、CPU 时间和 Stage2 delay；分别报告 stay 与 migrate 节点。
- `[待补实验：________________]` 0%、1%、5%、10%、20% 累积更新的长期曲线：跨分片边比例、remote graph reads/query、Recall、QPS 与后台成本。
- `[待补实验：________________]` 不同数据集、分片数和 METIS 初始质量；监控 shard capacity/负载倾斜，验证只优化 edge cut 不会制造热点。
- `[待补实验：________________]` insert/upsert/delete 和 backlink churn；比较 strict-gain=1 与更高收益阈值/显式 migration budget。若不实现新策略，只报告当前 tie-stay 策略。

## 7. 建议直接放入 Introduction 的四条贡献

1. **Exact-frontier GPU execution.** We reveal that authoritative Beam publication is an unnecessarily strong barrier for remote graph access. DVStor extracts an exact next-frontier certificate from immutable Beam/candidate runs, issues mandatory RDMA before full Beam materialization, and tolerates out-of-order completions without fragmenting canonical GPU scoring.
2. **Reachability-first disaggregated insertion.** DVStor separates query visibility from global refinement: Stage1 installs a bounded, query-visible reachability path, while a runnable Stage2 resumes from the saved local Beam and remote frontier and fuses dynamic expansion with owner-local scoring outside the ACK critical path.
3. **Incarnation-safe, self-healing live-extent RDMA.** DVStor decouples fixed update capacity from transfer size with incarnation-scoped, self-healing extent hints piggybacked on existing graph writes and dynamic-PQ misses, preserving a one-READ common path under insert, edge growth and slot reuse.
4. **Mutation-boundary local cut repair.** DVStor limits locality drift introduced by online nodes without periodic graph scans by reusing each mutation's final adjacency to compute a single-node outgoing cut and migrating only nodes with a strict locality benefit.

这四条现在是“可投稿的贡献骨架”，不是最终可提交的实验结论。C3 的静态端到端闭环最完整；C1 新 overlap 和 C2 两阶段协议仍各缺一组决定性的严格因果 A/B；C4 有清晰的局部结构性结果，但在长期/e2e/novelty 补齐前最容易被 reviewer 视为 C2 的 placement heuristic。

## 8. 投稿前实验优先级

| 优先级 | 必须补的实验 | 原因 |
|---|---|---|
| P0 | C1 coupled vs exact-core early issue 的同快照重复 A/B | 目前只有机制覆盖率，没有 overlap 的因果端到端收益 |
| P0 | C2 sync/restart/continuation/fusion/batching 消融，并单测 Stage1 ACK | 目前没有 one-stage baseline，也缺 wire-level RDMA 归因 |
| P0 | C3 fixed/static-only/DynaExtent 的可重置 triplet，并提高动态访问占比 | 当前 48.19% 是目标路径字节收益，动态图占比太低 |
| P0 | C4 relocation on/off 长期更新 A/B 和 migration 成本 | 当前证明了 cut 收益，尚未证明系统收益与低成本 |
| P1 | 至少三个数据集、两个硬件环境、多个更新率/并发度 | 支撑通用性，避免 SIFT100M/A800 特例 |
| P1 | 包含动态 ID 的 ground truth 与 durable-fence 后 Recall | 当前 base-only Recall 不能完整覆盖动态图质量 |
| P1 | 故障、重试、slot reuse、长时间 churn | 四项机制都依赖 incarnation/read-committed 边界 |

推荐的每项贡献证据结构统一为：

1. 一张动机图：原路径中浪费的时间、字节或跨分片边；
2. 一张机制图：只画被松弛的依赖和新增的最小状态；
3. 一张微观结果图：certificate coverage、独立远端访问、bytes/read 或 cross-edge；
4. 一张消融图：逐步加入核心机制；
5. 一张等 Recall 端到端图：QPS、mean/P99、更新吞吐与后台债务；
6. 一张通用性图：数据集、硬件、更新率或分片数。

## 9. 不能在当前论文版本中写出的结论

- 不能把 C1 写成“Beam 中任意节点按任意顺序立即扩展”；当前是乱序完成/验证，随后规范批量打分。
- 不能声称自适应 width 能为任意用户 Beam、数据集和硬件自动找到全局最优宽度；当前只调有界 optional tail，且最新 tail 几乎全部浪费。
- 不能把 Stable-Run 的 +20.14% 当成 PMEFC overlap 的收益。
- 不能声称 C2 的 Stage2 全部是 one-sided RDMA，也不能声称与离线 builder 或全局快照严格等价。
- 不能把 logical score 与独立远程 vector-access item 的 43.11% 差值归因于 home-side scoring 或 batching；其中还混有本地评分和 terminal/deleted 候选等来源。
- 不能把静态 Live-Extent 的 +8.90% QPS 直接写成 DynaExtent 的增益；当前动态部分只证明目标路径 bytes -48.19%。
- 不能把 C4 写成全图重分片、global-cut optimal 或低成本已经被端到端证明；当前只证明固定 outgoing adjacency 下的局部 cut 收益。

## 10. 代码与原始证据索引

以下位置是本次提炼实际核对的实现和原始数据，便于后续写论文时追溯：

- C1 candidate runs、exact preview、early issue 与 authoritative materialization：`src/gpu_search/persistent_kernel/candidate_scoring.cuh`、`src/gpu_search/persistent_kernel/query_traversal.cuh`、`src/gpu_search/adaptive_frontier.hh`；旧 A/B 见 `motivation/results/beam_merge_final/`，当前机制计数见 `experiment/reports/04_gpu_persistent_gpunetio/sift100m_04_gpu_persistent_gpunetio_20260805_093639.json`。
- C2 Stage1 本地搜索/remote frontier、Stage2 continuation 与 batch：`src/memory_node/storage_owner_index/partition_local_search.hh`、`src/memory_node/storage_owner_index/candidate_search.cc`、`src/memory_node/storage_owner_maintenance/worker.cc`、`src/memory_node/peer_rpc/stage1_requests.cc`、`src/service/storage_owner_protocol.hh`。
- C3 extent/tag/checksum/self-heal：`src/vamana/vamana_node.hh`、`src/vamana/dynamic_navigation_code.hh`、`src/memory_node/storage_owner_index/graph_access.cc`、`src/gpu_search/persistent_kernel/rdma_read.cuh`；静态配对汇总见 `motivation/results/live_extent_e2e/current_build_static_c256_live211636_fixed215008_summary.json`，动态数据见 `20260805_093639.json`。
- C4 final-home 选择与 authority relocation：`src/memory_node/storage_owner_index/partition_local_search.hh`、`src/memory_node/storage_owner_maintenance/worker.cc`、`src/memory_node/storage_owner_index/authority_directory_policy.hh`；同次运行的结构性 counterfactual 见 `20260805_093639.json` 的 `stage2` 字段。
