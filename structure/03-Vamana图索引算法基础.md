# 第3课：Vamana图索引算法基础

## 学习目标
- 理解Vamana图索引的算法原理
- 掌握Beam Search（束搜索）与RobustPrune（鲁棒剪枝）
- 对比Vamana与HNSW的差异

## 内容大纲

### 1. Vamana算法概述
Vamana（源自DiskANN论文）是单层有向图近似最近邻搜索算法：
- **图结构**: 每个节点有固定最大出度R，形成有向图
- **搜索**: Beam Search（维护大小为beam_width的候选集，每次扩展最佳未扩展节点）
- **构建**: 增量插入 + RobustPrune剪枝保持图多样性
- **关键区别**: 单层图 vs HNSW的多层层次图

### 2. Beam Search算法详解
```
输入: 查询向量q, 入口节点(medoid), beam_width L
输出: top-k个最近邻

1. beam = [(medoid_ptr, dist(q, medoid), expanded=false)]
2. visited = {medoid_ptr}
3. while 存在未扩展节点:
     a. 选择beam中dist最小且未扩展的节点p
     b. 标记p为已扩展
     c. 读取p的所有邻居（通过RDMA）
     d. 对每个未访问邻居:
        - 加入visited
        - 批量读取邻居向量（通过RDMA）
        - GPU计算距离
        - 插入beam（保持按距离排序，最多L个）
4. 返回beam中前k个
```

### 3. RobustPrune算法（α=1.2）
```
输入: 源节点p, 候选邻居集V（已按距离排序）, α, R
输出: 最多R个邻居

1. selected = []
2. for v in V (按距离升序):
     keep = true
     for s in selected:
         if α * dist(p, v) > dist(s, v):
             keep = false; break
     if keep:
         selected.append(v)
         if |selected| == R: break
3. return selected
```
**直觉**: 如果已选邻居s比候选v更接近p，且α*dist(p,v) > dist(s,v)，说明v被s"覆盖"，无需保留v。

### 4. 与HNSW的对比
| 特性 | HNSW | Vamana |
|------|------|--------|
| 图层结构 | 多层（底层密集，上层稀疏） | 单层 |
| 搜索入口 | 顶层entry point逐层下降 | 单入口(medoid)直接搜索 |
| 邻居选择 | HNSW启发式（基于距离） | RobustPrune（基于多样性） |
| 参数 | M, ef_construction, ef_search | R, beam_width, α |
| 优势 | 对数复杂度，适合内存场景 | 更少的内存跳数，适合远程内存 |

### 5. 在DVSTOR中Vamana的优势
- **更少的RDMA往返**: 单层图减少了跨层读取邻居的次数
- **可预测的节点大小**: 固定布局（header+vector+R*8B neighbors）
- **GPU友好**: 批量向量距离计算适合GPU并行化

## 课后任务
1. 手写Beam Search伪代码，标明每一步的RDMA操作
2. 用一个小例子（5个2D点）模拟RobustPrune的执行过程
3. 思考：α=1.0和α=2.0分别对图结构有什么影响？

## 参考文件
- `src/vamana/vamana.hh`
- `src/vamana/vamana_search.ipp`
- `src/vamana/vamana_insert.ipp`
- `src/vamana/vamana_helpers.ipp`
