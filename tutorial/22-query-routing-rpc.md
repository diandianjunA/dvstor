# 第 22 课：查询服务路径、RPC routing 与结果汇总

## 本课目标

本课学习 compute node 之间的查询 routing。学完后，你需要理解：

1. 本地查询、initiator routing、proxy search、remote search response 的区别。
2. `choose_destination()` 如何用 centroid 和 inflight 做简单负载分配。
3. RPC message 如何通过 RDMA SEND/RECV 携带 query 和 result。
4. 这套路由机制对性能、正确性和未来重构的限制。

代码入口：

- `src/service/compute_service/search.ipp`
- `src/service/compute_service/rpc_routing.ipp`
- `src/router/query_router.hh`
- `src/router/placement.hh`
- `src/router/kmeans.hh`
- `src/router/message_wrapper.hh`

本课重点以 `ComputeService` 的 `search.ipp` 和 `rpc_routing.ipp` 为准。`src/router/` 下还有更独立的 routing 组件，但当前 `ComputeService` 内部已经实现了一套轻量 RPC routing，所以学习时要区分“已有通用 router 代码”和“当前服务实际走的代码”。

## 1. 查询入口

`ComputeService` 暴露两个查询接口：

```cpp
vec<node_t> search(const vec<element_t>& query, u32 k);
vec<node_t> search_raw(VectorDType query_dtype, const byte_t* query_data, u32 dim, u32 k);
```

`search_raw()` 会先校验：

- `dim == config_.dim`
- `query_data != nullptr`

如果 routing enabled：

- raw query 会先 decode 成 float。
- 然后调用 `search(decoded, k)`。

如果 routing disabled：

- 直接调用 `search_local_raw(...)`，保留 query dtype，进入本地 queue。

这个分支说明：routing RPC 目前只传 float payload，不直接传 raw dtype query。对于 float16/int8 等 query，如果启用 routing，会发生一次 decode 到 float 的转换。

## 2. routing_enabled 的条件

`routing_enabled()` 的代码很简单：

```cpp
return config_.routing && cm_.num_total_clients > 1;
```

也就是说，只有同时满足：

1. 配置开启 routing。
2. compute client 总数超过 1。

才会走 compute-node 间 RPC。

如果只有一个 compute node，即使配置打开 routing，也会本地执行。

## 3. 本地查询路径

本地查询入口是：

- `search_local_result(...)`
- `search_local_raw_result(...)`
- `search_local(...)`
- `search_local_raw(...)`

`search_local_result()` 做的事：

1. 校验 query dim。
2. 创建 `service::breakdown::Sample`。
3. 记录 `enqueued_at`。
4. new 一个 `service::QueryRequest`。
5. 填入：
   - `components`
   - `entry_points`
   - `query_dtype`
   - `k`
   - `enqueued_at`
   - `breakdown_sample`
6. 从 `request->result` 获取 future。
7. `query_queue_.enqueue(request)`。
8. 等待 future。
9. 删除 request。
10. 返回 `QueryResult` 和 sample。

如果 storage-owner update mode 是 `local_stitch`，`entry_points` 不是空：

- 先用 anchor index 找最近 shard。
- 再从每个 shard 取最近 anchors。
- 去重后传给 Vamana search。

否则 entry points 为空，Vamana search 会从 medoid 开始。

## 4. search() 的决策树

`search(const vec<element_t>& query, u32 k)` 的行为可以整理成决策树：

1. 如果 `!routing_enabled()`：
   - 直接 `search_local(query, k)`。

2. 如果当前是 initiator：
   - 调用 `choose_destination(query)`。
   - 如果 destination 是自己：
     - 直接 `search_local(query, k)`。
   - 否则继续构造 RPC。

3. 创建 request id：
   - `next_request_id_.fetch_add(1)`

4. 创建 promise/future，并写入 `pending_queries_`。

5. 创建 `RpcOutbound`：
   - `request_id`
   - `top_k = min(k, kMaxRpcResults)`
   - `float_payload = query`

6. 如果当前是 initiator：
   - 再次调用 `choose_destination(query)`。
   - 如果此时 destination 变成自己：
     - 删除 pending promise。
     - 删除 outbound。
     - 本地搜索。
   - 否则：
     - `type = rpc_search_request`
     - `destination_client = destination`
     - `origin_client = cm_.client_id`
     - 增加 `routing_inflight_[destination]`

7. 如果当前不是 initiator：
   - 发送给 initiator。
   - `type = rpc_search_proxy`
   - `destination_client = 0`
   - `origin_client = cm_.client_id`

8. `enqueue_rpc(outbound)`。
9. 等待 future。

这里有一个细节：initiator 在函数开头和构造 outbound 后各调用了一次 `choose_destination()`。因为中间 routing inflight 可能变化，所以两次结果可能不一致。代码处理了第二次变成本地的情况。

## 5. RPC header 与 payload

`ComputeService` 内部定义：

```cpp
struct RpcHeader {
  u32 magic{};
  u32 type{};
  u32 source_client{};
  u32 origin_client{};
  u64 request_id{};
  u32 top_k{};
  u32 payload_count{};
};
```

RPC 类型：

- `rpc_register_centroid`
- `rpc_register_ack`
- `rpc_search_proxy`
- `rpc_search_request`
- `rpc_search_response`

`rpc_message_size()` 计算固定 slot 大小：

```cpp
payload_bytes = max(config_.dim * sizeof(element_t),
                    max(config_.k, kMaxRpcResults) * sizeof(node_t));
return sizeof(RpcHeader) + payload_bytes;
```

这说明 RPC buffer 是固定大小 slot：

- query payload 按 `float` 向量大小计算。
- response payload 按 result id 数计算。
- 如果 `k` 很大，response slot 变大。
- 如果 `dim` 很大，query slot 变大。

固定 slot 设计简化了 receive buffer 管理，但也会浪费内存。

## 6. RPC buffer 和 freelist

`start_rpc()` 只在 routing enabled 时运行。它会：

1. 计算 peer count：
   - initiator：`cm_.client_qps.size()`
   - non-initiator：1
2. 计算 buffer entries：
   - `max(16, peer_count * (kInitialRpcRecvsPerPeer + 8))`
3. 分配 `rpc_buffer_`。
4. 注册 `rpc_region_`。
5. 初始化 `rpc_freelist_`，每个元素是 slot offset。
6. 启动 `rpc_thread_` 运行 `run_rpc_loop()`。

发送和接收共用同一块 `rpc_buffer_` 和同一个 freelist。发送完成 CQE 会把 slot offset 放回 freelist；接收处理完也会把 offset 放回 freelist。

这种做法的好处：

- 简化内存注册。
- 避免频繁分配 RPC message。
- WR id 可以编码 peer 和 offset。

风险：

- freelist 是 rpc loop 单线程使用的假设较强。
- slot 数不足时，`flush_outbound_rpc()` 会轮询 send CQ 等待释放。
- 如果 send completion 不回来，outbound flush 会卡住。

## 7. RPC loop

`run_rpc_loop()` 的主循环：

1. `post_initial_rpc_receives()`。
2. 循环直到 `rpc_shutdown_`。
3. 如果 `rpc_paused_`：
   - 设置 `rpc_idle_ = true`
   - `yield`
   - continue
4. 设置 `rpc_idle_ = false`。
5. `flush_outbound_rpc()`。
6. poll receive CQ。
7. 对每个 receive：
   - decode wr id 得到 peer client 和 offset。
   - 从 buffer 取 header 和 payload。
   - `handle_rpc_receive(...)`
   - 归还 offset。
   - 重新 post receive。
8. poll send CQ，回收 send slot。
9. 如果没有 receive，`yield`。

这个 loop 同时承担：

- outbound queue drain。
- send completion progress。
- receive completion progress。
- RPC dispatch。

它没有 sleep 或 event wait，因此低负载下会有 CPU busy-yield 行为。

## 8. centroid 注册流程

routing 初始化时调用 `refresh_routing_state(true)`。

如果当前是 initiator：

- 如果要求等待远端注册，则等待：

```cpp
registered_remote_clients_ >= cm_.num_total_clients - 1
```

如果当前不是 initiator：

1. 创建 request id。
2. 创建 promise/future。
3. 写入 `pending_registration_acks_`。
4. 创建 outbound：
   - `destination_client = 0`
   - `type = rpc_register_centroid`
   - `origin_client = cm_.client_id`
   - payload 是自己的 centroid。
5. 如果要求等待，则 `future.get()`。

initiator 收到 `rpc_register_centroid` 后：

1. 从 payload 复制 centroid。
2. 写入 `routing_centroids_[source_client]`。
3. 如果是第一次注册，递增 `registered_remote_clients_` 并 notify。
4. 给 source client 发送 `rpc_register_ack`。

non-initiator 收到 ack 后：

- 从 `pending_registration_acks_` 找 promise。
- erase。
- set value。

centroid 的来源是 `compute_local_routing_centroid()`，它读取本地 medoid 节点的 vector。严格来说，这不是 shard 的真实 centroid，而是 medoid vector。命名上叫 centroid，但实现上是 medoid probe。

## 9. choose_destination

`choose_destination()` 只在 initiator 上做实际选择：

1. 如果 routing disabled 或当前不是 initiator：
   - 返回自己的 client id。

2. 遍历 `routing_centroids_`。

3. 跳过空 centroid。

4. 计算：

```cpp
distance = Distance::dist(query, routing_centroids_[client], config_.dim);
load_penalty = 1.0f + 0.2f * routing_inflight_[client];
score = distance * load_penalty;
```

5. 选择 score 最小的 client。

这是一种很轻量的 routing 策略：

- 数据相似性由 query 到 centroid 的距离估计。
- 负载由 inflight 请求数线性惩罚。
- 惩罚系数固定为 0.2。

性能和正确性上的问题：

1. 对 L2，距离越小越好，score 合理。
2. 对 IPDistance，`Distance::dist` 的语义要回到 `common/distance.hh` 检查，不能假设它就是越大越好。
3. inflight 只在 initiator 的 routing 路径中维护。
4. non-initiator 不能直接选择其他 non-initiator，它必须先 proxy 给 initiator。
5. centroid 只在 startup/load 后刷新，不反映在线插入后的数据分布变化。

## 10. proxy search 与 request forwarding

如果 non-initiator 调用 `search()`：

- 它向 initiator 发送 `rpc_search_proxy`。

initiator 收到 `rpc_search_proxy` 后：

1. 从 payload 复制 query。
2. `choose_destination(query)`。
3. 如果 destination 是 initiator 自己：
   - 本地 search。
   - 发送 `rpc_search_response` 给 source client。
4. 否则：
   - 增加 destination inflight。
   - 发送 `rpc_search_request` 给 destination。
   - `origin_client = header.source_client`。

remote compute node 收到 `rpc_search_request` 后：

1. 本地 search。
2. 发送 `rpc_search_response`：
   - 如果自己是 initiator，发送给 `origin_client`。
   - 否则发送给 initiator。

initiator 收到 response 时，如果发现：

```cpp
cm_.is_initiator &&
header.source_client != cm_.client_id &&
header.origin_client != cm_.client_id
```

说明这是转发链路上的远端 response，需要：

1. 减少 `routing_inflight_[source_client]`。
2. 再转发给 origin client。

如果 response 是发给自己的 pending query，则：

1. 减少 inflight。
2. 从 `pending_queries_` 找 promise。
3. set result。

## 11. 结果汇总的限制

当前 RPC search response 只携带 `node_t` id payload，没有携带 distance。`search()` 返回的 public 类型也是 `vec<node_t>`。

因此 routing 模式不是“多个 compute node 都查一遍再 merge top-k”。它是：

- 选择一个 destination。
- destination 单点执行 local search。
- 返回 id list。

这对扩展性和 recall 有重要影响：

1. 如果 partition/routing 不准确，query 可能被送到不包含最佳近邻的 shard。
2. 没有跨 shard result merge。
3. `kMaxRpcResults` 限制 response 上限。
4. response 中没有 distance，后续想 merge 多个 destination 时需要改协议。

如果未来要做多 shard fanout search，需要至少修改：

- `RpcHeader`
- `RpcOutbound`
- response payload 编码
- pending query 聚合结构
- `search()` 返回前的 merge 逻辑
- result distance 的保留方式

## 12. 性能影响

Routing 影响性能的路径包括：

1. non-initiator 的额外 hop：
   - non-initiator -> initiator
   - initiator -> destination
   - destination -> initiator
   - initiator -> origin

2. initiator 成为控制面瓶颈：
   - 所有 non-initiator 的 proxy search 都经 initiator 决策。
   - centroid 注册也经 initiator。

3. fixed slot buffer：
   - 高维 query 会放大 RPC buffer。
   - 大 `k` 会放大 response slot。

4. busy-yield RPC loop：
   - 低负载下 CPU 消耗。
   - 高负载下可能抢占 worker CPU。

5. payload copy：
   - outbound 将 query 复制到 `float_payload`。
   - flush 时再 memcpy 到 registered buffer。
   - receive 时再 memcpy 到本地 vec。

6. routing inflight 粒度粗：
   - 只按 client 计数。
   - 不区分 query cost、k、beam width、实际运行时间。

## 13. 设计异味

1. RPC routing 写在 `ComputeService` 模板类内部：
   - 增加编译压力。
   - 很难单独测试。

2. protocol 没有独立 codec：
   - header/payload 直接 memcpy。
   - message size 与 config 绑定。

3. routing 策略硬编码：
   - load penalty 系数固定。
   - centroid 刷新策略固定。
   - 没有抽象出 policy。

4. response 不携带 distance：
   - 限制未来 merge。

5. routing state 与 RPC progress 耦合：
   - `routing_inflight_` 在 handler 中增减。
   - pending promise 也由 RPC handler 管理。

6. `src/router/` 目录与 `ComputeService` 内置 routing 并存：
   - 需要确认哪些代码实际被使用。
   - 避免重构时误改未接入路径。

## 14. 可验证问题

1. 单 compute node：
   - `routing_enabled()` 应返回 false。
   - `search()` 直接走 local。

2. initiator destination 为自己：
   - 不应创建 pending query。
   - 不应进入 outbound RPC。

3. non-initiator search：
   - 应发送 `rpc_search_proxy` 给 initiator。
   - 最终 promise 应由 response 设置。

4. response 转发：
   - origin 不是 initiator 且 source 是另一个 remote client 时，initiator 应转发。

5. inflight 计数：
   - destination request 发出时 +1。
   - response 回来时 -1。
   - 异常或丢 response 时会不会泄漏。

6. k 限制：
   - `top_k = min(k, kMaxRpcResults)` 后，结果数是否小于调用方期望。

7. raw query routing：
   - 非 float dtype query 开启 routing 后是否 decode 成 float。
   - recall/latency 是否变化。

## 15. 学习任务

1. 画一张 query routing 决策树，从 `search()` 开始，到 local/proxy/request/response。
2. 画一张 RPC slot 生命周期图，标出 freelist、send CQ、recv CQ 如何回收 offset。
3. 找出 `kMaxRpcResults` 的定义，分析它对大 k 查询的影响。
4. 设计一个测试：两个 compute client，强行让 initiator 选择 remote destination，验证 response 能回到 origin。
5. 设计一个优化实验：调整 load penalty 公式，观测 tail latency、routing distribution、recall。

