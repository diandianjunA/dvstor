# 第25课：RPC路由与多CN协同

## 学习目标
- 理解Compute Node间的RPC通信
- 掌握Centroid注册与搜索代理
- 理解多CN查询路由的RPC实现

## 内容大纲

### 1. RPC消息类型
```cpp
enum RpcType : u32 {
    rpc_register_centroid = 1,  // CN注册自己的centroid
    rpc_register_ack     = 2,   // Centroid注册确认
    rpc_search_proxy     = 3,   // 搜索代理（MN→CN）
    rpc_search_request   = 4,   // 搜索请求（CN→CN）
    rpc_search_response  = 5,   // 搜索响应
};
```

### 2. RPC初始化
```cpp
void start_rpc() {
    // 1. 每个CN在MN上预投递recv buffer
    post_initial_rpc_receives();
    // 2. 启动RPC线程
    rpc_thread_ = thread(&ComputeService::run_rpc_loop, this);
    // 3. 等待所有CN注册centroid
    refresh_routing_state(true);
}

void post_initial_rpc_receives() {
    // 每个peer CN投递kInitialRpcRecvsPerPeer(8)个recv
    for each peer client:
        for i = 0; i < 8; i++:
            post_rpc_receive(peer_client);
}
```

### 3. Centroid注册
```cpp
void refresh_routing_state(wait_for_remote_registration) {
    // 1. 本地计算K-Means centroid
    auto centroid = compute_local_routing_centroid();
    // 2. 通过MN中继发送到所有CN
    enqueue_rpc(RpcOutbound{target_cn, rpc_register_centroid, centroid});
    // 3. 等待所有远端CN的注册确认
    while (registered_remote_clients_ < num_cns - 1) {
        // rpc_thread_处理ack消息
    }
    // 4. 更新routing_centroids_表
}
```

### 4. K-Means路由放置 (`router/placement.hh`, `router/kmeans.hh`)
```cpp
// 每个CN计算自己数据集的centroid
vec<element_t> compute_local_routing_centroid() {
    // 对本地已读向量做K-Means (k=1 → 单个centroid)
    return average(local_vectors);
}

// 查询时选择最近centroid对应的CN
auto closest_centroids(query) {
    // 返回排序的 (CN_id, distance) 对
    priority_queue<pair<u32, float>> pq;
    for each remote_centroid:
        pq.push({cn_id, l2(query, centroid)});
    return pq;
}
```

### 5. 搜索代理（Search Proxy）
当CN A收到路由到CN B的查询时：
```
1. CN A → MN: RPC消息 (destination=B, type=search_proxy)
2. MN → CN B: 转发消息
3. CN B: 本地处理查询
4. CN B → CN A: RPC search_response (包含结果)
5. CN A: resolve对应的promise
```

### 6. RPC线程主循环
```cpp
void run_rpc_loop() {
    while (!rpc_shutdown_) {
        // 1. 轮询所有peer QP的recv CQ
        for each peer:
            poll recv WC
            if received: handle_rpc_receive(header, payload)
        // 2. 刷新待发送RPC
        flush_outbound_rpc();
    }
}

void handle_rpc_receive(header, payload) {
    switch (header.type) {
    case rpc_register_centroid: handle_register_centroid(...);
    case rpc_register_ack:     handle_register_ack(...);
    case rpc_search_proxy:     handle_search_proxy(...);  // 仅MN
    case rpc_search_request:   handle_search_request(...);
    case rpc_search_response:  handle_search_response(...);
    }
}
```

### 7. RPC缓冲区管理
```cpp
// 固定大小的RPC缓冲区，freelist管理
unique_ptr<byte_t[]> rpc_buffer_;
unique_ptr<LocalMemoryRegion> rpc_region_;
vec<idx_t> rpc_freelist_;

// 所有peer共享同一缓冲区（通过freelist分配/回收）
```

## 课后任务
1. 画图展示3个CN的场景下centroid注册的完整消息流
2. 分析：如果某个CN在注册centroid时崩溃会怎样？
3. 对比RPC路由和QueryRouter的异同

## 参考文件
- `src/service/compute_service/rpc_routing.ipp`
- `src/service/compute_service.hh`（RPC相关成员）
- `src/router/placement.hh`
- `src/router/kmeans.hh`
