# 第23课：HTTP服务与外部接口

## 学习目标
- 理解HTTP API的设计与实现
- 掌握插入队列和查询队列的服务调度

## 内容大纲

### 1. HTTP API端点 (`http/vamana_service_scheduler.hh`)
```
POST /insert   — 插入向量
    请求: {"id": 123, "vector": [0.1, 0.2, ...]}
    响应: {"success": true, "vectors_inserted": 1000}

POST /search   — 搜索k个最近邻
    请求: {"vector": [0.1, 0.2, ...], "k": 10}
    响应: {"ids": [5, 23, 67, ...], "distances": [0.01, 0.02, ...]}

GET /status    — 获取服务状态
    响应: {"vectors_inserted": 1000, "dimension": 128, "threads": 16}

POST /load_index — 加载索引 (可选)
POST /store_index — 存储索引 (可选)
```

### 2. 服务类型 (`http/service_types.hh`)
```cpp
namespace service {

struct QueryResult {
    vec<std::pair<node_t, distance_t>> results;
};

using InsertQueue = concurrent_queue<InsertItem>;
using QueryQueue = concurrent_queue<QueryItem>;
// or: blocking bounded queue

}
```

### 3. 插入队列调度
```
HTTP handler → InsertQueue.enqueue(item)
     ↓
Worker Threads (insert角色)
     ↓
Vamana::insert() → RDMA/GPU 操作 → Memory Node 存储
     ↓
vectors_inserted_ 原子递增
```

### 4. 查询队列调度
```
HTTP handler → QueryQueue.enqueue(query)
     ↓
Worker Threads (query角色)
     ↓
Vamana::knn() → Beam Search → 返回结果
     ↓
结果存入 thread->query_results[q_id]
```

### 5. 并发控制
- HTTP请求与Worker线程通过concurrent_queue解耦
- 查询结果通过query_id关联（异步响应）
- 可暂停/恢复workers (`pause_workers()` / `resume_workers()`)

### 6. 数据序列化
请求使用nlohmann/json：
```cpp
// 插入请求解析
json j = json::parse(request_body);
node_t id = j["id"];
vec<element_t> vec = j["vector"].get<vec<element_t>>();

// 搜索响应构建
json response;
response["ids"] = ids;
response["distances"] = distances;
```

## 课后任务
1. 实现一个新的HTTP端点（如`POST /batch_insert`）
2. 分析pause/resume机制对inflight请求的影响
3. 添加请求超时处理逻辑

## 参考文件
- `src/http/vamana_service_scheduler.hh`
- `src/http/service_types.hh`
