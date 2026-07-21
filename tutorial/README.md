# DVSTOR 架构阅读入口

旧版逐行教程描述的是已经删除的查询、更新、路由和回收路径，不能再作为当前
实现的说明，因此不再保留。

当前实现只维护一套在线语义：

- GPU 查询从最近物理分片的版本化 centroid live entries 起步；
- Stage1 在 centroid 选择的单一物理 home 完成本地搜索和查询可见反向边；
- Stage2 延续 Stage1 的 beam/visited/frontier，只沿真实图边补充远端搜索；
- 最终 placement 最小化跨分片边，并在反向边协调后更新 centroid membership；
- 动态记录采用 incarnation-tagged read-committed，不使用计算侧更新副本或固定客户端 ACK。

请以以下文档和源码为准：

- [`README.md`](../README.md)：系统契约、部署与实验入口；
- [`docs/gpu_persistent_engine.md`](../docs/gpu_persistent_engine.md)：查询与动态更新语义；
- [`docs/source_layout.md`](../docs/source_layout.md)：源码模块边界；
- [`experiment/README.md`](../experiment/README.md)：schema-16 索引构建和实验流程。

这些文档随当前源码维护；若行为和文档不一致，应修正实现或同步更新上述契约，
不能重新引入已删除的兼容旁路。
