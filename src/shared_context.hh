#pragma once

#include <library/connection_manager.hh>
#include <library/detached_qp.hh>

template <typename T>
class SharedContext {
public:
  SharedContext(Context& channel_context,
                ClientConnectionManager& cm,
                HugePage<byte_t>& buffer,
                const MemoryRegionTokens& remote_mrts,
                u32 qp_pool_size = 1)
      : context(channel_context.get_config()), remote_mrts(remote_mrts) {
    const u32 num_nodes = cm.server_qps.size();
    qps.resize(num_nodes);
    for (u32 node = 0; node < num_nodes; ++node) {
      auto& per_node = qps[node];
      per_node.reserve(qp_pool_size);
      for (u32 p = 0; p < qp_pool_size; ++p) {
        auto& dqp = per_node.emplace_back(
            std::make_unique<DetachedQP>(context, context.get_send_cq(), context.get_receive_cq()));
        dqp->connect(channel_context, context.get_lid(), cm.server_qps[node]);
      }
    }
    memory_region = std::make_unique<LocalMemoryRegion>(context, buffer.get_full_buffer(), buffer.buffer_size);
  }

  void register_thread(T* thread) {
    registered_threads.push_back(thread);
    thread->ctx = this;
    thread->ctx_tid = registered_threads.size() - 1;
  }

  ibv_cq* get_cq() { return context.get_send_cq(); }
  u32 get_lkey() const { return memory_region->get_lkey(); }
  MemoryRegionToken* get_remote_mrt(u32 memory_node) { return remote_mrts[memory_node].get(); }

public:
  Context context;
  vec<vec<u_ptr<DetachedQP>>> qps;  // [memory_node][pool_index]
  vec<T*> registered_threads;

private:
  u_ptr<LocalMemoryRegion> memory_region;
  const MemoryRegionTokens& remote_mrts;
};
