#pragma once

#include <cerrno>

#include <infiniband/verbs.h>
#include <library/types.hh>

namespace rdma {

struct SendChainResult {
  bool success{};
  u32 post_calls{};
  u32 retries{};
  int error{};
};

template <typename PostFn, typename PollFn>
SendChainResult post_send_chain_with_retry(ibv_send_wr* first,
                                           PostFn&& post,
                                           PollFn&& poll) {
  SendChainResult result;
  ibv_send_wr* first_unposted = first;
  for (;;) {
    ibv_send_wr* bad = nullptr;
    ++result.post_calls;
    const int rc = post(first_unposted, &bad);
    if (rc == 0) {
      result.success = true;
      return result;
    }

    if ((rc != ENOMEM && rc != EAGAIN && rc != EBUSY) || bad == nullptr) {
      result.error = rc;
      return result;
    }
    ++result.retries;
    first_unposted = bad;
    poll();
  }
}

}  // namespace rdma
