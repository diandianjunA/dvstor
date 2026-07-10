#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "gpu_search/ring.hh"

int main() {
  gpu_search::BoundedMpmcRing<unsigned> ring(1024);
  constexpr unsigned kProducers = 4;
  constexpr unsigned kPerProducer = 10000;
  constexpr unsigned kTotal = kProducers * kPerProducer;
  std::atomic<unsigned> consumed{0};
  std::atomic<unsigned long long> sum{0};

  std::thread consumer([&] {
    unsigned value = 0;
    while (consumed.load(std::memory_order_relaxed) < kTotal) {
      if (!ring.try_pop(value)) {
        std::this_thread::yield();
        continue;
      }
      sum.fetch_add(value, std::memory_order_relaxed);
      consumed.fetch_add(1, std::memory_order_relaxed);
    }
  });

  std::vector<std::thread> producers;
  for (unsigned producer = 0; producer < kProducers; ++producer) {
    producers.emplace_back([&, producer] {
      const unsigned begin = producer * kPerProducer;
      for (unsigned i = 0; i < kPerProducer; ++i) {
        const unsigned value = begin + i + 1;
        while (!ring.try_push(value)) std::this_thread::yield();
      }
    });
  }
  for (auto& producer : producers) producer.join();
  consumer.join();

  const unsigned long long expected =
    static_cast<unsigned long long>(kTotal) * (kTotal + 1) / 2;
  assert(consumed == kTotal);
  assert(sum == expected);
  assert(ring.approximate_size() == 0);
  return 0;
}
