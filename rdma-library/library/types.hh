#ifndef RDMA_LIBRARY_TYPES_HH
#define RDMA_LIBRARY_TYPES_HH

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "extern/concurrentqueue.hh"

using i8 = int8_t;
using u8 = uint8_t;

using i16 = int16_t;
using u16 = uint16_t;

using i32 = int32_t;
using u32 = uint32_t;

using i64 = int64_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

using byte_t = uint8_t;
using str = std::string;
using size_t = std::size_t;
using idx_t = std::size_t;

using intptr_t = std::intptr_t;

template <typename T>
using u_ptr = std::unique_ptr<T>;

template <typename T>
using s_ptr = std::shared_ptr<T>;

template <typename T>
using func = std::function<T>;

template <typename T>
using vec = std::vector<T>;

template <typename T>
class span {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using pointer = T*;
  using reference = T&;
  using iterator = pointer;

  constexpr span() noexcept = default;
  constexpr span(pointer data, size_t size) noexcept : data_(data), size_(size) {}

  template <typename Allocator,
            typename U = T,
            std::enable_if_t<std::is_convertible_v<value_type*, U*>, int> = 0>
  span(std::vector<value_type, Allocator>& values) noexcept : data_(values.data()), size_(values.size()) {}

  template <typename Allocator,
            typename U = T,
            std::enable_if_t<std::is_convertible_v<const value_type*, U*>, int> = 0>
  span(const std::vector<value_type, Allocator>& values) noexcept : data_(values.data()), size_(values.size()) {}

  template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
  constexpr span(const span<U>& other) noexcept : data_(other.data()), size_(other.size()) {}

  constexpr pointer data() const noexcept { return data_; }
  constexpr size_t size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size_; }
  constexpr reference operator[](size_t index) const noexcept { return data_[index]; }

private:
  pointer data_{};
  size_t size_{};
};

template <typename T>
using concurrent_queue = moodycamel::ConcurrentQueue<T>;

#endif  // RDMA_LIBRARY_TYPES_HH
