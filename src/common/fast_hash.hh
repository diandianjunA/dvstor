#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace fast_hash {

namespace detail {

inline size_t next_power_of_two(size_t value) {
  size_t capacity = 1;
  while (capacity < value) capacity <<= 1;
  return capacity;
}

inline size_t mix_hash(size_t value) {
  uint64_t x = static_cast<uint64_t>(value);
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<size_t>(x);
}

}  // namespace detail

template <typename Key, typename Hash = std::hash<Key>, typename Eq = std::equal_to<Key>>
class FlatHashSet {
private:
  enum class State : uint8_t { empty = 0, full = 1, deleted = 2 };

  struct Slot {
    State state{State::empty};
    Key key{};
  };

public:
  class iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = Key;
    using pointer = Key*;
    using reference = Key&;
    using iterator_category = std::forward_iterator_tag;

    iterator() = default;
    iterator(Slot* slots, size_t capacity, size_t index)
        : slots_(slots), capacity_(capacity), index_(index) {
      skip_empty();
    }

    reference operator*() const { return slots_[index_].key; }
    pointer operator->() const { return &slots_[index_].key; }
    iterator& operator++() {
      ++index_;
      skip_empty();
      return *this;
    }
    bool operator==(const iterator& other) const {
      return slots_ == other.slots_ && index_ == other.index_;
    }
    bool operator!=(const iterator& other) const { return !(*this == other); }

  private:
    friend class FlatHashSet;

    void skip_empty() {
      while (slots_ != nullptr && index_ < capacity_ &&
             slots_[index_].state != State::full) {
        ++index_;
      }
    }

    Slot* slots_{nullptr};
    size_t capacity_{0};
    size_t index_{0};
  };

  class const_iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = Key;
    using pointer = const Key*;
    using reference = const Key&;
    using iterator_category = std::forward_iterator_tag;

    const_iterator() = default;
    const_iterator(const Slot* slots, size_t capacity, size_t index)
        : slots_(slots), capacity_(capacity), index_(index) {
      skip_empty();
    }

    reference operator*() const { return slots_[index_].key; }
    pointer operator->() const { return &slots_[index_].key; }
    const_iterator& operator++() {
      ++index_;
      skip_empty();
      return *this;
    }
    bool operator==(const const_iterator& other) const {
      return slots_ == other.slots_ && index_ == other.index_;
    }
    bool operator!=(const const_iterator& other) const { return !(*this == other); }

  private:
    void skip_empty() {
      while (slots_ != nullptr && index_ < capacity_ &&
             slots_[index_].state != State::full) {
        ++index_;
      }
    }

    const Slot* slots_{nullptr};
    size_t capacity_{0};
    size_t index_{0};
  };

  FlatHashSet() { rehash(kInitialCapacity); }

  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] size_t size() const { return size_; }
  [[nodiscard]] size_t capacity() const { return slots_.size(); }

  iterator begin() { return iterator(slots_.data(), slots_.size(), 0); }
  iterator end() { return iterator(slots_.data(), slots_.size(), slots_.size()); }
  const_iterator begin() const {
    return const_iterator(slots_.data(), slots_.size(), 0);
  }
  const_iterator end() const {
    return const_iterator(slots_.data(), slots_.size(), slots_.size());
  }

  void clear() {
    for (auto& slot : slots_) slot.state = State::empty;
    size_ = 0;
    deleted_ = 0;
  }

  void reserve(size_t expected_size) {
    const size_t needed = capacity_for(expected_size);
    if (needed > slots_.size()) rehash(needed);
  }

  bool contains(const Key& key) const { return find_index(key) != npos; }

  iterator find(const Key& key) {
    const size_t index = find_index(key);
    return index == npos ? end() : iterator(slots_.data(), slots_.size(), index);
  }

  const_iterator find(const Key& key) const {
    const size_t index = find_index(key);
    return index == npos ? end()
                         : const_iterator(slots_.data(), slots_.size(), index);
  }

  std::pair<iterator, bool> insert(const Key& key) {
    ensure_insert_capacity();
    const auto [index, inserted] = insert_no_grow(key);
    return {iterator(slots_.data(), slots_.size(), index), inserted};
  }

  std::pair<iterator, bool> insert(Key&& key) {
    ensure_insert_capacity();
    const auto [index, inserted] = insert_no_grow(std::move(key));
    return {iterator(slots_.data(), slots_.size(), index), inserted};
  }

  iterator erase(iterator it) {
    if (it.slots_ != slots_.data() || it.index_ >= slots_.size() ||
        slots_[it.index_].state != State::full) {
      return end();
    }
    slots_[it.index_].kv = {};
    slots_[it.index_].state = State::deleted;
    --size_;
    ++deleted_;
    iterator next(slots_.data(), slots_.size(), it.index_ + 1);
    maybe_cleanup_deleted();
    return next;
  }

private:
  static constexpr size_t kInitialCapacity = 8;
  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  static size_t capacity_for(size_t expected_size) {
    const size_t min_capacity = std::max<size_t>(kInitialCapacity,
        (expected_size * 10 + 6) / 7);
    return detail::next_power_of_two(min_capacity);
  }

  size_t bucket(const Key& key) const {
    return detail::mix_hash(hash_(key)) & (slots_.size() - 1);
  }

  size_t find_index(const Key& key) const {
    if (slots_.empty()) return npos;
    const size_t mask = slots_.size() - 1;
    size_t index = bucket(key);
    for (;;) {
      const Slot& slot = slots_[index];
      if (slot.state == State::empty) return npos;
      if (slot.state == State::full && eq_(slot.key, key)) return index;
      index = (index + 1) & mask;
    }
  }

  void ensure_insert_capacity() {
    if ((size_ + deleted_ + 1) * 10 >= slots_.size() * 7) {
      rehash(slots_.size() * 2);
    }
  }

  void maybe_cleanup_deleted() {
    if (deleted_ > slots_.size() / 4) {
      rehash(slots_.size());
    }
  }

  template <typename K>
  std::pair<size_t, bool> insert_no_grow(K&& key) {
    const size_t mask = slots_.size() - 1;
    size_t index = bucket(key);
    size_t first_deleted = npos;
    for (;;) {
      Slot& slot = slots_[index];
      if (slot.state == State::full) {
        if (eq_(slot.key, key)) return {index, false};
      } else if (slot.state == State::deleted) {
        if (first_deleted == npos) first_deleted = index;
      } else {
        const size_t target = first_deleted == npos ? index : first_deleted;
        Slot& target_slot = slots_[target];
        target_slot.key = std::forward<K>(key);
        if (target_slot.state == State::deleted) --deleted_;
        target_slot.state = State::full;
        ++size_;
        return {target, true};
      }
      index = (index + 1) & mask;
    }
  }

  void rehash(size_t new_capacity) {
    new_capacity = detail::next_power_of_two(std::max(new_capacity, kInitialCapacity));
    std::vector<Slot> old_slots = std::move(slots_);
    slots_.assign(new_capacity, Slot{});
    size_ = 0;
    deleted_ = 0;
    for (auto& slot : old_slots) {
      if (slot.state == State::full) {
        (void)insert_no_grow(std::move(slot.key));
      }
    }
  }

  std::vector<Slot> slots_;
  size_t size_{0};
  size_t deleted_{0};
  Hash hash_{};
  Eq eq_{};
};

template <typename Key,
          typename T,
          typename Hash = std::hash<Key>,
          typename Eq = std::equal_to<Key>>
class FlatHashMap {
private:
  enum class State : uint8_t { empty = 0, full = 1, deleted = 2 };

  struct Slot {
    State state{State::empty};
    std::pair<Key, T> kv{};
  };

public:
  class iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = std::pair<Key, T>;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    iterator() = default;
    iterator(Slot* slots, size_t capacity, size_t index)
        : slots_(slots), capacity_(capacity), index_(index) {
      skip_empty();
    }

    reference operator*() const { return slots_[index_].kv; }
    pointer operator->() const { return &slots_[index_].kv; }
    iterator& operator++() {
      ++index_;
      skip_empty();
      return *this;
    }
    bool operator==(const iterator& other) const {
      return slots_ == other.slots_ && index_ == other.index_;
    }
    bool operator!=(const iterator& other) const { return !(*this == other); }

  private:
    friend class FlatHashMap;

    void skip_empty() {
      while (slots_ != nullptr && index_ < capacity_ &&
             slots_[index_].state != State::full) {
        ++index_;
      }
    }

    Slot* slots_{nullptr};
    size_t capacity_{0};
    size_t index_{0};
  };

  class const_iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = const std::pair<Key, T>;
    using pointer = const std::pair<Key, T>*;
    using reference = const std::pair<Key, T>&;
    using iterator_category = std::forward_iterator_tag;

    const_iterator() = default;
    const_iterator(const Slot* slots, size_t capacity, size_t index)
        : slots_(slots), capacity_(capacity), index_(index) {
      skip_empty();
    }

    reference operator*() const { return slots_[index_].kv; }
    pointer operator->() const { return &slots_[index_].kv; }
    const_iterator& operator++() {
      ++index_;
      skip_empty();
      return *this;
    }
    bool operator==(const const_iterator& other) const {
      return slots_ == other.slots_ && index_ == other.index_;
    }
    bool operator!=(const const_iterator& other) const { return !(*this == other); }

  private:
    void skip_empty() {
      while (slots_ != nullptr && index_ < capacity_ &&
             slots_[index_].state != State::full) {
        ++index_;
      }
    }

    const Slot* slots_{nullptr};
    size_t capacity_{0};
    size_t index_{0};
  };

  FlatHashMap() { rehash(kInitialCapacity); }

  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] size_t size() const { return size_; }
  [[nodiscard]] size_t capacity() const { return slots_.size(); }

  iterator begin() { return iterator(slots_.data(), slots_.size(), 0); }
  iterator end() { return iterator(slots_.data(), slots_.size(), slots_.size()); }
  const_iterator begin() const {
    return const_iterator(slots_.data(), slots_.size(), 0);
  }
  const_iterator end() const {
    return const_iterator(slots_.data(), slots_.size(), slots_.size());
  }

  void clear() {
    for (auto& slot : slots_) {
      if (slot.state != State::empty) {
        slot.kv = {};
        slot.state = State::empty;
      }
    }
    size_ = 0;
    deleted_ = 0;
  }

  void reserve(size_t expected_size) {
    const size_t needed = capacity_for(expected_size);
    if (needed > slots_.size()) rehash(needed);
  }

  bool contains(const Key& key) const { return find_index(key) != npos; }

  iterator find(const Key& key) {
    const size_t index = find_index(key);
    return index == npos ? end() : iterator(slots_.data(), slots_.size(), index);
  }

  const_iterator find(const Key& key) const {
    const size_t index = find_index(key);
    return index == npos ? end()
                         : const_iterator(slots_.data(), slots_.size(), index);
  }

  std::pair<iterator, bool> insert(const std::pair<Key, T>& value) {
    ensure_insert_capacity();
    const auto [index, inserted] = insert_no_grow(value.first, value.second);
    return {iterator(slots_.data(), slots_.size(), index), inserted};
  }

  std::pair<iterator, bool> insert(std::pair<Key, T>&& value) {
    ensure_insert_capacity();
    const auto [index, inserted] =
        insert_no_grow(std::move(value.first), std::move(value.second));
    return {iterator(slots_.data(), slots_.size(), index), inserted};
  }

  T& operator[](const Key& key) {
    ensure_insert_capacity();
    const auto [index, inserted] = insert_key_no_grow(key);
    (void)inserted;
    return slots_[index].kv.second;
  }

  T& operator[](Key&& key) {
    ensure_insert_capacity();
    const auto [index, inserted] = insert_key_no_grow(std::move(key));
    (void)inserted;
    return slots_[index].kv.second;
  }

  iterator erase(iterator it) {
    if (it.slots_ != slots_.data() || it.index_ >= slots_.size() ||
        slots_[it.index_].state != State::full) {
      return end();
    }
    slots_[it.index_].state = State::deleted;
    --size_;
    ++deleted_;
    iterator next(slots_.data(), slots_.size(), it.index_ + 1);
    maybe_cleanup_deleted();
    return next;
  }

private:
  static constexpr size_t kInitialCapacity = 8;
  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  static size_t capacity_for(size_t expected_size) {
    const size_t min_capacity = std::max<size_t>(kInitialCapacity,
        (expected_size * 10 + 6) / 7);
    return detail::next_power_of_two(min_capacity);
  }

  size_t bucket(const Key& key) const {
    return detail::mix_hash(hash_(key)) & (slots_.size() - 1);
  }

  size_t find_index(const Key& key) const {
    if (slots_.empty()) return npos;
    const size_t mask = slots_.size() - 1;
    size_t index = bucket(key);
    for (;;) {
      const Slot& slot = slots_[index];
      if (slot.state == State::empty) return npos;
      if (slot.state == State::full && eq_(slot.kv.first, key)) return index;
      index = (index + 1) & mask;
    }
  }

  void ensure_insert_capacity() {
    if ((size_ + deleted_ + 1) * 10 >= slots_.size() * 7) {
      rehash(slots_.size() * 2);
    }
  }

  void maybe_cleanup_deleted() {
    if (deleted_ > slots_.size() / 4) {
      rehash(slots_.size());
    }
  }

  template <typename K, typename V>
  std::pair<size_t, bool> insert_no_grow(K&& key, V&& value) {
    const size_t mask = slots_.size() - 1;
    size_t index = bucket(key);
    size_t first_deleted = npos;
    for (;;) {
      Slot& slot = slots_[index];
      if (slot.state == State::full) {
        if (eq_(slot.kv.first, key)) {
          return {index, false};
        }
      } else if (slot.state == State::deleted) {
        if (first_deleted == npos) first_deleted = index;
      } else {
        const size_t target = first_deleted == npos ? index : first_deleted;
        Slot& target_slot = slots_[target];
        target_slot.kv.first = std::forward<K>(key);
        target_slot.kv.second = std::forward<V>(value);
        if (target_slot.state == State::deleted) --deleted_;
        target_slot.state = State::full;
        ++size_;
        return {target, true};
      }
      index = (index + 1) & mask;
    }
  }

  template <typename K>
  std::pair<size_t, bool> insert_key_no_grow(K&& key) {
    const size_t mask = slots_.size() - 1;
    size_t index = bucket(key);
    size_t first_deleted = npos;
    for (;;) {
      Slot& slot = slots_[index];
      if (slot.state == State::full) {
        if (eq_(slot.kv.first, key)) return {index, false};
      } else if (slot.state == State::deleted) {
        if (first_deleted == npos) first_deleted = index;
      } else {
        const size_t target = first_deleted == npos ? index : first_deleted;
        Slot& target_slot = slots_[target];
        target_slot.kv.first = std::forward<K>(key);
        target_slot.kv.second = T{};
        if (target_slot.state == State::deleted) --deleted_;
        target_slot.state = State::full;
        ++size_;
        return {target, true};
      }
      index = (index + 1) & mask;
    }
  }

  void rehash(size_t new_capacity) {
    new_capacity = detail::next_power_of_two(std::max(new_capacity, kInitialCapacity));
    std::vector<Slot> old_slots = std::move(slots_);
    slots_.assign(new_capacity, Slot{});
    size_ = 0;
    deleted_ = 0;
    for (auto& slot : old_slots) {
      if (slot.state == State::full) {
        (void)insert_no_grow(std::move(slot.kv.first), std::move(slot.kv.second));
      }
    }
  }

  std::vector<Slot> slots_;
  size_t size_{0};
  size_t deleted_{0};
  Hash hash_{};
  Eq eq_{};
};

}  // namespace fast_hash
