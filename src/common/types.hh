#pragma once

#include <filesystem>
#include <unordered_map>
#include <library/types.hh>

#include "common/fast_hash.hh"

using node_t = u32;
using element_t = f32;
using distance_t = f32;

using filepath_t = std::filesystem::path;

template <typename T>
using hashset_t = fast_hash::FlatHashSet<T>;

template <typename K, typename V>
using hashmap_t = std::unordered_map<K, V>;
