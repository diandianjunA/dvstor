#pragma once

#include <filesystem>
#include <library/types.hh>

#include "ankerl/unordered_dense.h"

using node_t = u32;
using element_t = f32;
using distance_t = f32;

using filepath_t = std::filesystem::path;

template <typename T>
using hashset_t = ankerl::unordered_dense::set<T>;

template <typename K, typename V>
using hashmap_t = ankerl::unordered_dense::map<K, V>;
