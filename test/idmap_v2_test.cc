#include <cassert>
#include <cstring>
#include <sstream>
#include <unordered_set>

#include "vamana/idmap.hh"

namespace {

constexpr u64 kBuildFingerprint = 0x123456789abcdef0ULL;
constexpr u64 kOwnerFingerprint = 0x0fedcba987654321ULL;
constexpr u32 kOwner = 1;
constexpr u32 kShards = 3;
constexpr u32 kNodeSize = 64;
constexpr u64 kNodeBase = 16;

const vec<u64> kStaticCounts{2, 2, 2};

vamana::idmap::ValidationContext context() {
  return {
    .build_fingerprint = kBuildFingerprint,
    .owner_shard_fingerprint = kOwnerFingerprint,
    .node_base_offset = kNodeBase,
    .owner_shard = kOwner,
    .shard_count = kShards,
    .node_size = kNodeSize,
    .id_offset = 8,
    .generation_offset = 12,
    .slot_incarnation_offset = 16,
    .static_entry_counts = span<const u64>{kStaticCounts},
  };
}

vec<vamana::idmap::Entry> valid_entries() {
  return {
    {.id = 1, .rptr_raw = RemotePtr{0, kNodeBase}.raw_address},
    {.id = 4,
     .rptr_raw = RemotePtr{2, kNodeBase + kNodeSize}.raw_address},
  };
}

vamana::idmap::Header make_header(
    span<const vamana::idmap::Entry> entries) {
  vamana::idmap::Header header;
  header.build_fingerprint = kBuildFingerprint;
  header.owner_shard_fingerprint = kOwnerFingerprint;
  header.owner_shard = kOwner;
  header.shard_count = kShards;
  header.node_base_offset = kNodeBase;
  header.node_size = kNodeSize;
  header.id_offset = 8;
  header.generation_offset = 12;
  header.slot_incarnation_offset = 16;
  header.entry_count = entries.size();
  const bool sized = vamana::idmap::checked_payload_bytes(
    header.entry_count, header.payload_bytes);
  assert(sized);
  header.payload_checksum = vamana::idmap::checksum(
    span<const byte_t>{reinterpret_cast<const byte_t*>(entries.data()),
                       static_cast<size_t>(header.payload_bytes)});
  header.header_checksum =
    vamana::idmap::compute_header_checksum(header);
  return header;
}

str payload(span<const vamana::idmap::Entry> entries) {
  return {reinterpret_cast<const char*>(entries.data()),
          entries.size() * sizeof(vamana::idmap::Entry)};
}

bool validate_payload(const vamana::idmap::Header& header,
                      const str& bytes) {
  std::istringstream input(bytes, std::ios::in | std::ios::binary);
  std::unordered_set<node_t> unique;
  return vamana::idmap::read_validated_payload(
    input, header, context(), [&](const vamana::idmap::Entry& entry) {
      return unique.insert(entry.id).second;
    });
}

}  // namespace

int main() {
  const vec<vamana::idmap::Entry> entries = valid_entries();
  const str bytes = payload(entries);
  const vamana::idmap::Header header = make_header(entries);
  const u64 file_bytes = sizeof(header) + bytes.size();
  assert(sizeof(vamana::idmap::Header) == 128);
  assert(sizeof(vamana::idmap::Entry) == 24);
  assert(vamana::idmap::valid_header(header, file_bytes, context()));
  assert(validate_payload(header, bytes));

  // The exact file envelope is part of the contract.
  assert(!vamana::idmap::valid_header(header, file_bytes + 1, context()));

  // A payload mutation is caught even if every entry remains structurally
  // plausible.
  str tampered_payload = bytes;
  tampered_payload.back() ^= 1;
  assert(!validate_payload(header, tampered_payload));

  // Version 1 is intentionally incompatible, even with a recomputed header
  // checksum.
  auto old = header;
  old.version = 1;
  old.header_checksum = vamana::idmap::compute_header_checksum(old);
  assert(!vamana::idmap::valid_header(old, file_bytes, context()));

  // Same-size files from another build or owner cannot be substituted.
  auto cross_build = header;
  cross_build.build_fingerprint ^= 0x55;
  cross_build.header_checksum =
    vamana::idmap::compute_header_checksum(cross_build);
  assert(!vamana::idmap::valid_header(
    cross_build, file_bytes, context()));
  auto cross_owner = header;
  cross_owner.owner_shard = 2;
  cross_owner.owner_shard_fingerprint ^= 0xaa;
  cross_owner.header_checksum =
    vamana::idmap::compute_header_checksum(cross_owner);
  assert(!vamana::idmap::valid_header(
    cross_owner, file_bytes, context()));

  // Duplicate logical IDs are rejected by the streaming consumer without a
  // second payload-sized validation copy.
  vec<vamana::idmap::Entry> duplicate = entries;
  duplicate[1].id = duplicate[0].id;
  auto duplicate_header = make_header(duplicate);
  assert(!validate_payload(duplicate_header, payload(duplicate)));

  // Authority modulo, immutable generation/flags/reserved, and tagged static
  // pointer bounds are checked independently of the checksum.
  vec<vamana::idmap::Entry> invalid = entries;
  invalid[0].id = 2;
  assert(!validate_payload(make_header(invalid), payload(invalid)));
  invalid = entries;
  invalid[0].generation = 1;
  assert(!validate_payload(make_header(invalid), payload(invalid)));
  invalid = entries;
  invalid[0].flags = 1;
  assert(!validate_payload(make_header(invalid), payload(invalid)));
  invalid = entries;
  invalid[0].reserved = 1;
  assert(!validate_payload(make_header(invalid), payload(invalid)));
  invalid = entries;
  invalid[0].rptr_raw = RemotePtr{0, kNodeBase, 1}.raw_address;
  assert(!validate_payload(make_header(invalid), payload(invalid)));
  invalid = entries;
  invalid[0].rptr_raw = RemotePtr{0, kNodeBase + 2 * kNodeSize}.raw_address;
  assert(!validate_payload(make_header(invalid), payload(invalid)));

  return 0;
}
