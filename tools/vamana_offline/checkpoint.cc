#include "tools/vamana_offline/checkpoint.hh"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include <library/utils.hh>

#include "tools/vamana_offline/progress.hh"

namespace tools::vamana_offline {

volatile std::sig_atomic_t g_checkpoint_pause_requested = 0;

namespace {

void checkpoint_signal_handler(int) { g_checkpoint_pause_requested = 1; }

filepath_t neighbors_path(const filepath_t& prefix) {
  return filepath_t{prefix.string() + ".ckpt.neighbors"};
}
filepath_t degrees_path(const filepath_t& prefix) {
  return filepath_t{prefix.string() + ".ckpt.degrees"};
}

void write_full(int fd, const void* data, size_t bytes, off_t offset,
                const str& what) {
  const char* cursor = static_cast<const char*>(data);
  size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t written = ::pwrite(fd, cursor, remaining, offset);
    if (written < 0) {
      if (errno == EINTR) continue;
      lib_failure("checkpoint pwrite failed for " + what + ": " +
                  std::strerror(errno));
    }
    cursor += written;
    offset += written;
    remaining -= static_cast<size_t>(written);
  }
}

void read_full(int fd, void* data, size_t bytes, off_t offset,
               const str& what) {
  char* cursor = static_cast<char*>(data);
  size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t got = ::pread(fd, cursor, remaining, offset);
    if (got < 0) {
      if (errno == EINTR) continue;
      lib_failure("checkpoint pread failed for " + what + ": " +
                  std::strerror(errno));
    }
    if (got == 0)
      lib_failure("checkpoint file truncated while reading " + what);
    cursor += got;
    offset += got;
    remaining -= static_cast<size_t>(got);
  }
}

int open_checked(const filepath_t& path, int flags, const str& what) {
  const int fd = ::open(path.c_str(), flags, 0644);
  if (fd < 0)
    lib_failure("checkpoint cannot open " + what + " " + path.string() +
                ": " + std::strerror(errno));
  return fd;
}

void parse_meta_line(GraphCheckpointMeta& meta, const str& line) {
  const size_t eq = line.find('=');
  if (eq == str::npos) return;
  const str key = line.substr(0, eq);
  const str value = line.substr(eq + 1);
  std::istringstream in(value);
  if (key == "version") in >> meta.version;
  else if (key == "num_nodes") in >> meta.num_nodes;
  else if (key == "dim") in >> meta.dim;
  else if (key == "R") in >> meta.R;
  else if (key == "build_capacity") in >> meta.build_capacity;
  else if (key == "beam_width") in >> meta.beam_width;
  else if (key == "alpha") in >> meta.alpha;
  else if (key == "build_slack_factor") in >> meta.build_slack_factor;
  else if (key == "legacy_reverse_drop") in >> std::boolalpha >> meta.legacy_reverse_drop;
  else if (key == "random_initial_graph") in >> std::boolalpha >> meta.random_initial_graph;
  else if (key == "final_reverse_consolidation") in >> std::boolalpha >> meta.final_reverse_consolidation;
  else if (key == "vector_data_type") in >> meta.vector_data_type;
  else if (key == "medoid") in >> meta.medoid;
  else if (key == "seed") in >> meta.seed;
  else if (key == "phase") in >> meta.phase;
  else if (key == "next_step") in >> meta.next_step;
  else if (key == "next_chunk") in >> meta.next_chunk;
  else if (key == "segment_size") in >> meta.segment_size;
}

void write_array_file(const filepath_t& path, const void* data, size_t bytes,
                      size_t num_threads, const str& what) {
  const int fd = open_checked(path, O_WRONLY | O_CREAT | O_TRUNC, what);
  const size_t workers =
      std::max<size_t>(1, effective_thread_count(static_cast<u32>(num_threads)));
  const size_t block = size_t{64} << 20;  // 64 MiB per pwrite task
  const size_t blocks = (bytes + block - 1) / block;
  std::cerr << "checkpoint: writing " << what << " (" << (bytes >> 30)
            << " GiB, " << blocks << " blocks)...\n";
  const auto io_start = std::chrono::steady_clock::now();
  parallel_for(size_t{0}, blocks, workers, [&](size_t b, size_t) {
    const size_t begin = b * block;
    const size_t length = std::min(block, bytes - begin);
    write_full(fd, static_cast<const char*>(data) + begin, length,
               static_cast<off_t>(begin), what);
  });
  if (::fsync(fd) != 0)
    lib_failure("checkpoint fsync failed for " + what + ": " +
                std::strerror(errno));
  ::close(fd);
  const auto io_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - io_start)
          .count();
  std::cerr << "checkpoint: " << what << " written in " << io_seconds
            << " seconds (" << (static_cast<double>(bytes >> 20) / io_seconds)
            << " MiB/s)\n";
}

void read_array_file(const filepath_t& path, void* data, size_t bytes,
                     size_t num_threads, const str& what) {
  const int fd = open_checked(path, O_RDONLY, what);
  struct stat st {};
  if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) != bytes)
    lib_failure("checkpoint file size mismatch for " + what + ": " +
                path.string());
  const size_t workers =
      std::max<size_t>(1, effective_thread_count(static_cast<u32>(num_threads)));
  const size_t block = size_t{64} << 20;
  const size_t blocks = (bytes + block - 1) / block;
  std::cerr << "checkpoint: loading " << what << " (" << (bytes >> 30)
            << " GiB)...\n";
  const auto io_start = std::chrono::steady_clock::now();
  parallel_for(size_t{0}, blocks, workers, [&](size_t b, size_t) {
    const size_t begin = b * block;
    const size_t length = std::min(block, bytes - begin);
    read_full(fd, static_cast<char*>(data) + begin, length,
              static_cast<off_t>(begin), what);
  });
  ::close(fd);
  const auto io_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - io_start)
          .count();
  std::cerr << "checkpoint: " << what << " loaded in " << io_seconds
            << " seconds\n";
}

}  // namespace

void install_checkpoint_signal_handler() {
  struct sigaction action {};
  action.sa_handler = checkpoint_signal_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGUSR1, &action, nullptr);
}

bool checkpoint_pause_requested() { return g_checkpoint_pause_requested != 0; }

filepath_t checkpoint_meta_path(const filepath_t& prefix) {
  return filepath_t{prefix.string() + ".ckpt.meta"};
}

bool graph_checkpoint_exists(const filepath_t& prefix) {
  std::error_code ec;
  return std::filesystem::exists(checkpoint_meta_path(prefix), ec);
}

GraphCheckpointMeta read_graph_checkpoint_meta(const filepath_t& prefix) {
  const filepath_t path = checkpoint_meta_path(prefix);
  std::ifstream in(path);
  lib_assert(in.good(), "cannot read checkpoint meta: " + path.string());
  GraphCheckpointMeta meta;
  str line;
  while (std::getline(in, line)) parse_meta_line(meta, line);
  lib_assert(meta.version == 1 || meta.version == 2,
             "unsupported checkpoint meta version: " + std::to_string(meta.version));
  if (meta.version == 1) meta.build_capacity = meta.R;
  lib_assert(meta.num_nodes > 0 && meta.R > 0 &&
                 meta.build_capacity >= meta.R && meta.segment_size > 0,
             "checkpoint meta is incomplete: " + path.string());
  lib_assert(meta.phase == "main_loop" || meta.phase == "consolidate" ||
                 (meta.version == 2 && meta.phase == "final_prune"),
             "checkpoint meta has unknown phase: " + meta.phase);
  return meta;
}

void load_graph_checkpoint_arrays(const filepath_t& prefix, VamanaGraph& graph,
                                  const GraphCheckpointMeta& meta,
                                  size_t num_threads) {
  lib_assert(graph.num_nodes == meta.num_nodes && graph.R == meta.R &&
                 graph.capacity == meta.build_capacity,
             "graph layout does not match checkpoint meta");
  read_array_file(neighbors_path(prefix), graph.neighbors.data(),
                  graph.neighbors.size() * sizeof(u32), num_threads,
                  "neighbors");
  read_array_file(degrees_path(prefix), graph.degrees.data(),
                  graph.degrees.size() * sizeof(u8), num_threads, "degrees");
  graph.medoid = meta.medoid;
}

void save_graph_checkpoint(const filepath_t& prefix, const VamanaGraph& graph,
                           const GraphCheckpointMeta& meta,
                           size_t num_threads) {
  std::error_code ec;
  const filepath_t dir = prefix.parent_path();
  if (!dir.empty()) std::filesystem::create_directories(dir, ec);
  write_array_file(neighbors_path(prefix), graph.neighbors.data(),
                   graph.neighbors.size() * sizeof(u32), num_threads,
                   "neighbors");
  write_array_file(degrees_path(prefix), graph.degrees.data(),
                   graph.degrees.size() * sizeof(u8), num_threads, "degrees");

  const filepath_t meta_tmp = filepath_t{prefix.string() + ".ckpt.meta.tmp"};
  {
    std::ofstream out(meta_tmp, std::ios::trunc);
    lib_assert(out.good(), "cannot write checkpoint meta: " + meta_tmp.string());
    out.precision(17);
    out << "version=" << meta.version << '\n';
    out << "num_nodes=" << meta.num_nodes << '\n';
    out << "dim=" << meta.dim << '\n';
    out << "R=" << meta.R << '\n';
    out << "build_capacity=" << meta.build_capacity << '\n';
    out << "beam_width=" << meta.beam_width << '\n';
    out << "alpha=" << meta.alpha << '\n';
    out << "build_slack_factor=" << meta.build_slack_factor << '\n';
    out << "legacy_reverse_drop=" << std::boolalpha << meta.legacy_reverse_drop << '\n';
    out << "random_initial_graph=" << meta.random_initial_graph << '\n';
    out << "final_reverse_consolidation=" << meta.final_reverse_consolidation << '\n';
    out << "vector_data_type=" << meta.vector_data_type << '\n';
    out << "medoid=" << meta.medoid << '\n';
    out << "seed=" << meta.seed << '\n';
    out << "phase=" << meta.phase << '\n';
    out << "next_step=" << meta.next_step << '\n';
    out << "next_chunk=" << meta.next_chunk << '\n';
    out << "segment_size=" << meta.segment_size << '\n';
    out.flush();
    lib_assert(out.good(), "failed writing checkpoint meta: " + meta_tmp.string());
  }
  std::filesystem::rename(meta_tmp, checkpoint_meta_path(prefix), ec);
  lib_assert(!ec, "failed to commit checkpoint meta: " + ec.message());
  std::cerr << "checkpoint: committed " << checkpoint_meta_path(prefix)
            << " phase=" << meta.phase
            << (meta.phase == "main_loop"
                    ? " next_step=" + std::to_string(meta.next_step)
                    : " next_chunk=" + std::to_string(meta.next_chunk))
            << '\n';
}

void remove_graph_checkpoint(const filepath_t& prefix) {
  std::error_code ec;
  std::filesystem::remove(neighbors_path(prefix), ec);
  std::filesystem::remove(degrees_path(prefix), ec);
  std::filesystem::remove(checkpoint_meta_path(prefix), ec);
}

}  // namespace tools::vamana_offline
