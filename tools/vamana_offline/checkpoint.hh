#pragma once

#include <atomic>
#include <csignal>

#include "tools/vamana_offline/graph.hh"

namespace tools::vamana_offline {

// SIGUSR1 flips this flag; build_vamana_graph checks it at segment
// boundaries and commits an on-disk checkpoint before exiting.
extern volatile std::sig_atomic_t g_checkpoint_pause_requested;

void install_checkpoint_signal_handler();
bool checkpoint_pause_requested();

// Unwinds to the builder main() after a pause checkpoint was committed.
struct CheckpointPauseException {};

struct GraphCheckpointMeta {
  u32 version = 1;
  size_t num_nodes = 0;
  u32 dim = 0;
  u32 R = 0;
  size_t medoid = 0;
  i32 seed = 0;
  // "main_loop": resume the insertion loop at next_step (segment boundary).
  // "consolidate": main loop finished; resume reverse-edge consolidation at
  // next_chunk.
  str phase{"main_loop"};
  size_t next_step = 0;
  size_t next_chunk = 0;
  size_t segment_size = 0;
};

filepath_t checkpoint_meta_path(const filepath_t& prefix);
bool graph_checkpoint_exists(const filepath_t& prefix);
// Light meta read used to size the graph before loading the arrays.
GraphCheckpointMeta read_graph_checkpoint_meta(const filepath_t& prefix);
// graph must already be initialised with meta.num_nodes / meta.R.
void load_graph_checkpoint_arrays(const filepath_t& prefix, VamanaGraph& graph,
                                  const GraphCheckpointMeta& meta,
                                  size_t num_threads);
void save_graph_checkpoint(const filepath_t& prefix, const VamanaGraph& graph,
                           const GraphCheckpointMeta& meta, size_t num_threads);
void remove_graph_checkpoint(const filepath_t& prefix);

}  // namespace tools::vamana_offline
