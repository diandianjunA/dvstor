#!/usr/bin/env bash
set -euo pipefail

# This runner requires its trusted hook to restore a certified snapshot before
# every case.  Reports bind the reset log and snapshot ID; the analyzer also
# verifies the same input source, insert-ID range, initial Recall, and completed
# update count. Concurrent clients do not provide a deterministic per-operation
# commit order, and this experiment intentionally makes no mutation-order-hash
# claim.

MOTIVATION_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$MOTIVATION_DIR/.." && pwd)"
PROFILE="${PROFILE:-04_gpu_persistent_gpunetio}"
REPETITIONS="${REPETITIONS:-3}"
RESULT_ROOT="${DYNAEXTENT_RESULT_ROOT:-$MOTIVATION_DIR/results/dynaextent_mixed_ab}"
BEFORE_CASE_HOOK="${DYNAEXTENT_BEFORE_CASE_HOOK:-}"
CONCURRENCY=336
declare -A REPETITION_SNAPSHOT_IDS=()

if (( $# != 0 )); then
  echo "usage: DYNAEXTENT_BEFORE_CASE_HOOK=/absolute/reset-hook $0" >&2
  exit 1
fi
if [[ ! "$REPETITIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPETITIONS must be a positive integer: $REPETITIONS" >&2
  exit 1
fi
if (( REPETITIONS % 3 != 0 )); then
  echo "REPETITIONS must be a multiple of 3 so every policy occupies every Latin-square position: $REPETITIONS" >&2
  exit 1
fi
if [[ -z "$BEFORE_CASE_HOOK" ]]; then
  echo "DYNAEXTENT_BEFORE_CASE_HOOK is required; refusing to run without a per-case snapshot reset" >&2
  exit 1
fi
if [[ ! -f "$BEFORE_CASE_HOOK" || ! -s "$BEFORE_CASE_HOOK" ||
      ! -x "$BEFORE_CASE_HOOK" ]]; then
  echo "DYNAEXTENT_BEFORE_CASE_HOOK must name a nonempty executable file: $BEFORE_CASE_HOOK" >&2
  exit 1
fi

# Resolve and validate the shared profile before consuming any benchmark case.
# Both live-extent policies require this exact static-node sidecar.
source "$PROJECT_DIR/experiment/common.sh"
load_experiment_profile "$PROFILE"
if [[ ! -s "${INDEX_PREFIX}.gextent8" ]]; then
  echo "missing Live-Extent sidecar: ${INDEX_PREFIX}.gextent8" >&2
  echo "generate it where all .dat shards are available before starting the triplet" >&2
  exit 1
fi

config_for_policy() {
  case "$1" in
    fixed)
      echo "$MOTIVATION_DIR/configs/dynaextent_fixed.env"
      ;;
    static-only)
      echo "$MOTIVATION_DIR/configs/dynaextent_static_only.env"
      ;;
    dynaextent)
      echo "$MOTIVATION_DIR/configs/dynaextent_enabled.env"
      ;;
    *)
      echo "unsupported DynaExtent policy: $1" >&2
      return 1
      ;;
  esac
}

run_one() {
  local policy="$1"
  local repetition="$2"
  local latin_position="$3"
  local latin_cycle=$(( (repetition - 1) / 3 + 1 ))
  local config_file
  local run_dir
  local reset_log
  local hook_status
  local snapshot_id
  local report_path
  local -a snapshot_ids
  local -a report_paths

  config_file="$(config_for_policy "$policy")"
  run_dir="$RESULT_ROOT/$policy/concurrency_${CONCURRENCY}/repeat_${repetition}"
  reset_log="$run_dir/before_case_reset.log"
  if [[ -e "$run_dir" ]]; then
    echo "case directory already exists; refusing to mix or overwrite reports: $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"

  hook_status=0
  {
    echo "hook=$BEFORE_CASE_HOOK"
    echo "policy=$policy"
    echo "concurrency=$CONCURRENCY"
    echo "repetition=$repetition"
    echo "latin_position=$latin_position"
    echo "latin_cycle=$latin_cycle"
    echo "run_dir=$run_dir"
    echo "started_at=$(date --iso-8601=seconds)"
    DYNAEXTENT_POLICY="$policy" \
      DYNAEXTENT_CONCURRENCY="$CONCURRENCY" \
      DYNAEXTENT_REPETITION="$repetition" \
      DYNAEXTENT_LATIN_POSITION="$latin_position" \
      DYNAEXTENT_LATIN_CYCLE="$latin_cycle" \
      DYNAEXTENT_RUN_DIR="$run_dir" \
      "$BEFORE_CASE_HOOK" \
        "$policy" "$CONCURRENCY" "$repetition" "$run_dir" || hook_status=$?
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "exit_status=$hook_status"
  } >"$reset_log" 2>&1
  if (( hook_status != 0 )); then
    echo "before-case reset failed for policy=$policy repeat=$repetition; see $reset_log" >&2
    exit "$hook_status"
  fi
  mapfile -t snapshot_ids < <(sed -n 's/^snapshot_id=//p' "$reset_log")
  if (( ${#snapshot_ids[@]} != 1 )) ||
     [[ ! "${snapshot_ids[0]:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._:+/@-]{7,255}$ ]]; then
    echo "before-case reset must emit exactly one 'snapshot_id=<immutable-id-or-digest>' line for policy=$policy repeat=$repetition; see $reset_log" >&2
    exit 1
  fi
  snapshot_id="${snapshot_ids[0]}"
  if [[ -n "${REPETITION_SNAPSHOT_IDS[$repetition]+present}" &&
        "${REPETITION_SNAPSHOT_IDS[$repetition]}" != "$snapshot_id" ]]; then
    echo "snapshot_id mismatch within repetition=$repetition: expected ${REPETITION_SNAPSHOT_IDS[$repetition]}, found $snapshot_id for policy=$policy" >&2
    exit 1
  fi
  REPETITION_SNAPSHOT_IDS[$repetition]="$snapshot_id"

  (
    set -a
    source "$MOTIVATION_DIR/configs/common.env"
    source "$config_file"

    # Pre-registered dynamic mixed-workload contract.  Assign after sourcing
    # all configs so caller environment cannot silently change a case.
    WORKLOAD=mixed
    MIXED_MODE=rate_limited
    TARGET_QUERY_QPS=40000
    TARGET_WRITE_QPS=1000
    BENCHMARK_CLIENT_THREADS="$CONCURRENCY"
    READ_RATIO=0.5
    BENCHMARK_MODE=time
    WARMUP_SECONDS=30
    MEASURE_SECONDS=120
    RECALL_QUERIES=1000
    RECALL_K=10
    RECALL_MODE=all
    WRITE_INSERT_RATIO=1
    WRITE_UPSERT_RATIO=0
    WRITE_DELETE_RATIO=0
    GPU_GRAPH_PREFETCH_DEPTH=16
    GPU_QUERY_BEAM_MERGE_POLICY=stable-run
    GPU_TRAVERSAL_BEAM_WIDTH=128
    GPU_MAX_EXPANSIONS=384
    GPU_FINAL_RERANK_WIDTH=128
    QUERY_RDMA_TRACE_MODE=off
    REPORT_DIR="$run_dir"
    set +a
    "$PROJECT_DIR/experiment/run_breakdown.sh" "$PROFILE"
  )

  shopt -s nullglob
  report_paths=("$run_dir/$PROFILE"/sift100m_*.json)
  shopt -u nullglob
  if (( ${#report_paths[@]} != 1 )); then
    echo "expected exactly one benchmark JSON to bind to the reset certificate in $run_dir/$PROFILE, found ${#report_paths[@]}" >&2
    exit 1
  fi
  report_path="${report_paths[0]}"
  python3 - "$report_path" "$reset_log" "$snapshot_id" "$policy" \
    "$CONCURRENCY" "$repetition" "$latin_position" "$latin_cycle" \
    <<'PY_RESET_CERTIFICATE'
import hashlib
import json
import os
import sys

(
    report_path,
    reset_log_path,
    snapshot_id,
    policy,
    concurrency,
    repetition,
    latin_position,
    latin_cycle,
) = sys.argv[1:]

with open(reset_log_path, "rb") as stream:
    reset_log_sha256 = hashlib.sha256(stream.read()).hexdigest()
with open(report_path, "r", encoding="utf-8") as stream:
    report = json.load(stream)
if not isinstance(report, dict):
    raise SystemExit(f"benchmark report root is not an object: {report_path}")
if "dynaextent_reset" in report:
    raise SystemExit(
        f"benchmark report already contains dynaextent_reset: {report_path}")

report["dynaextent_reset"] = {
    "schema_version": 1,
    "snapshot_id": snapshot_id,
    "reset_log_sha256": reset_log_sha256,
    "policy": policy,
    "concurrency": int(concurrency),
    "repetition": int(repetition),
    "latin_position": int(latin_position),
    "latin_cycle": int(latin_cycle),
}
temporary_path = report_path + ".dynaextent-reset.tmp"
with open(temporary_path, "w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
os.replace(temporary_path, report_path)
PY_RESET_CERTIFICATE
}

# A cyclic 3x3 Latin square prevents any one policy from always occupying the
# first, middle, or final thermal/network position.  Additional repetitions
# repeat the same balanced rotation.
policies=(fixed static-only dynaextent)
for ((repetition = 1; repetition <= REPETITIONS; ++repetition)); do
  offset=$(( (repetition - 1) % ${#policies[@]} ))
  for ((step = 0; step < ${#policies[@]}; ++step)); do
    policy="${policies[$(( (offset + step) % ${#policies[@]} ))]}"
    echo "[DynaExtent] repeat=$repetition position=$((step + 1)) policy=$policy"
    run_one "$policy" "$repetition" "$((step + 1))"
  done
done

python3 "$MOTIVATION_DIR/summarize_dynaextent_mixed_ab.py" "$RESULT_ROOT"
echo "DynaExtent mixed triplet reports: $RESULT_ROOT"
