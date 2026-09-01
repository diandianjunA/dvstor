#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

QUERY_THREADS="${QUERY_THREADS:-20}"
UPDATE_THREADS="${UPDATE_THREADS:-0}"
PROFILES="${PROFILES:-baseline}"
DATASET="${DATASET:-sift100m}"
WARMUP_SECONDS="${WARMUP_SECONDS:-15}"
MEASURE_SECONDS="${MEASURE_SECONDS:-120}"
REPEATS="${REPEATS:-1}"
EXPERIMENT_ID="${EXPERIMENT_ID:-mixed_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-}"
NO_STORAGE_PROMPT="${NO_STORAGE_PROMPT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARIZE_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./experiment/mixed_test/run_mixed_test.sh \
         --query-threads N --update-threads N[,N...] [options]

Options:
  --query-threads N       Fixed dedicated query threads (required)
  --update-threads LIST   Update-thread sweep, for example 0,8,16,32 (required)
  --profiles LIST         Profile aliases/names (default: baseline,full)
  --dataset NAME          sift100m, deep100m, or spacev100m (default: sift100m)
  --warmup-seconds N      Warmup duration (default: 15)
  --measure-seconds N     Measurement duration (default: 120)
  --repeats N             Independent repetitions per point (default: 1)
  --experiment-id NAME    Result group name (default: timestamp)
  --result-root PATH      Explicit result group directory (also enables resume)
  --no-storage-prompt     Do not pause; caller must reset storage before each case
  --summarize-only        Rebuild CSV files in RESULT_ROOT without running cases
  --dry-run               Print all resolved cases without running
  -h, --help              Show this help

Aliases: baseline=04_gpu_persistent_gpunetio_baseline,
         full=04_gpu_persistent_gpunetio.
The same settings can be supplied with the uppercase environment variables.
EOF
}

need_value() {
  if (( $# < 2 )); then
    echo "missing value for $1" >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --query-threads) need_value "$@"; QUERY_THREADS="$2"; shift 2 ;;
    --update-threads) need_value "$@"; UPDATE_THREADS="$2"; shift 2 ;;
    --profiles) need_value "$@"; PROFILES="$2"; shift 2 ;;
    --dataset) need_value "$@"; DATASET="$2"; shift 2 ;;
    --warmup-seconds) need_value "$@"; WARMUP_SECONDS="$2"; shift 2 ;;
    --measure-seconds) need_value "$@"; MEASURE_SECONDS="$2"; shift 2 ;;
    --repeats) need_value "$@"; REPEATS="$2"; shift 2 ;;
    --experiment-id) need_value "$@"; EXPERIMENT_ID="$2"; shift 2 ;;
    --result-root) need_value "$@"; RESULT_ROOT="$2"; shift 2 ;;
    --no-storage-prompt) NO_STORAGE_PROMPT=1; shift ;;
    --summarize-only) SUMMARIZE_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$DATASET" in
  sift100m|deep100m|spacev100m) ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac
EXPERIMENT_DIR="$PROJECT_DIR/experiment/$DATASET"
BENCHMARK_RUNNER="$EXPERIMENT_DIR/run_breakdown.sh"
[[ -x "$BENCHMARK_RUNNER" ]] || {
  echo "missing executable benchmark runner: $BENCHMARK_RUNNER" >&2
  exit 1
}

if [[ -z "$RESULT_ROOT" ]]; then
  RESULT_ROOT="$SCRIPT_DIR/results/$EXPERIMENT_ID"
fi
if (( SUMMARIZE_ONLY )); then
  python3 "$SCRIPT_DIR/summarize_results.py" "$RESULT_ROOT"
  exit
fi

[[ "$QUERY_THREADS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--query-threads must be a positive integer" >&2; exit 2;
}
[[ "$WARMUP_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--warmup-seconds must be a positive integer" >&2; exit 2;
}
[[ "$MEASURE_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--measure-seconds must be a positive integer" >&2; exit 2;
}
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--repeats must be a positive integer" >&2; exit 2;
}

IFS=',' read -r -a update_values <<< "$UPDATE_THREADS"
(( ${#update_values[@]} > 0 )) || {
  echo "--update-threads requires at least one value" >&2; exit 2;
}
for updates in "${update_values[@]}"; do
  [[ "$updates" =~ ^[0-9]+$ ]] || {
    echo "invalid update-thread value: $updates" >&2; exit 2;
  }
done

resolve_profile() {
  case "$1" in
    baseline) echo 04_gpu_persistent_gpunetio_baseline ;;
    full) echo 04_gpu_persistent_gpunetio ;;
    *) echo "$1" ;;
  esac
}

profile_modes() {
  case "$1" in
    04_gpu_persistent_gpunetio_baseline) echo coupled fixed coupled ;;
    04_gpu_persistent_gpunetio) echo decoupled adaptive decoupled ;;
    *) echo "" ;;
  esac
}

IFS=',' read -r -a requested_profiles <<< "$PROFILES"
profiles=()
for requested in "${requested_profiles[@]}"; do
  [[ -n "$requested" ]] || { echo "empty profile in --profiles" >&2; exit 2; }
  profile="$(resolve_profile "$requested")"
  [[ -f "$EXPERIMENT_DIR/profiles/$profile.env" ]] || {
    echo "profile does not exist for $DATASET: $profile" >&2; exit 2;
  }
  profiles+=("$profile")
done

case_count=$(( ${#profiles[@]} * ${#update_values[@]} * REPEATS ))
echo "[mixed-test] dataset=$DATASET query_threads=$QUERY_THREADS updates=$UPDATE_THREADS"
echo "[mixed-test] profiles=${profiles[*]} repeats=$REPEATS cases=$case_count"
echo "[mixed-test] warmup=${WARMUP_SECONDS}s measure=${MEASURE_SECONDS}s result=$RESULT_ROOT"

case_index=0
for profile in "${profiles[@]}"; do
  for updates in "${update_values[@]}"; do
    total_threads=$((QUERY_THREADS + updates))
    read_ratio="$(awk -v q="$QUERY_THREADS" -v total="$total_threads" \
      'BEGIN { printf "%.17g", q / total }')"
    workload=mixed
    (( updates > 0 )) || workload=query
    for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
      ((case_index += 1))
      case_name="${DATASET}_${profile}_q${QUERY_THREADS}_u${updates}_rep${repeat}"
      case_dir="$RESULT_ROOT/runs/$case_name"
      run_token="${EXPERIMENT_ID}_${case_index}_${case_name}"
      echo
      echo "[mixed-test] case $case_index/$case_count: profile=$profile q=$QUERY_THREADS u=$updates repeat=$repeat"
      echo "[mixed-test] workload=$workload clients=$total_threads read_ratio=$read_ratio"

      if [[ -f "$case_dir/DONE" ]]; then
        echo "[mixed-test] completed case reused: $case_dir"
        continue
      fi
      if [[ -e "$case_dir" ]]; then
        echo "incomplete case already exists: $case_dir" >&2
        echo "move it aside or choose another --result-root" >&2
        exit 1
      fi
      if (( DRY_RUN )); then
        continue
      fi

      storage_log_dir=""
      if [[ "$NO_STORAGE_PROMPT" != 1 ]]; then
        echo "请在存储节点重启本 case 的干净存储服务："
        printf '  %q --dataset %q --profile %q --run-token %q\n' \
          "$SCRIPT_DIR/start_storage_profile.sh" "$DATASET" "$profile" "$run_token"
        read -r -p "看到 storage ready 后按 Enter（输入 q 退出）: " answer
        [[ "$answer" != q && "$answer" != Q ]] || exit 130
        ready_file="$SCRIPT_DIR/.storage_ready.tsv"
        [[ -s "$ready_file" ]] || {
          echo "missing storage ready marker: $ready_file" >&2; exit 1;
        }
        IFS=$'\t' read -r ready_token ready_dataset ready_profile storage_log_dir _ < "$ready_file"
        if [[ "$ready_token" != "$run_token" || "$ready_dataset" != "$DATASET" ||
              "$ready_profile" != "$profile" ]]; then
          echo "storage marker does not match this case" >&2
          echo "marker: $(<"$ready_file")" >&2
          exit 1
        fi
      else
        echo "[mixed-test][warning] storage reset is delegated to the caller"
      fi

      mkdir -p "$case_dir"
      runner_env=(env
        WORKLOAD="$workload"
        BENCHMARK_MODE=time
        BENCHMARK_CLIENT_THREADS="$total_threads"
        MIXED_MODE=fixed_threads
        READ_RATIO="$read_ratio"
        TARGET_QUERY_QPS=0
        TARGET_WRITE_QPS=0
        MIXED_WRITE_THREADS=0
        WRITE_INSERT_RATIO=1
        WRITE_UPSERT_RATIO=0
        WRITE_DELETE_RATIO=0
        WARMUP_SECONDS="$WARMUP_SECONDS"
        MEASURE_SECONDS="$MEASURE_SECONDS"
        REPORT_DIR="$case_dir/reports")
      read -r update_mode access_mode progression_mode < <(profile_modes "$profile")
      if [[ -n "$update_mode" ]]; then
        runner_env+=(
          STORAGE_OWNER_UPDATE_COMPLETION_MODE="$update_mode"
          GPU_DYNAMIC_GRAPH_ACCESS_MODE="$access_mode"
          GPU_RDMA_SEARCH_PROGRESSION_MODE="$progression_mode")
      fi
      if [[ -n "$storage_log_dir" ]]; then
        runner_env+=(LOG_DIR="$storage_log_dir")
      fi
      "${runner_env[@]}" "$BENCHMARK_RUNNER" "$profile" 2>&1 | tee "$case_dir/driver.log"

      mapfile -t reports < <(find "$case_dir/reports" -type f -name '*.json' -print)
      if (( ${#reports[@]} != 1 )); then
        echo "expected one JSON report, found ${#reports[@]}" >&2
        exit 1
      fi
      cp "${reports[0]}" "$case_dir/report.json"
      python3 "$SCRIPT_DIR/validate_report.py" \
        --report "$case_dir/report.json" --dataset "$DATASET" \
        --profile "$profile" --query-threads "$QUERY_THREADS" \
        --update-threads "$updates" --warmup-seconds "$WARMUP_SECONDS" \
        --measure-seconds "$MEASURE_SECONDS"
      python3 - "$case_dir/run_metadata.json" "$DATASET" "$profile" \
        "$QUERY_THREADS" "$updates" "$read_ratio" "$repeat" "$run_token" <<'PY'
import json
import sys

(path, dataset, profile, queries, updates, ratio, repeat, token) = sys.argv[1:]
with open(path, "w", encoding="utf-8") as stream:
    json.dump({
        "dataset": dataset,
        "profile": profile,
        "query_threads": int(queries),
        "update_threads": int(updates),
        "configured_read_ratio": float(ratio),
        "repeat": int(repeat),
        "run_token": token,
    }, stream, indent=2)
    stream.write("\n")
PY
      touch "$case_dir/DONE"
      python3 "$SCRIPT_DIR/summarize_results.py" "$RESULT_ROOT"
    done
  done
done

if (( DRY_RUN )); then
  echo "[mixed-test] dry run complete; no files were written"
else
  python3 "$SCRIPT_DIR/summarize_results.py" "$RESULT_ROOT"
  echo "[mixed-test] complete: $RESULT_ROOT"
fi
