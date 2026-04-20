#!/bin/bash
# =============================================================================
# DVSTOR Breakdown Benchmark Runner
# =============================================================================
# 运行 DVSTOR breakdown benchmark（配置驱动）。
# 所有优化开关（如 gpudirect-rdma、后续新增优化）都从 service-config 读取。
# 默认执行 50% 读 / 50% 写 mixed 场景，并在结束后打印吞吐与延迟摘要。
# 时间模式默认采用 drain 语义：到达截止时间后不再发新请求，并等待已启动请求完成；
# 同时 benchmark 会根据已观测到的单次调用耗时，避免在窗口尾部再启动明显会拖很久的新调用。
# 该脚本不会启动 memory node，请先用 start_memory_node.sh 或现有部署方式启动服务端。
#
# 用法:
#   ./run_breakdown_test.sh [选项...]
#
# 选项:
#   -c, --service-config <path>   Compute 侧配置文件（默认: test/config/remote_singe_cn.ini）
#   -w, --workload <mode>         负载模式: mixed/query/insert/both（默认: mixed）
#       --read-ratio <ratio>      mixed 模式下读比例（默认: 0.5）
#       --client-threads <n>      前台压测线程数（默认: 4）
#       --warmup-seconds <n>      预热时长（秒，默认: 30）
#       --measure-seconds <n>     正式测量时长（秒，默认: 60）
#       --warmup-ops <n>          预热请求数（兼容旧模式）
#       --measure-ops <n>         正式测量请求数（兼容旧模式）
#       --query-file <path>       外部 query 文件（.fbin），不提供则使用 synthetic query
#       --report-dir <path>       报告输出目录（默认: ./reports/breakdown）
#       --label <name>            报告文件名前缀（默认: 时间戳）
#   -h, --help                    显示帮助信息
#
# 环境变量:
#   SERVICE_CONFIG, WORKLOAD, READ_RATIO, CLIENT_THREADS, WARMUP_SECONDS,
#   MEASURE_SECONDS, WARMUP_OPS, MEASURE_OPS, QUERY_FILE,
#   REPORT_DIR, LABEL
#
# 示例:
#   ./run_breakdown_test.sh
#   ./run_breakdown_test.sh --client-threads 8 --measure-seconds 120
#   ./run_breakdown_test.sh --workload query --query-file /data/queries.fbin
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build/dvstor_breakdown_benchmark"

SERVICE_CONFIG="${SERVICE_CONFIG:-$PROJECT_DIR/test/config/remote_singe_cn.ini}"
WORKLOAD="${WORKLOAD:-mixed}"
READ_RATIO="${READ_RATIO:-0.5}"
CLIENT_THREADS="${CLIENT_THREADS:-16}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
MEASURE_SECONDS="${MEASURE_SECONDS:-60}"
WARMUP_OPS="${WARMUP_OPS:-}"
MEASURE_OPS="${MEASURE_OPS:-}"
QUERY_FILE="${QUERY_FILE:-}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_DIR/reports/breakdown}"
LABEL="${LABEL:-$(date +%Y%m%d_%H%M%S)}"

usage() {
    sed -n '/^# 用法:/,/^# =====/p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--service-config) SERVICE_CONFIG="$2"; shift 2 ;;
        -w|--workload)       WORKLOAD="$2"; shift 2 ;;
        --read-ratio)        READ_RATIO="$2"; shift 2 ;;
        --client-threads)    CLIENT_THREADS="$2"; shift 2 ;;
        --warmup-seconds)    WARMUP_SECONDS="$2"; shift 2 ;;
        --measure-seconds)   MEASURE_SECONDS="$2"; shift 2 ;;
        --warmup-ops)        WARMUP_OPS="$2"; shift 2 ;;
        --measure-ops)       MEASURE_OPS="$2"; shift 2 ;;
        --query-file)        QUERY_FILE="$2"; shift 2 ;;
        --report-dir)        REPORT_DIR="$2"; shift 2 ;;
        --label)             LABEL="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 查看帮助"
            exit 1
            ;;
    esac
done

if [[ ! -f "$SERVICE_CONFIG" ]]; then
    echo "错误: 配置文件不存在: $SERVICE_CONFIG"
    exit 1
fi

if [[ -n "$QUERY_FILE" && ! -f "$QUERY_FILE" ]]; then
    echo "错误: QUERY_FILE 不存在或不是有效文件: $QUERY_FILE"
    echo "如果想使用 synthetic query，请不要设置 QUERY_FILE。"
    exit 1
fi

if [[ ! -x "$BINARY" ]]; then
    echo "错误: 找不到可执行文件 $BINARY"
    echo "请先编译项目: cd $PROJECT_DIR && mkdir -p build && cd build && cmake .. && make -j"
    exit 1
fi

mkdir -p "$REPORT_DIR"

JSON_REPORT="$REPORT_DIR/${LABEL}.json"
TEXT_REPORT="$REPORT_DIR/${LABEL}.txt"

ARGS=(
    --service-config "$SERVICE_CONFIG"
    --workload "$WORKLOAD"
    --read-ratio "$READ_RATIO"
    --client-threads "$CLIENT_THREADS"
    --report-json "$JSON_REPORT"
    --report-text "$TEXT_REPORT"
)

if [[ -n "$WARMUP_SECONDS" || -n "$MEASURE_SECONDS" ]]; then
    if [[ -z "$WARMUP_SECONDS" || -z "$MEASURE_SECONDS" ]]; then
        echo "错误: 使用时间模式时，WARMUP_SECONDS 和 MEASURE_SECONDS 都必须设置"
        exit 1
    fi
    ARGS+=(--warmup-seconds "$WARMUP_SECONDS" --measure-seconds "$MEASURE_SECONDS")
else
    if [[ -z "$WARMUP_OPS" || -z "$MEASURE_OPS" ]]; then
        echo "错误: 使用 ops 模式时，WARMUP_OPS 和 MEASURE_OPS 都必须设置"
        exit 1
    fi
    ARGS+=(--warmup-ops "$WARMUP_OPS" --measure-ops "$MEASURE_OPS")
fi

if [[ -n "$QUERY_FILE" ]]; then
    ARGS+=(--query-file "$QUERY_FILE")
fi

get_ini_value() {
    local key="$1"
    awk -F'=' -v k="$key" '
        {
            line=$0
            sub(/[;#].*$/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            if (line == "") next
            split(line, kv, "=")
            kk=kv[1]
            vv=substr(line, index(line, "=")+1)
            gsub(/^[ \t]+|[ \t]+$/, "", kk)
            gsub(/^[ \t]+|[ \t]+$/, "", vv)
            if (kk == k) val=vv
        }
        END { if (val != "") print val }
    ' "$SERVICE_CONFIG"
}

echo "[DVSTOR Breakdown] 运行参数:"
echo "  配置文件:       $SERVICE_CONFIG"
echo "  search-mode:    $(get_ini_value search-mode || true)"
echo "  load-index:     $(get_ini_value load-index || true)"
echo "  index-prefix:   $(get_ini_value index-prefix || true)"
echo "  gpudirect-rdma: $(get_ini_value gpudirect-rdma || true)"
echo "  neighbor-cache: $(get_ini_value neighbor-cache-mb || true) MB"
echo "  gpu-rabitq-cache: $(get_ini_value gpu-rabitq-cache-mb || true) MB"
echo "  负载模式:       $WORKLOAD"
echo "  读比例:         $READ_RATIO"
echo "  前台线程数:     $CLIENT_THREADS"
if [[ -n "$WARMUP_SECONDS" || -n "$MEASURE_SECONDS" ]]; then
    echo "  运行模式:       time (drain)"
    echo "  预热时长:       ${WARMUP_SECONDS}s"
    echo "  测量时长:       ${MEASURE_SECONDS}s"
else
    echo "  运行模式:       ops"
    echo "  预热请求数:     $WARMUP_OPS"
    echo "  测量请求数:     $MEASURE_OPS"
fi
echo "  操作粒度:       single_vector"
if [[ -n "$QUERY_FILE" ]]; then
    echo "  Query 文件:     $QUERY_FILE"
else
    echo "  Query 文件:     synthetic"
fi
echo "  JSON 报告:      $JSON_REPORT"
echo "  文本报告:       $TEXT_REPORT"
echo ""

"$BINARY" "${ARGS[@]}"

if command -v python3 >/dev/null 2>&1; then
    python3 - "$JSON_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
with report_path.open("r", encoding="utf-8") as f:
    doc = json.load(f)

def get(d, *keys, default=0):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

mode = get(doc, "meta", "run_mode", default="")
measure_seconds = get(doc, "meta", "measure_seconds", default=0)
if mode != "time":
    measure_seconds = 0

reads = get(doc, "meta", "measure_mixed", "completed_reads", default=0)
writes = get(doc, "meta", "measure_mixed", "completed_writes", default=0)
total_ops = reads + writes
throughput = (total_ops / measure_seconds) if measure_seconds else 0.0

q = get(doc, "query_breakdown", "latency", default={})
i = get(doc, "insert_breakdown", "latency", default={})

def ns_to_ms(x):
    return float(x) / 1_000_000.0

print("\n[DVSTOR Breakdown] 关键指标")
if measure_seconds:
    print(f"  throughput: {throughput:.2f} ops/s (total={total_ops}, duration={measure_seconds}s)")
if reads or writes:
    print(f"  mixed completed: reads={reads}, writes={writes}")
if q:
    print(
        "  query latency(ms): "
        f"mean={ns_to_ms(get(q, 'mean_end_to_end_ns')):.4f}, "
        f"p95={ns_to_ms(get(q, 'p95_end_to_end_ns')):.4f}, "
        f"p99={ns_to_ms(get(q, 'p99_end_to_end_ns')):.4f}"
    )
if i:
    print(
        "  insert latency(ms): "
        f"mean={ns_to_ms(get(i, 'mean_end_to_end_ns')):.4f}, "
        f"p95={ns_to_ms(get(i, 'p95_end_to_end_ns')):.4f}, "
        f"p99={ns_to_ms(get(i, 'p99_end_to_end_ns')):.4f}"
    )
PY
else
    echo "[DVSTOR Breakdown] 未检测到 python3，跳过关键指标摘要打印。"
fi
