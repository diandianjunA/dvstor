#!/bin/bash
# =============================================================================
# SHINE Memory Node Launcher
# =============================================================================
# 启动 SHINE 内存节点（服务模式）。
# 支持前台/后台运行，提供 start/stop/status/restart 操作。
#
# 用法:
#   ./start_memory_node.sh [命令] [选项...]
#
# 命令:
#   start       启动内存节点（默认）
#   stop        停止后台运行的内存节点
#   restart     重启内存节点
#   status      查看内存节点运行状态
#
# 选项:
#   -n, --num-clients <n>     客户端数量（默认: 1）
#   -p, --port <port>         RDMA 通信端口（默认: 1234）
#       --mn-memory <GB>      内存节点内存（GB）（默认: 10）
#       --index-file <path>   启动时直接加载本地 shard 文件
#   -f, --foreground          前台运行（默认后台运行）
#   -h, --help                显示此帮助信息
#
# 环境变量:
#   所有参数均可通过同名大写环境变量覆盖，例如:
#     NUM_CLIENTS=2 MN_MEMORY=20 ./start_memory_node.sh
#
# 示例:
#   ./start_memory_node.sh                          # 使用默认参数后台启动
#   ./start_memory_node.sh -n 2 --mn-memory 20      # 指定客户端数和内存
#   ./start_memory_node.sh -p 1235                  # 在同机启动另一个 memory node 实例
#   ./start_memory_node.sh -f                       # 前台运行（调试用）
#   ./start_memory_node.sh stop                     # 停止节点
#   ./start_memory_node.sh status                   # 查看状态
#   ./start_memory_node.sh restart                  # 重启节点
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build/shine"
LOG_DIR="$PROJECT_DIR/logs"

# ---- 默认参数（可通过环境变量覆盖） ----
NUM_CLIENTS="${NUM_CLIENTS:-1}"
PORT="${PORT:-1234}"
MN_MEMORY="${MN_MEMORY:-10}"
INDEX_FILE="${INDEX_FILE:-}"
FOREGROUND=false

# ---- 帮助信息 ----
usage() {
    sed -n '/^# 用法:/,/^# =====/p' "$0" | sed 's/^# \?//'
    exit 0
}

# ---- 解析命令行参数 ----
COMMAND="start"
EXTRA_ARGS=()

# 第一个非 - 开头的参数作为命令
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    COMMAND="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-clients)  NUM_CLIENTS="$2"; shift 2 ;;
        -p|--port)         PORT="$2"; shift 2 ;;
        --mn-memory)       MN_MEMORY="$2"; shift 2 ;;
        --index-file)      INDEX_FILE="$2"; shift 2 ;;
        -f|--foreground)   FOREGROUND=true; shift ;;
        -h|--help)         usage ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

PID_FILE="$PROJECT_DIR/.memory_node_${PORT}.pid"
LOG_FILE="$LOG_DIR/memory_node_${PORT}.log"

# ---- 工具函数 ----
get_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# ---- 命令实现 ----
do_status() {
    local pid
    if pid=$(get_pid); then
        echo "[SHINE Memory Node] 运行中 (PID: $pid)"
        echo "  日志文件: $LOG_FILE"
        echo "  PID 文件: $PID_FILE"
        return 0
    else
        echo "[SHINE Memory Node] 未运行"
        return 1
    fi
}

do_stop() {
    local pid
    if pid=$(get_pid); then
        echo "[SHINE Memory Node] 正在停止 (PID: $pid) ..."
        kill "$pid"
        # 等待进程退出，最多 10 秒
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                rm -f "$PID_FILE"
                echo "[SHINE Memory Node] 已停止"
                return 0
            fi
            sleep 1
        done
        echo "[SHINE Memory Node] 进程未响应，强制终止 ..."
        kill -9 "$pid" 2>/dev/null
        rm -f "$PID_FILE"
        echo "[SHINE Memory Node] 已强制停止"
    else
        echo "[SHINE Memory Node] 未运行"
    fi
}

do_start() {
    # 检查是否已在运行
    if pid=$(get_pid); then
        echo "[SHINE Memory Node] 已在运行 (PID: $pid)，如需重启请使用 restart 命令"
        exit 1
    fi

    # 检查二进制文件
    if [[ ! -x "$BINARY" ]]; then
        echo "错误: 找不到可执行文件 $BINARY"
        echo "请先编译项目: cd $PROJECT_DIR && mkdir -p build && cd build && cmake .. && make"
        exit 1
    fi

    # 构建参数列表
    local args=(
        --is-server
        --num-clients "$NUM_CLIENTS"
        --port "$PORT"
        --mn-memory "$MN_MEMORY"
    )

    if [[ -n "$INDEX_FILE" ]]; then
        args+=(--server-index-file "$INDEX_FILE")
    fi

    args+=("${EXTRA_ARGS[@]}")

    echo "[SHINE Memory Node] 启动参数:"
    echo "  RDMA 端口:    $PORT"
    echo "  客户端数:     $NUM_CLIENTS"
    echo "  内存(GB):     $MN_MEMORY"
    if [[ -n "$INDEX_FILE" ]]; then
        echo "  索引文件:     $INDEX_FILE"
    fi

    if [[ "$FOREGROUND" == true ]]; then
        echo "  模式:         前台运行"
        echo ""
        exec "$BINARY" "${args[@]}"
    else
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "  模式:         后台运行"
        echo "  日志文件:     $LOG_FILE"
        echo ""
        nohup "$BINARY" "${args[@]}" >> "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        # 短暂等待，检查进程是否立即退出
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            echo "[SHINE Memory Node] 已启动 (PID: $pid)"
            echo ""
            echo "常用操作:"
            echo "  查看状态:  $0 status"
            echo "  查看日志:  tail -f $LOG_FILE"
            echo "  停止节点:  $0 stop"
        else
            rm -f "$PID_FILE"
            echo "错误: 进程启动后立即退出，请检查日志:"
            echo "  tail -20 $LOG_FILE"
            exit 1
        fi
    fi
}

# ---- 执行命令 ----
case "$COMMAND" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    *)
        echo "未知命令: $COMMAND"
        echo "可用命令: start, stop, restart, status"
        echo "使用 -h 查看帮助"
        exit 1
        ;;
esac
