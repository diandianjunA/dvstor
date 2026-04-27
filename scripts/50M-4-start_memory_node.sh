#!/bin/bash
# =============================================================================
# DVSTOR Memory Node Launcher (1M dataset)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build/dvstor_memory_node"
PID_FILE="$PROJECT_DIR/.memory_node4.pid"
LOG_FILE="$PROJECT_DIR/logs/memory_node4.log"

NUM_CLIENTS="${NUM_CLIENTS:-1}"
PORT="${PORT:-1237}"
MN_MEMORY="${MN_MEMORY:-50}"
FOREGROUND=false

usage() {
    echo "Usage: $0 [start|stop|restart|status] [options]"
    echo "Options:"
    echo "  -n, --num-clients <n>   (default: 1)"
    echo "  -p, --port <port>       (default: 1237)"
    echo "      --mn-memory <GB>    (default: 50)"
    echo "  -f, --foreground"
    exit 0
}

COMMAND="start"
EXTRA_ARGS=()

if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
    COMMAND="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-clients)  NUM_CLIENTS="$2"; shift 2 ;;
        -p|--port)         PORT="$2"; shift 2 ;;
        --mn-memory)       MN_MEMORY="$2"; shift 2 ;;
        -f|--foreground)   FOREGROUND=true; shift ;;
        -h|--help)         usage ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

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

do_status() {
    local pid
    if pid=$(get_pid); then
        echo "[DVSTOR Memory Node] running (PID: $pid)"
        return 0
    else
        echo "[DVSTOR Memory Node] not running"
        return 1
    fi
}

do_stop() {
    local pid
    if pid=$(get_pid); then
        echo "[DVSTOR Memory Node] stopping (PID: $pid) ..."
        kill "$pid"
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                rm -f "$PID_FILE"
                echo "[DVSTOR Memory Node] stopped"
                return 0
            fi
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null
        rm -f "$PID_FILE"
        echo "[DVSTOR Memory Node] force stopped"
    else
        echo "[DVSTOR Memory Node] not running"
    fi
}

do_start() {
    if pid=$(get_pid); then
        echo "[DVSTOR Memory Node] already running (PID: $pid)"
        exit 1
    fi

    if [[ ! -x "$BINARY" ]]; then
        echo "Error: binary not found: $BINARY"
        exit 1
    fi

    local args=(
        --is-server
        --num-clients "$NUM_CLIENTS"
        --port "$PORT"
        --mn-memory "$MN_MEMORY"
    )
    args+=("${EXTRA_ARGS[@]}")

    echo "[DVSTOR Memory Node] parameters:"
    echo "  port:         $PORT"
    echo "  num_clients:  $NUM_CLIENTS"
    echo "  mn_memory:    ${MN_MEMORY}GB"

    if [[ "$FOREGROUND" == true ]]; then
        exec "$BINARY" "${args[@]}"
    else
        mkdir -p "$(dirname "$LOG_FILE")"
        nohup "$BINARY" "${args[@]}" >> "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            echo "[DVSTOR Memory Node] started (PID: $pid)"
            echo "  log: tail -f $LOG_FILE"
        else
            rm -f "$PID_FILE"
            echo "Error: process exited immediately, check: tail -20 $LOG_FILE"
            exit 1
        fi
    fi
}

case "$COMMAND" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    *)       echo "Unknown command: $COMMAND"; exit 1 ;;
esac


