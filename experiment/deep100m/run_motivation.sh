#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
case "${1:-all}" in
  program1) exec "$SCRIPT_DIR/program1/run_program1.sh" "${@:2}" ;;
  program2) exec "$SCRIPT_DIR/program2/run_program2.sh" "${@:2}" ;;
  program3) exec "$SCRIPT_DIR/program3/run_program3.sh" "${@:2}" ;;
  all)
    "$SCRIPT_DIR/program1/run_program1.sh" all
    "$SCRIPT_DIR/program2/run_program2.sh" all
    "$SCRIPT_DIR/program3/run_program3.sh" all
    ;;
  *) echo "usage: $0 [all|program1|program2|program3] [program-action]" >&2; exit 2 ;;
esac
