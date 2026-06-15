#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/run_stage_breakdown.sh"
"$SCRIPT_DIR/run_concurrency_sweep.sh"
"$SCRIPT_DIR/run_data_path.sh"
"$SCRIPT_DIR/run_expansion_sweep.sh"
"$SCRIPT_DIR/summarize.py"

