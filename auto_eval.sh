#!/usr/bin/env bash
# Evaluate one or more BayesianGS-SLAM run directories.
#
# Usage:
#   bash auto_eval.sh /path/to/run1 [/path/to/run2 ...]
#   bash auto_eval.sh --novel_view_render /path/to/run
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: bash auto_eval.sh [--novel_view_render] CHECKPOINT [...]"
  exit 1
fi

extra=()
while [ "$#" -gt 0 ] && [[ "$1" == --* ]]; do
  extra+=("$1")
  shift
done

for run in "$@"; do
  python run_evaluation.py --checkpoint_path "$run" "${extra[@]}"
done
