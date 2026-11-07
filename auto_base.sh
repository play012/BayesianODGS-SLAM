#!/usr/bin/env bash
# Run the omnidirectional BayesianGS-SLAM pipeline.
#
# Pass one or more scene configs; defaults to the example panoramic scene.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- configs/Omni/example.yaml
fi

for cfg in "$@"; do
  python run_slam.py "$cfg" --keyframe_count 4
done
