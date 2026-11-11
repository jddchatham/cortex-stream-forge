#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Model-free CPU build and validation only; never invokes real-model execution.
env -u LD_LIBRARY_PATH python3 -I -B qies099/build.py
env -u LD_LIBRARY_PATH python3 -I -B qies099/prepare_plans.py
for plan in qies099/checks/layer-*.plan; do
    env -u LD_LIBRARY_PATH qies099/checks/ricursive-layer --describe "$plan"
done
