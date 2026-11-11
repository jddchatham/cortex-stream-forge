#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Synthetic fixtures and saved guard-rejection evidence only; no real-model execution.
env -u LD_LIBRARY_PATH python3 -I -B qies100/run_comparison.py
env -u LD_LIBRARY_PATH python3 -I -B qies100/test_guard_adapter.py
