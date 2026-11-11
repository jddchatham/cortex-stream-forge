#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Synthetic arithmetic, real quantized byte storage, and readiness checks only.
# No real-weight numerical execution and no full-model inference.
env -u LD_LIBRARY_PATH python3 -I -B qies102/build.py
env -u LD_LIBRARY_PATH python3 -I -B qies102/validate.py
