#!/bin/bash
set -euo pipefail

MODEL_BLOB="/home/user/.ollama/models/blobs/sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5"
INCLUDE_DIR="/home/user/bitnet-cortex/cortex-antigravity-development/runtime-flashnext-layer-exp"
BIN_DIR="/home/user/bitnet-cortex/cortex-stream-forge/runtime-flashnext-layer-exp-build/bin"

echo "=== Compiling Cortex Small-Model Streaming Engine ==="
g++ -O3 cortex-small-streaming.cpp \
  -I "${INCLUDE_DIR}/include" \
  -I "${INCLUDE_DIR}/src" \
  -I "${INCLUDE_DIR}/ggml/include" \
  -L "${BIN_DIR}" \
  -lllama -lggml -lggml-base -lggml-cpu \
  -Wl,-rpath,"${BIN_DIR}" \
  -o cortex-small-streaming

echo "=== Executing Cortex Small-Model Streaming Engine ==="
./cortex-small-streaming "${MODEL_BLOB}"
