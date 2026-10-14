#!/usr/bin/env bash
set -euo pipefail

MODEL="${QWEN38_27B_MODEL:-/home/user/.ollama/models/blobs/sha256-6c2c13cef89238c3604d756b07b3ef5fafebbd61095feb8553ff449c95e4c1c6}"
RUNNER="/home/user/bitnet-cortex/cortex-stream-forge/runtime-qwen38-pread-exp/bin/qwenny"

# Forward debug/diag flags
DEBUG_MODE=0
for arg in "$@"; do
    case "$arg" in
        --debug|--diag)
            DEBUG_MODE=1
            ;;
    esac
done

if [[ "$DEBUG_MODE" -eq 1 ]]; then
    export CORTEX_DEBUG=1
fi

exec "$RUNNER"
