#!/usr/bin/env bash
set -euo pipefail

# Cortex Stream Forge — 96 GB Bounded Streaming Chat Launcher
# Default mode: Clean normal chat (readable AI conversation, prompts, essential errors).
# Debug mode (--debug, --diag, -v): Full diagnostic logging including layer traversal,
# pread operations, residency checkpoints, and tensor metrics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORTEX_ROOT="${CORTEX_ROOT:-/home/user/bitnet-cortex/cortex-antigravity-development}"
BIN="${CORTEX_BIN:-$CORTEX_ROOT/build/bin/llama-qwen38-lowres-chat}"
MODEL="${CORTEX_MODEL:-/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf}"

if [[ ! -x "$BIN" ]]; then
    printf 'ERROR: Cortex streaming chat executable not found or not executable: %s\n' "$BIN" >&2
    printf 'Run: cmake --build %s/build --target llama-qwen38-lowres-chat\n' "$CORTEX_ROOT" >&2
    exit 1
fi

if [[ ! -r "$MODEL" ]]; then
    printf 'ERROR: Model shard not found or not readable: %s\n' "$MODEL" >&2
    exit 1
fi

DEBUG_MODE=0
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --debug|--diag|-v)
            DEBUG_MODE=1
            PASSTHROUGH_ARGS+=("$arg")
            ;;
        *)
            PASSTHROUGH_ARGS+=("$arg")
            ;;
    esac
done

if [[ "$DEBUG_MODE" -eq 1 ]]; then
    export CORTEX_DEBUG=1
    printf '==================================================\n'
    printf 'Cortex Stream Forge — DEBUG / DIAGNOSTIC MODE\n'
    printf 'Executable: %s\n' "$BIN"
    printf 'Model:      %s\n' "$MODEL"
    printf '==================================================\n\n'
    exec "$BIN" "$MODEL" "${PASSTHROUGH_ARGS[@]}"
else
    unset CORTEX_DEBUG || true
    unset CORTEX_DIAG || true
    unset CORTEX_VERBOSE || true
    exec "$BIN" "$MODEL" "${PASSTHROUGH_ARGS[@]}"
fi
