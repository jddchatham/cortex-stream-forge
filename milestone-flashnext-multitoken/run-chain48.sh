#!/bin/bash
# Bounded CHAIN_0_48 checkpoint runner (isolated diagnostic).
# Usage: run-chain48.sh <tag> <binary> <N_layers> <logfile>
set -u
TAG="$1"; BIN="$2"; N="$3"; LOG="$4"
DIR=/home/user/quortex-diag-20261103
MODEL=/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf
INDEX="$DIR/src/flashnext-tensor-index.json"
mkdir -p "$DIR/dump-$TAG" "$DIR/captures-$TAG"
start=$(date +%s%N)
CORTEX_FLASHNEXT_CHAIN_0_48=1 \
CORTEX_FLASHNEXT_CHAIN_0_N="$N" \
CORTEX_FLASHNEXT_CHAIN_TWO_TOKEN="${CHAIN_TWO_TOKEN:-0}" \
CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS=1 \
CORTEX_FLASHNEXT_DIAG_DUMP="$DIR/dump-$TAG" \
FLASHNEXT_COMPARE_DIR="$DIR/captures-$TAG" \
timeout 3600 "$BIN" "$MODEL" "$INDEX" > "$DIR/$TAG.stdout" 2> "$LOG" &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
    rss=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
    sleep 0.5
done
wait "$pid"; ec=$?
end=$(date +%s%N)
{
    echo "RUN_TAG=$TAG N=$N"
    echo "EXIT_STATUS=$ec"
    echo "ELAPSED_MS=$(( (end - start) / 1000000 ))"
    echo "PEAK_RSS_KB=$peak"
} | tee "$DIR/$TAG.meta"
exit "$ec"
