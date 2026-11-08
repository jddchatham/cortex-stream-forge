#!/bin/sh
set -eu
: "${BASELINE_ROOT:?Set BASELINE_ROOT to the recreated baseline checkout}"
: "${CANDIDATE_ROOT:?Set CANDIDATE_ROOT to the patched worktree}"
: "${BUILD_OUTPUT:?Set BUILD_OUTPUT to a new output directory}"
exec python3 -I -B "$(dirname "$0")/tools/build_pair.py" --baseline "$BASELINE_ROOT" --candidate "$CANDIDATE_ROOT" --output "$BUILD_OUTPUT"
