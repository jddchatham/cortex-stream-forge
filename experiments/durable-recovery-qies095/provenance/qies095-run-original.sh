#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Model-free only. Each invocation preserves all previous milestone/test artifacts.
run_dir=$(mktemp -d "$PWD/qies095-checks/complete-XXXXXX")
mkdir "$run_dir/qies094-regression" "$run_dir/replay"
env -u LD_LIBRARY_PATH python3 -I -B qies095-preserved/qies094-prototype/test_recovery.py "$run_dir/qies094-regression" > "$run_dir/qies094-regression/tests.log" 2>&1
env -u LD_LIBRARY_PATH python3 -I -B qies095-prototype/test_replay.py "$run_dir/replay" > "$run_dir/replay/tests.log" 2>&1
cat "$run_dir/qies094-regression/tests.log"
cat "$run_dir/replay/tests.log"
printf 'Results: %s\n' "$run_dir"
