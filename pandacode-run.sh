#!/bin/sh
# Source-only handoff status; never loads weights or launches inference.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
unset LD_LIBRARY_PATH
exec python3 -I -B "$root/qies105/replay.py" inspect
