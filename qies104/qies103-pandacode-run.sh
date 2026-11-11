#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Readiness audit and synthetic-only stock reference tests; no real-weight arithmetic.
env -u LD_LIBRARY_PATH python3 -I -B qies103/contract.py
env -u LD_LIBRARY_PATH python3 -I -B qies103/audit.py
env -u LD_LIBRARY_PATH python3 -I -B qies103/compare_capsule.py
env -u LD_LIBRARY_PATH python3 -I -B qies103/tests.py
