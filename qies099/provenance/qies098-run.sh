#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Analysis only. Download is separate and explicitly user-authorized; this never launches inference.
env -u LD_LIBRARY_PATH python3 -I -B qies098-tools/finalize_metadata.py
env -u LD_LIBRARY_PATH python3 -I -B -W error::ResourceWarning qies098-tools/test_metadata.py
env -u LD_LIBRARY_PATH python3 -I -B qies098-tools/test_contract.py
