#!/usr/bin/env bash
set -u

# Portable launcher template. Configure these paths in the environment or
# replace them when installing locally; no machine-specific path is embedded.
EXE="${CORTEX_FLASHNEXT_EXECUTABLE:-}"
MODEL="${CORTEX_FLASHNEXT_MODEL:-}"
INDEX="${CORTEX_FLASHNEXT_INDEX:-}"

printf 'Cortex Stream Forge\n'
if [[ -z "$EXE" || -z "$MODEL" || -z "$INDEX" ]]; then
  printf 'Status: NOT CONFIGURED\n'
  printf 'Set CORTEX_FLASHNEXT_EXECUTABLE, CORTEX_FLASHNEXT_MODEL and CORTEX_FLASHNEXT_INDEX.\n'
  exit 1
fi
if [[ ! -x "$EXE" || ! -r "$MODEL" || ! -r "$INDEX" ]]; then
  printf 'Status: NOT READY\n'
  printf 'Executable, model, or tensor index is unavailable.\n'
  exit 1
fi

printf 'Status: configured; prompt chat requires an explicitly reviewed command.\n'
printf 'No model is started automatically by this desktop entry.\n'
printf 'The bounded layer traversal is not presented as a chat application.\n'
read -r -p 'Press Enter to close.' _
