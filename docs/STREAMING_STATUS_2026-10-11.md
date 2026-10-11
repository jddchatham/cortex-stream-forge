# Cortex Stream Forge — Development Status (2026-10-11)

## Latest verified milestone: bounded physical-layer traversal

A single supervised real-weight `[1,48)` transaction completed successfully on an AMD Athlon Silver 3050U with a 2 GiB memory cgroup and zero swap.

| Metric | Result |
| --- | --- |
| Executed physical layers | 1–47, exactly once (47 layers) |
| Activation handoffs | 46/46 byte-exact |
| Layer output shape | `[2560,4,1]` |
| Output validity | All finite |
| Transactions and state commits | All PASS |
| Peak observed cgroup memory | 1,301,467,136 bytes (~1.21 GiB) |
| Final observed memory | 694,145,024 bytes |
| Swap | 0 |
| OOM events | 0 |
| Wall time | 68.127 seconds |
| Final layer-47 output SHA-256 | `e8b30895616977e747762f8c09b428ada5e0e33a7c7fbc140ce9f1243f8a2506` |

Reported local evidence directory: `/tmp/cortex-fasttrack-48-evidence.EskT5m`. This path is on the development machine, not included in this repository.

Reported executable SHA-256: `1bd0e58e1ef5f1c1c274307cc7b7bf422c99f4895fd131ce304d081231232497`.

Reported linked library SHA-256: `49bde1a9a2d6790731bf3689b181c7584f5729e2df2acee49bf342dbbaf9557c`.

The earlier stale JSONL PLE evidence schema was traced to an executable target not rebuilt after the source change. A clean target rebuild corrected it. The evidence now records PLE rows provisioned, bytes read, and layer-conditional consumption metadata. Consumption metadata is **not** independent numerical validation.

## Current scope and limitations

**Proven:** bounded streaming execution of physical layers 1–47 using a synthetic HC activation fixture, with persistent per-layer evidence and byte-exact handoffs.

**Not yet proven:** end-to-end prompt-to-response inference, numerical equivalence to an independent full-model reference, or tokens-per-second throughput for real generation.

Still required:
1. Model tokenizer/chat template and genuine layer-0 token embedding path.
2. Complete physical `[0,48)` orchestration.
3. Final normalization, bounded vocabulary projection, token selection, and text decoding.
4. Correct recurrent/attention state preservation across prompt and generated tokens.
5. Independent numerical checks where feasible and multi-token interactive generation.

## Omarchy launcher

Reported local launcher:
- `/home/user/bin/cortex-stream-forge`
- `/home/user/.local/share/applications/cortex-stream-forge.desktop`

It uses `foot`, performs readiness checks, and does not misrepresent the current runtime as a working chatbot. No existing Hyprland shortcut was changed. These files are reported as created locally; this status update does **not** publish their source files.

## Repository synchronization notice

This document records results supplied from supervised local execution. It does **not** claim that the disposable worktree's latest C++ implementation, built binaries, test logs, or launcher scripts have been committed to GitHub. Those changes must be reviewed and pushed from the development worktree separately. Do not commit model shards or temporary evidence directories.
