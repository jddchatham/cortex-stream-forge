# Cortex Stream Forge

**CPU-first, bounded-memory streaming inference for models larger than available RAM.**

Our immediate engineering target is the approximately 97.5 GB Qwen3.8 Flash-Next Abliterated IQ4_XS GGUF model. The goal is to execute the full model by streaming weights from storage while keeping activations and necessary state resident within a configurable memory budget.

## Project status

**Experimental — not yet a working full-model inference runtime.**

- Verified bounded tensor-selection planning exists for layer ranges `[0,1)` and `[1,2)` in [draft PR #3](https://github.com/jddchatham/cortex-stream-forge/pull/3).
- The next milestone is a byte-identical, numerically verified owned-layer transaction under a verified memory gate.
- Full 48-layer execution, multi-token decoding, and tokens/second performance are **not yet demonstrated**.

## One engine, many computers

Hardware profiles configure resource constraints and test expectations. They do not fork the numerical implementation.

| Profile | Intended hardware | Status |
| --- | --- | --- |
| `potato-2gb` | Low-end x86 CPU, 2 GiB memory, zero swap | Reference safety target |
| `desktop-16gb` | Typical CPU desktop with more RAM | Proposed, not validated |
| `workstation` | Larger RAM and future accelerators | Proposed, not validated |

The 2 GiB constraint is mandatory for the **reference low-memory regression**, not for every contributor's machine. See [hardware profiles](docs/HARDWARE_PROFILES.md).

## How to contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](docs/ROADMAP.md). Help is especially welcome with GGUF tensor selection, disk I/O, numerical reference capture, owned-layer execution, memory accounting, and cross-machine benchmarking.

**Safety:** Do not run the approximately 97.5 GB model through an unbounded loader on memory-constrained hardware. Never infer that a successful `--plan` proves full inference.

## Current work

- [Issue #1 — streaming inference coordination](https://github.com/jddchatham/cortex-stream-forge/issues/1)
- [PR #3 — bounded selection planning](https://github.com/jddchatham/cortex-stream-forge/pull/3)

## License

See the repository's license and upstream dependency licenses before redistribution. This README does not grant rights to model weights or third-party code.
