# Contributing to Cortex Stream Forge

We welcome contributors with **any hardware**, including machines with much more RAM than the original reference laptop. The shared goal is correct, bounded-memory streaming inference; hardware-specific optimizations should remain optional.

## Where to help

1. **Correctness:** reproducible, byte-identical layer inputs/state and reference outputs; compare numerical error.
2. **Memory safety:** allocation accounting, bounded tensor residency, strict failure on unsupported configurations.
3. **I/O:** pread-based streaming, caching and storage benchmarks with byte counts.
4. **Portability:** CPU and future accelerated backends without changing model mathematics.
5. **Documentation and tests:** reproducible manifests, hardware reports and small isolated regression tests.

## Development workflow

- Open or claim an issue describing the smallest verifiable change.
- Branch from `main` and submit a draft PR early. Do not rewrite another contributor's branch.
- Keep the owned-layer streaming engine as the common execution path; avoid competing inference engines.
- Include model format and quantization, hardware, OS, memory/swap limits, exact command, source commit, observed memory peak, I/O, test output, and whether inference was actually executed.
- Never commit GGUF model weights, private files, credentials, or large captured activation payloads.
- Clearly distinguish **planned**, **compiled**, **executed**, **numerically verified**, and **end-to-end working**.
- Avoid making model-specific manifest totals universal for all layers.

## Resource safeguards

For the `potato-2gb` reference profile, actual-child verification must establish `memory.max=2147483648`, `memory.swap.max=0`, a bounded wall timeout and recorded OOM counters. Higher-resource contributors may use different limits but should report them.

Do not launch a high-risk inference test on someone else's computer without their approval. For PRs that only change planning or documentation, prefer non-executing tests.

## Reviews

Separate architectural changes from numerical validation and performance claims. Reference comparisons require matching model shards, exact inputs, initial state, positions, graph boundaries and source revision.

See [roadmap](docs/ROADMAP.md) and [hardware profiles](docs/HARDWARE_PROFILES.md).
