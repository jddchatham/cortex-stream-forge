# Full-model arithmetic validation

This procedure is prepared for a host with sufficient RAM. It has not been run
on the current Athlon host.

## Build

Dependencies: CMake, Apple Clang or another C++17 compiler, pthreads, and the CPU ggml backend. The
recovery build is isolated in `recovery-build-4`.

```sh
cd /home/user/bitnet-cortex/cortex-stream-forge-recovery
cmake --build recovery-build-4 --target llama-flashnext-chat -j2
```

On Apple Silicon, use a fresh native build directory:

```sh
cmake -S runtime-flashnext-layer-exp -B recovery-build-m4 \
  -DGGML_METAL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build recovery-build-m4 --target llama-flashnext-chat -j$(sysctl -n hw.ncpu)
```

The current environment has not performed a macOS build or run, so this remains
an unverified compatibility step.

Model shards:

```text
/home/user/models/qwen38-flashnext-abliterated/
  Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf
  Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00002-of-00003.gguf
  Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00003-of-00003.gguf
```

The executable receives shard 0; GGUF metadata resolves the sibling shards.
Set `CORTEX_TENSOR_INDEX` to the absolute index path when running outside the
repository directory; otherwise the executable uses `flashnext-tensor-index.json`
from the current working directory.

## Prompt token sequence

The documented 22-token sequence is:

```text
248045 846 198 3710 369 220 17 478 220 17 30
21134 440 1132 279 1324 13 248046 198 248045 74455 198
```

The normal chat path must process all tokens causally and then produce exactly
one continuation token. The existing two/three-token diagnostic mode is only a
bounded preflight and is not the acceptance test.

## Memory guard

The chat executable checks the actual process after each layer teardown. Defaults
are deliberately conservative:

```text
CORTEX_CHAT_MAX_RSS_KIB=5242880
CORTEX_CHAT_MIN_AVAILABLE_KIB=2097152
```

Override only on a larger host, for example:

```sh
export CORTEX_CHAT_MAX_RSS_KIB=8388608
export CORTEX_CHAT_MIN_AVAILABLE_KIB=4194304
```

The guard prints `CHAT_MEMORY_GUARD` records and stops the run when either limit
is crossed. `CHAT_LIFECYCLE` records include RSS, VmHWM, VmSwap, and retained
state bytes. The rollback snapshot is released after committed state is copied
into `layer_states[il]` (`res.state_before.clear()` plus `shrink_to_fit()`).

## Controlled command

```sh
timeout --signal=TERM 1800s \
  recovery-build-4/bin/llama-flashnext-chat \
  /home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf 1 \
  > /tmp/cortex-full-chat.log 2>&1
```

The second argument is the generation limit and must be `1`, not the prompt
length. The current normal path is interactive: provide the documented arithmetic
user text at the `You:` prompt, then verify that the tokenizer emits the listed
22 IDs before accepting the result. A non-interactive harness that asserts the
22 IDs is still required before calling this a reproducible acceptance run.
The run must be supervised externally for the actual PID, systemd-oomd events,
and swap growth. Stop after one generated token; do not use the diagnostic layer
limit or run extended generation.

## Capacity requirement

The measured 24-layer/three-token run retained 2.835 GiB and peaked at 3.347
GiB. Linear scaling projects about 5.28 GiB retained state at 48 layers and
approximately 6.6–6.8 GiB transient peak. A practical minimum is 16 GiB RAM,
with at least 8 GiB available before launch and swap treated as a safety reserve,
not working memory. On a 16 GiB M4, begin with
`CORTEX_CHAT_MAX_RSS_KIB=7340032` and `CORTEX_CHAT_MIN_AVAILABLE_KIB=4194304`.
The current Athlon host does not meet the conservative available-memory
requirement.
