# QUORTEX — QIES-072: Native Non-Thinking First Answer

## Result

**Success.** The installed GGUF template supports native non-thinking mode. QUORTEX and stock llama.cpp evaluated the same 26-token formatted prompt and then produced the same answer and stop decision:

- Generated answer: **`4`**
- QUORTEX generated token IDs: `19`
- Stock generated token IDs: `19`
- Stop: the next-token argmax was native EOG/EOS ID `248046`; the 16-token cap was not reached.
- All 26 prompt-prefix logits vectors, the answer-selection vector, and the post-answer EOG-selection vector matched stock byte-for-byte: **28/28 comparisons exact**. There were zero differing elements, max absolute error 0, RMS error 0.

The rendered output is `4` followed by the CLI’s display newline. It contains no `<think>` trace.

## Native template evidence

The model is `Qwen3.8-Flash-Next-Uncensored-IQ4_XS`, with a 248,320-token vocabulary and EOS ID 248046. Its embedded `tokenizer.chat_template` explicitly uses the `enable_thinking` variable:

- When `enable_thinking` is undefined or true, the template may add reasoning-effort instructions to a system message.
- For an assistant generation prompt, `enable_thinking=false` emits an empty think block, `<think>\n\n</think>\n\n`, after the assistant header. The answer is generated after that closed block.

The installed llama C helper `llama_chat_apply_template` exposes only the fixed-template interface; it cannot pass Jinja variables such as `enable_thinking`. The isolated CLI therefore checks that the loaded model template contains the `enable_thinking` branch and renders the audited text-only, user-only false branch verbatim. This changes only prompt formatting. It does not filter generated tokens or alter logits, weights, sampling, or inference arithmetic. There was no system message in this test; that is the GGUF template’s normal user-only path.

## Exact prompt and tokens

User text:

> What is 2 + 2? Answer with only the number.

Exact formatted prompt (newlines shown literally):

```text
<|im_start|>user
What is 2 + 2? Answer with only the number.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Formatted prompt length: 112 bytes. Both implementations tokenized it to these same IDs:

```text
248045,846,198,3710,369,220,17,478,220,17,30,21134,440,1132,279,1324,13,248046,198,248045,74455,198,248068,271,248069,271
```

| Position | Input token ID | Stock logits vs QUORTEX | Position | Input token ID | Stock logits vs QUORTEX |
|---:|---:|:---:|---:|---:|:---:|
| 0 | 248045 | Exact | 13 | 1132 | Exact |
| 1 | 846 | Exact | 14 | 279 | Exact |
| 2 | 198 | Exact | 15 | 1324 | Exact |
| 3 | 3710 | Exact | 16 | 13 | Exact |
| 4 | 369 | Exact | 17 | 248046 | Exact |
| 5 | 220 | Exact | 18 | 198 | Exact |
| 6 | 17 | Exact | 19 | 248045 | Exact |
| 7 | 478 | Exact | 20 | 74455 | Exact |
| 8 | 220 | Exact | 21 | 198 | Exact |
| 9 | 17 | Exact | 22 | 248068 | Exact |
| 10 | 30 | Exact | 23 | 271 | Exact |
| 11 | 21134 | Exact | 24 | 248069 | Exact |
| 12 | 440 | Exact | 25 | 271 | Exact |

Every vector contains 248,320 little-endian F32 values (993,280 bytes). Stock and QUORTEX matched for each prompt position. Greedy argmax IDs also match at every position.

| Decode boundary | Stock reference | QUORTEX | Result |
|---|---|---|---|
| Final prompt logits, position 25 | `logits-select-25.bin` | `logits-select-0.bin` | Byte-exact; argmax 19 |
| After input answer token 19, position 26 | `logits-generated-1.bin` | `logits-select-1.bin` | Byte-exact; argmax 248046 (EOG) |

The first generated token (19) decodes natively to `4`. The next logits select token 248046, which `llama_vocab_is_eog` classifies as end-of-generation. The CLI stops before appending or displaying that control token. Stock and QUORTEX both report one generated token.

## Runs and provenance

Canonical repository: `QIES-065 baseline repository`, branch `feature/flashnext-continuous-generation`, HEAD `59a13b1dfbe020ace12a5b209eea4b2d657a807a`. Isolated CLI branch: `feature/flashnext-first-text-generation`, same HEAD. Existing uncommitted work and QIES-065 through QIES-071 artifacts were preserved. No canonical source file was changed; no commit or push was made.

The first candidate attempt was deliberately rejected and stopped: I had compiled the older worktree CLI instead of the QIES-068/069 validated overlay. That source lacked the dedicated transaction residual/injection and per-layer router-logit handoffs; its first prompt-position logits differed. Its run is preserved at `qies072-quortex-20261106a` and is not included in the 28 valid comparisons. The corrected candidate below was built from the validated QIES-068 overlay plus only the QIES-072 prompt-format change, then run in a fresh directory.

- Published QUORTEX CLI source SHA256: `3e78f699b92a21f199b01274adaf50f5e88b69729359eab002424e988339e8d0`
- Source overlay used for the recorded QIES-072 run SHA256: `54ee59c1a7ef1e73c6d5999b05ea8a3f4a5e7182dba2b11c3a91b6f0754dc876` (same generation logic; publication changes only the fallback tensor-index path)
- Validated QUORTEX executable used for the recorded run SHA256: `2e4971e10f6cf56454252065ff0663c5e30687f4f4f3fa259d4210a5081dafec` (binary excluded from publication)
- Stock runner source included in this publication SHA256: `e169159b26d68409c178bc8fbe496fc5582e240c8c9ff6804e78ac2385c8b35f`
- Validated stock executable used for the recorded run SHA256: `eda8ceefc94f296a66d217b79d0aba64317b5f24295b822e086e7d3736169fb2` (binary excluded from publication)
- **Provenance limitation:** the original run note recorded a different stock source-overlay digest (`4faa9bd2…`); that exact source snapshot is not present in the preserved overlay directory. The stock executable digest is preserved, and the included stock runner implements the same tested sequential prompt and non-thinking-template protocol, but was not rebuilt and rerun during publication.
- Tensor index SHA256: `bc78052954dc3ab6d37ad189a2643c68cc8ad6ecfab5a6bb76361dc0b94f22c7`
- Model shard hashes (same model identity recorded in QIES-067/071):
  - `Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf`: `28a99b125cb905fc3bdc06baf6266a56b0cbd1b27e833b9142acc2017b673704`
  - `Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00002-of-00003.gguf`: `2a309e0b112fde96ba3bcba5a6b58cc05e5df7bb7fad5a990eaa51df335b0e43`
  - `Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00003-of-00003.gguf`: `ebc43c58e2eaeba1d5bdf62c8cb1f0eac198c4dc01941f771921edeebf574bc3`

The stock oracle was a fresh CPU-only, no-diagnostic, sequential stock run. The candidate used a CPU-only `GGML_CPU_REPACK=ON` build, `CORTEX_FLASHNEXT_LAYER_RANGE=1`, loader cache ON, expert arena 1,048,576 bytes, and `OMP_NUM_THREADS=1`. The stock runner build was the preserved stock configuration; REPACK ON/OFF equivalence was previously validated in QIES-071. Each run used a fresh state/output directory and the same model shard, index, prompt, tokenization, positions, sequence, and greedy argmax.

Run directories (preserved):

- Stock: `QIES072 stock run output (retained locally; not published)`
- Corrected QUORTEX: `QIES072 corrected QUORTEX run output (retained locally; not published)`

## Resource and timing results

The independent sampler ran at approximately two-second intervals. Guards were process RSS ≥8 GiB, host MemAvailable <1.5 GiB, disk free <10 GiB, or any cgroup `oom_kill` increment. None triggered. The stock run took about 745 seconds; corrected QUORTEX completed in 2,238 seconds (37m 18s wall-clock including external monitoring).

| Metric | Stock oracle | Corrected QUORTEX |
|---|---:|---:|
| Peak process RSS | 6,278,608 KiB (5.99 GiB) | 6,148,788 KiB (5.86 GiB) |
| Peak process swap | 217,628 KiB (0.21 GiB) | 116,468 KiB (0.11 GiB) |
| Peak cgroup memory | 7,111,434,240 B (6.62 GiB) | 6,935,883,776 B (6.46 GiB) |
| Peak cgroup swap | 1,151,762,432 B (1.07 GiB) | 1,087,991,808 B (1.01 GiB) |
| Lowest sampled host MemAvailable | 4,939 MiB | 5,278 MiB |
| Lowest sampled free disk | 142.3 GiB | 138.9 GiB |
| Major-fault count at end of sampled run | 183,538 | 94,373 |
| Process read-bytes delta | Not captured by stock sampler | 137,724,358,656 B |
| Process write-bytes delta | Not captured by stock sampler | 156,995,366,912 B |
| cgroup `oom_kill` | 0 | 0 |

QUORTEX state files total **5,705,790,144 bytes** (about 5.71 GB decimal); capture files total about **1.5 GiB**. The state footprint is disk-backed and should not be read as resident memory. Candidate timing from the CLI: vocabulary load 1.034 s, prefill 2,151.130 s, first generated token latency 2,152.168 s, generation interval 84.724 s, total 2,236.889 s. Only one answer token was generated, so there is no meaningful sustained tokens/sec figure.

## Reproduction command

This exact command shape runs the tested prompt with a new state and output directory; the binary and model paths are those used for this isolated validation:

```sh
# Build from the repository root; set these to the local model and tensor-index files.
export QUORTEX_MODEL=/path/to/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf
export QUORTEX_TENSOR_INDEX="$PWD/flashnext-tensor-index.json"
export QUORTEX_BIN="$PWD/build-quortex/bin/quortex"
run_dir=$(mktemp -d /tmp/quortex-question.XXXXXX)
CORTEX_FLASHNEXT_LOADER_CACHE=1 \
CORTEX_FLASHNEXT_EXPERT_ARENA_BYTES=1048576 \
CORTEX_FLASHNEXT_LAYER_RANGE=1 \
OMP_NUM_THREADS=1 \
  "$QUORTEX_BIN" \
  --model "$QUORTEX_MODEL" \
  --tensor-index "$QUORTEX_TENSOR_INDEX" \
  --prompt "What is 2 + 2? Answer with only the number." \
  --max-tokens 16 \
  --state-dir "$run_dir/state" \
  --logits-dir "$run_dir/logits"
```

The binary is an isolated diagnostic build, not an installed system command. The CLI’s current non-thinking renderer is guarded against a model template that lacks the verified branch; this path is validated for the model’s simple text-only user message. General multi-message, tool, or multimodal template rendering with arbitrary Jinja variables remains a separate integration task.

## Decision

QIES-072’s first-answer objective is met: native non-thinking template formatting produced the direct answer `4`, with exact stock logits and token agreement through the EOG decision. No approximate comparison, output filtering, or inference-math change was used.


## Build instructions

From the repository root, with the model shards and generated tensor index available locally:

```sh
cmake -S runtime-flashnext-layer-exp -B build-quortex \
  -DGGML_CUDA=OFF -DGGML_CPU_REPACK=ON \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-quortex --target llama-flashnext-chat -j2
```

The target emits `build-quortex/bin/quortex`. A fresh out-of-tree configure and build of this exact target passed during publication review (built binary SHA256 `9a7173d7c76bbf21d53502e3e86212074888c1e4033195238a350ecc109873f7`). The model and tensor index are not distributed in this repository. This milestone validates the text-only single-user prompt path on the tested model; general multi-message or multimodal Jinja template evaluation is not claimed. The run took about 37 minutes for the tested prompt on the reference CPU host, so this is a correctness milestone rather than an interactive-speed claim.
