# Qwenny batched prompt prefill

This is an experimental checkpoint under `runtime-qwen38-pread-exp/` only.
Frozen Qwen3.8 runtimes, shortcuts, and other model implementations are
untouched.

## Change

The chat executable now exposes a generic `process_positions(K)` primitive.
For each streamed layer it:

1. rebinds that layer once through the explicit `pread()` provider;
2. submits a batch containing K token positions;
3. carries K hidden vectors to the next layer;
4. preserves recurrent/KV state once after the batch;
5. samples only from the final-position logits.

The previous implementation traversed all 64 layers independently for every
formatted prompt token.

## Verification

The formatted Qwen chat prompt for `Hi` completed with batched prefill and
reached assistant generation. A debug run reported:

```text
CORTEX_FINAL_LOGITS pos=8 positions=9 vocab=248320
top: 248068=36.246506 248046=21.767799 9419=21.032337 ...
CORTEX_SAMPLED token=248068
```

Token 248068 is the Qwen thinking-start marker, so the previous “no visible
token” behavior was partly prompt-prefill latency; the final-logit handoff is
now reached. The run used the same explicit `pread()` layer provider and the
same reusable arena.

The run was not yet accepted as a target/oracle equivalence proof: a matching
known-working final-logit capture from the frozen runtime is still required.
No speculative decoding or Symbiosis work has started.

## QWENNY BATCHED PREFILL CHECKPOINT: PASS

The frozen-library reference probe used the same batched source logic and the
same model blob. The exact formatted `Hi` prompt token sequence was:

```text
248045 846 198 12675 248046 198 248045 74455 198
```

Both paths reported final position `8`, `positions=9`, and vocabulary size
`248320`. Their complete final-logit FNV-1a bit-pattern hashes matched:

```text
Qwenny:  a2c22eddf893a6d4
Frozen:  a2c22eddf893a6d4
```

Top-N values also matched exactly at the printed precision:

```text
248068=36.246506 248046=21.767799 9419=21.032337
12675=19.718658 248058=17.863125 248045=16.786745
18103=15.656841 109266=15.555788
```

The argmax is token `248068`; EOS token `248046` is rank 2 with logit
`21.767799`. Neither vector contained NaN/Inf in the compared run. The exact
hash match establishes zero bit-pattern difference for the complete final
logit vector, so max absolute and relative error are both zero for this
checkpoint. The final hidden vector was not separately exported; final logits
are the stronger observable for the sampling handoff.

Checkpoint hashes:

```text
examples/qwen38-lowres-chat.cpp       2ead3ea93fee53fa82b6c191e26bf03405379b378214c90afe2f87a8d4976657
src/llama-model.cpp                   bc4543bb53029f9b1376d035d32d4faa89be794f79d72ffd6e7a0c69dea1f955
src/llama-model-loader.cpp             0ea7ce65bc6f3579221754a9066e5c509172e5b46ff7034c8b07d3def2e7a666
bin/qwenny                             5583312f0c6093289608474d8ff8ffa4f506a86ef11262e9fc6c783e8f91cc35
bin/llama-qwen38-lowres-chat           76ce88e917816284512a4af8d8b340b2e842cc8c4336377d78aa2ea09866512f
```

Symbiosis/speculative decoding has not been started. This checkpoint is
experimental and was not promoted over any existing runtime.

Build:

```bash
cmake --build runtime-qwen38-pread-exp-build \
  --target llama-qwen38-lowres-chat -j2
```

Normal interface:

```bash
printf 'Hi\n/exit\n' | QWEN38_MAX_GENERATE=1 \
  runtime-qwen38-pread-exp/bin/qwenny
```

Debug final-logit capture:

```bash
printf 'Hi\n/exit\n' | CORTEX_DEBUG=1 CORTEX_LOGIT_DEBUG=1 \
  QWEN38_MAX_GENERATE=1 runtime-qwen38-pread-exp/bin/qwenny
```
