# QIES109 complete synthetic Ricursive layer

Run from this isolated worktree:

```sh
./pandacode-run.sh --smoke
./pandacode-run.sh
# Resume only an unfinished campaign, with identical sources/binaries:
./pandacode-run.sh --resume qies109/runs/CAMPAIGN
```

The driver verifies the preserved QIES107 63 captures/100 faults and QIES108
228 captures/105 faults against original files before new arithmetic. Compiler,
source, SDK and worker identities are saved per campaign. CPU-only fixed
procedural synthetic inputs are accepted; no model-weight reader exists.

The candidate streams 128 output rows, preserves complete dot-product dimensions,
and reuses one 1 MiB workspace across attention and FFN. At the branch boundary
all attention projection/graph buffers and attention-only activations are released;
only the FFN residual and tentative KV remain. Reference computation is one
independently assembled conventional stock-GGML full-layer graph.

The layer follows the pinned Ricursive order, including both attention and FFN
RMS/gamma, value mixing, Q/K norm/RoPE or NoPE, masks, softmax, gating, both
residual connections and the upstream clamped SwiGLU operator. Synthetic cases
represent actual layers 0 and 4 (Q5_K FFN), not every layer/quantization variant.

KV checkpoints have a new full-layer identity (109), incompatible with standalone
attention checkpoints. The input is read-only. A new checkpoint remains provisional
until attention, FFN, final output and serialization succeed. It is published with
all captures in an exclusive non-overwriting directory. Only successful, verified
results can feed the next fresh-process token. Faults leave input KV unchanged.
This diagnostic format is not a production durable multi-model checkpoint ABI.

The full campaign covers positions 0/1, SWA/global positions 1023–1025, repeated
continuation, 78 integration faults and malformed-state/interface refusals.
Boundary history is explicitly synthetic, not 1023 evaluated prefix tokens.
All bytes, shapes and eight weight identities must match; no tolerance fallback.

Requirements match QIES107–108: Linux bubblewrap, C++17, OpenSSL, Python and the
pinned source-built CPU SDKs. No NumPy, GPU, model download or global install.
QIES106_SDK_STATE may select an equivalent verified source-build identity file.
The real 6 GiB gate and unconditional real-execution lock remain unchanged.

Per-job receipts permit verified resumption after interruption. Incomplete attempts
are preserved under new names; completed campaigns refuse overwrite. Original
milestones are preserved in their worktrees and by qies109/preservation.json.
