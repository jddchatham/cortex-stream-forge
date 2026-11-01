# FlashNext runtime audit

The audited runtime source is commit `dea54af133e4b71f159ea57b7d172c4c6b452ecf`.
The publication branch is `quortex-staged-runtime-dea54af`; the portability
fix is a subsequent commit on that branch.

## Build

```sh
cmake -S runtime-flashnext-layer-exp -B build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU=ON
cmake --build build --target llama-flashnext-two-range llama -j2
```

The inspected build used `/usr/bin/cc`, `/usr/bin/c++`, GCC 16.2.1, Release
mode, CPU backend enabled, tests/examples disabled, and backend loading disabled.

Inspected executable hash:
`13215449904d87d04b7fc48fc6ffa214878594d26622bb89898019c668af8029`

Available local `build-qies116` library hash:
`c0e614211a5f1bac9ff1db35303571b6ecbdc4f3249644e75efa9a8a51d009bd`

Previously deployed library pin:
`53df80306688a700aa02f5370c131b8e9064a1e139f7a7ad826463dd660de934`

These hashes are not proven equivalent to this source revision; rebuild and
hash artifacts independently.

## Runtime configuration and scope

The chat executable requires the explicit `FLASHNEXT_TENSOR_INDEX` environment
variable and has no machine-specific fallback. The trusted launcher should
validate and pass it explicitly.

The source contains staged Qwen4Exp execution, KV-input handling, input audit,
router/boundary capture, and staged-MoE seams. Stage-A execution/capture is
recorded in project history; complete Stage-B/Stage-C numerical equivalence,
final logits, and generation remain unproven.

The root project is MIT licensed. Vendored components retain their notices in
`gguf-py/LICENSE`, `licenses/`, and `vendor/*/LICENSE*`.
