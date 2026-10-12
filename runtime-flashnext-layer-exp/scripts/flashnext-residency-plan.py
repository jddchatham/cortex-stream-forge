#!/usr/bin/env python3
"""Metadata-only Qwen4Exp graph-required residency planner.

The family predicates mirror the tensor branches in qwen4exp.cpp and the
required set in llama-model.cpp; this script never reads GGUF tensor payloads.
"""
import argparse, json, re

ALIGN = 64
BUDGET = 768 * 1024 * 1024

def align(n): return (n + ALIGN - 1) // ALIGN * ALIGN

def plan_runtime_manifest(path):
    layers = {}
    seen = set()
    for line in open(path, encoding="utf-8"):
        if not line.startswith("CORTEX_DESCRIPTOR layer="):
            continue
        fields = dict(re.findall(r"(layer|name|bytes|storage)=([^ ]+)", line))
        layer = int(fields["layer"])
        name = fields["name"]
        key = (layer, name)
        if key in seen:
            raise ValueError(f"duplicate runtime descriptor: {layer} {name}")
        seen.add(key)
        if fields["storage"] != "ARENA_RESIDENT":
            continue
        layers.setdefault(layer, []).append((name, int(fields["bytes"])))
    if set(layers) != set(range(48)):
        raise ValueError("runtime manifest does not cover physical layers 0..47")
    out = {}
    for il in range(48):
        entries = sorted(layers[il])
        raw = sum(size for _, size in entries)
        out[il] = {
            "layer": il,
            "type": "attention" if (il + 1) % 4 == 0 else "recurrent",
            "count": len(entries),
            "bytes": raw,
            "aligned_bytes": sum(align(size) for _, size in entries),
            "largest": max(entries, key=lambda x: x[1]),
            "names": [name for name, _ in entries],
        }
    return out

def plan(index):
    tensors = {t["name"]: t for t in index["tensors"]}
    layers = {}
    for il in range(48):
        pref = f"blk.{il}."
        names = [n for n in tensors if n.startswith(pref)]
        recurrent = ((il + 1) % 4) != 0
        required = []
        for n in names:
            tail = n[len(pref):]
            common = tail.startswith("hc_") or tail in {
                "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp.weight",
                "attn_norm.weight"}
            if common:
                required.append(n); continue
            if recurrent:
                if tail in {"attn_qkv.weight", "attn_gate.weight", "ssm_a",
                    "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
                    "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight"}:
                    required.append(n)
            elif tail in {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                "indexer.q_proj.weight", "indexer.k_proj.weight",
                "indexer.q_norm.weight", "indexer.k_norm.weight"}:
                required.append(n)
            if il == 1 and tail in {"ple_key.weight", "ple_value.weight",
                "ple_norm_key.weight", "ple_norm_query.weight",
                "ple_norm_conv.weight", "ple_conv1d.weight"}:
                required.append(n)
        if len(required) != len(set(required)): raise ValueError(f"duplicate layer {il}")
        missing = [n for n in required if n not in tensors]
        if missing: raise ValueError(f"missing layer {il}: {missing}")
        sizes = [(n, int(tensors[n]["bytes"])) for n in required]
        raw = sum(b for _, b in sizes)
        layers[il] = {"layer": il, "type": "recurrent" if recurrent else "attention",
                      "count": len(required), "bytes": raw, "aligned_bytes": align(raw),
                      "largest": max(sizes, key=lambda x:x[1]), "names": required}
    return layers

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("index", nargs="?"); ap.add_argument("--runtime-manifest"); ap.add_argument("--json", action="store_true"); a=ap.parse_args()
    if bool(a.index) == bool(a.runtime_manifest):
        ap.error("provide exactly one of INDEX or --runtime-manifest")
    layers=plan_runtime_manifest(a.runtime_manifest) if a.runtime_manifest else plan(json.load(open(a.index)))
    maximum=max(layers.values(), key=lambda x:x["aligned_bytes"])
    out={"alignment":ALIGN,"budget":BUDGET,"layers":list(layers.values()),"maximum":maximum,
         "fits_budget":maximum["aligned_bytes"] <= BUDGET}
    print(json.dumps(out, indent=2) if a.json else "\n".join(f"{x['layer']:2} {x['type']:10} {x['count']:2} {x['aligned_bytes']:10} {x['largest'][0]}={x['largest'][1]}" for x in layers.values()))
    if not out["fits_budget"]: raise SystemExit(2)

if __name__ == "__main__": main()
