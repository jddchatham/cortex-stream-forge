#!/usr/bin/env bash
set -euo pipefail
index=${1:?index json}
model=${2:?main gguf shard}
shift 2
ids=("$@")
if (( ${#ids[@]} == 0 )); then ids=(136 77 317 257 226 142 175 414 373 131); fi
if (( ${#ids[@]} != 10 )); then exit 2; fi
expected=$(jq -r '.model_dir + "/" + (.tensors[] | select(.name=="blk.0.ffn_gate_exps.weight") | .shard)' "$index")
[[ "$model" == "$expected" ]] || exit 3
file_bytes=$(stat -c '%s' "$model")
get() { jq -r --arg n "$1" '.tensors[] | select(.name==$n) | [.offset,.bytes,.type] | map(tostring) | join(" ")' "$index"; }
read gate_base gate_bytes gate_type < <(get blk.0.ffn_gate_exps.weight)
read up_base up_bytes up_type < <(get blk.0.ffn_up_exps.weight)
read down_base down_bytes down_type < <(get blk.0.ffn_down_exps.weight)
read sg sg_bytes sg_type < <(get blk.0.ffn_gate_shexp.weight)
read su su_bytes su_type < <(get blk.0.ffn_up_shexp.weight)
read sd sd_bytes sd_type < <(get blk.0.ffn_down_shexp.weight)
[[ $gate_type == 23 && $up_type == 23 && $down_type == 20 ]] || exit 4
[[ $gate_bytes == 445644800 && $up_bytes == 445644800 && $down_bytes == 471859200 ]] || exit 5
for pair in "$gate_base $gate_bytes" "$up_base $up_bytes" "$down_base $down_bytes"; do read off len <<< "$pair"; (( off + len <= file_bytes )) || exit 6; done
gate_stride=$((gate_bytes/512)); up_stride=$((up_bytes/512)); down_stride=$((down_bytes/512))
echo "FLASHNEXT STATE PROBE CONFIG: PASS"
echo "index=$index model=$model layer=0 flags=CORTEX_FLASHNEXT_ROUTER_ONLY=1 CORTEX_PREAD_ARENA=1"
echo "gate=$gate_base bytes=$gate_bytes type=$gate_type stride=$gate_stride"
echo "up=$up_base bytes=$up_bytes type=$up_type stride=$up_stride"
echo "down=$down_base bytes=$down_bytes type=$down_type stride=$down_stride"
args=("$model" "$model")
for id in "${ids[@]}"; do (( id >= 0 && id < 512 )) || exit 7; args+=("$((gate_base+id*gate_stride))" "$((up_base+id*up_stride))" "$((down_base+id*down_stride))"); done
args+=("$sg" "$su" "$sd")
echo "resolved_ids=${ids[*]}"
printf 'resolved_command='
printf '%q ' "$PWD/../runtime-flashnext-layer-exp-build/bin/llama-flashnext-owned-layer" "${args[@]}"
echo
