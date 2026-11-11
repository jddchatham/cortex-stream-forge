"""Model-free Ricursive single-layer manifest/admission checks. No native decode."""
from pathlib import Path
import json

GIB=1024**3
REQUIRED_SUFFIXES={'attn_gate.weight','attn_k.weight','attn_k_norm.weight','attn_norm.weight',
 'attn_output.weight','attn_q.weight','attn_q_norm.weight','attn_v.weight','attn_vres_lambda.weight',
 'ffn_down.weight','ffn_gate.weight','ffn_norm.weight','ffn_up.weight',
 'post_attention_norm.weight','post_ffw_norm.weight'}


def plan(index,layer=0,context=256):
 m=index['metadata']
 if m['general.architecture']!='ricursive' or m['ricursive.block_count']!=80:raise ValueError('Wrong architecture')
 expected={'ricursive.embedding_length':5120,'ricursive.feed_forward_length':40960,'ricursive.attention.head_count':80,'ricursive.attention.head_count_kv':8,'ricursive.attention.key_length':128,'ricursive.attention.value_length':128,'ricursive.attention.sliding_window':1024}
 if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong target dimensions')
 if not 0<=layer<80 or not 1<=context<=256:raise ValueError('Bounded test requires layer 0..79, context 1..256')
 pattern=m['ricursive.attention.sliding_window_pattern']
 if pattern!=[i%5!=4 for i in range(80)]:raise ValueError('Attention layout')
 prefix=f'blk.{layer}.';tensors=[t for t in index['tensors'] if t['name'].startswith(prefix)]
 if {t['name'][len(prefix):] for t in tensors}!=REQUIRED_SUFFIXES:raise ValueError('Missing/unreviewed layer tensor')
 shapes={'attn_gate.weight':[5120,10240],'attn_q.weight':[5120,10240],'attn_k.weight':[5120,1024],'attn_v.weight':[5120,1024],'attn_output.weight':[10240,5120],'attn_q_norm.weight':[128],'attn_k_norm.weight':[128],'attn_vres_lambda.weight':[1],'ffn_gate.weight':[5120,40960],'ffn_up.weight':[5120,40960],'ffn_down.weight':[40960,5120]}
 for name in ['attn_norm.weight','ffn_norm.weight','post_attention_norm.weight','post_ffw_norm.weight']:shapes[name]=[5120]
 if any(t['shape']!=shapes[t['name'][len(prefix):]] for t in tensors):raise ValueError('Layer tensor shape mismatch')
 kv_bytes=2*m['ricursive.attention.head_count_kv']*m['ricursive.attention.key_length']*context*2
 return {'kind':'Ricursive_SINGLE_LAYER_TEST_CONTRACT_ONLY','layer':layer,'attention':'sliding' if pattern[layer] else 'global',
  'n_ctx':context,'n_batch':1,'seq_id':0,'reference_commit':'240d68fcfe0c2a1f298c8e18ca181368b1914259',
  'weight_bytes':sum(t['bytes'] for t in tensors),'kv_f16_payload_bytes':kv_bytes,
  'kv_payload_assumes_one_active_layer':True,'full_80_layer_f16_payload_at_this_context':80*kv_bytes,
  'native_cache_allocation_not_measured':True,
  'input_hidden_f32_bytes':5120*4,'original_scaled_embedding_f32_bytes':5120*4,
  'gate_up_activation_f32_bytes_each':40960*4,'tensors':tensors,
  'mandatory_comparisons':['attn_norm','Qcur_normed','Kcur_normed','Qcur','Kcur','Vcur','attn_out','attn_gate','attn_gated','attn_post_norm','ffn_inp','ffn_norm','ffn_swiglu','ffn_out','ffn_post_norm','l_out','KV positions and bytes'],
  'inference_executed':False,'native_adapter_implemented':False}


def admission(mem_available,verified_download,qualified_reference,native_adapter):
 if mem_available<6*GIB:return False,'MemAvailable below unchanged 6 GiB gate'
 if not verified_download:return False,'Download not SHA256 verified'
 if not qualified_reference:return False,'Reference executable/runtime oracle not qualified'
 if not native_adapter:return False,'Isolated Ricursive layer execution adapter not implemented'
 return True,'Additional unchanged guard/cgroup/swap/disk checks still required before launch'


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('index',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 data=json.loads(args.index.read_text());plans=[plan(data,0),plan(data,4)]
 args.output.write_text(json.dumps({'plans':plans,'status':'PREPARED_NOT_EXECUTABLE; no native layer backend supplied'},indent=2)+'\n')
 print('Prepared bounded layer-0 SWA and layer-4 global contracts; no inference')
