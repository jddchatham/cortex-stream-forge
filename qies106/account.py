"""Descriptor-derived sizes and lifetime estimates; does not open the model."""
from pathlib import Path
import json,hashlib,collections
R=Path(__file__).resolve().parent;W=R.parent
p=W/'qies098-checks/ripple-tensor-index.json';d=json.loads(p.read_text());stock={}
for line in (W/'qies104/reference-1203-descriptors.tsv').read_text().splitlines():
 n,*v=line.split('\t');stock[n]=list(map(int,v))
assert len(stock)==len(d['tensors'])==1203
for q in d['tensors']:assert stock[q['name']]==[q['offset'],q['bytes'],q['type'],*q['shape'],*[1]*(4-len(q['shape']))]
block={0:(1,4),8:(32,34),12:(256,144),13:(256,176),14:(256,210),23:(256,136)}
for q in d['tensors']:
 b,s=block[q['type']];n=1
 for x in q['shape']:n*=x
 assert n%b==0 and n//b*s==q['bytes'],q['name']
rows=[]
for q in d['tensors']:
 if not q['name'].startswith('blk.0.'):continue
 shape=q['shape'];n=shape[0];m=shape[1] if len(shape)>1 else 1;bs,bb=block[q['type']]
 rows.append(dict(q,plain_float_equivalent=n*m*4,row_bytes=n//bs*bb,tile_128_bytes=min(m,128)*(n//bs*bb),input_f32=n*4,output_f32=m*4))
kind=collections.defaultdict(list)
for q in d['tensors']:
 if q['name'].startswith('blk.'):
  kind[q['name'].split('.',2)[2]].append(q)
summary={n:{'types':dict(collections.Counter(q['quantization'] for q in v)),'min_bytes':min(q['bytes'] for q in v),'max_bytes':max(q['bytes'] for q in v),'all_layers_total':sum(q['bytes'] for q in v)} for n,v in kind.items()}
kv=[]
for c in (256,1024,4096,16384,32768,65536,2097152):
 kv.append(dict(context=c,per_layer_full_f16_kv=4096*c,all_80_full_f16_kv=80*4096*c,swa_64_windowed_plus_global_16=4096*(64*min(c,1024)+16*c),global_attention_scores_f32=80*c*4,global_transposed_v_f16=1024*c*2))
result={'model_sha256':d['model_sha256'],'index_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reference_descriptor_sha256':hashlib.sha256((W/'qies104/reference-1203-descriptors.tsv').read_bytes()).hexdigest(),'descriptor_cross_checks':1203,'basis':'calculated from archived verified descriptors; no model I/O and no measured graph allocator claim','layer0_weight_bytes':sum(q['bytes'] for q in rows),'layer0':rows,'all_layer_type_ranges':summary,'activation_bytes':{'hidden':5120*4,'x0':5120*4,'q':10240*4,'k':1024*4,'v':1024*4,'q_norm_weights':128*4,'k_norm_weights':128*4,'attention_gate':10240*4,'ffn_gate':40960*4,'ffn_up':40960*4,'ffn_swiglu':40960*4,'full_ffn_gate_up_swiglu_coexist':3*40960*4,'full_swiglu_plus_residual':(40960+5120)*4},'kv_scaling':kv,'tile_strategy':'output-row tiles only; full input/inner dimension retained, no partial sums/reduction-axis reassociation'}
(R/'memory-accounting.json').write_text(json.dumps(result,indent=2)+'\n')
cases=[('attn_q','blk.0.attn_q.weight'),('attn_k','blk.0.attn_k.weight'),('attn_output','blk.0.attn_output.weight'),('ffn_gate','blk.0.ffn_gate.weight'),('ffn_down','blk.0.ffn_down.weight')]
for role,lookup in [('candidate',{q['name']:[q['type'],q['shape'][0],q['shape'][1]] for q in rows if len(q['shape'])==2}),('reference',{n:[v[2],v[3],v[4]] for n,v in stock.items()})]:
 (R/(role+'-cases.inc')).write_text(''.join('{"'+label+'",'+','.join(map(str,lookup[n]))+'},\n' for label,n in cases))
print('PASS 1203 independent descriptor comparisons; generated exact real-width case catalogs and calculated memory accounting')
