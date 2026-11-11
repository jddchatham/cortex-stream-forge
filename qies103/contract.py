"""Deterministic activation/mask/zero-state contract, never golden model outputs."""
from pathlib import Path
import json,struct,math,hashlib
R=Path(__file__).resolve().parent;W=R.parent
MODEL='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
def sha(b):return hashlib.sha256(b).hexdigest()
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def write_exact(path,data):
 if path.exists():
  if path.read_bytes()!=data:raise ValueError('existing contract artifact differs')
 else:
  with path.open('xb') as f:f.write(data)
def generate(folder):
 folder=Path(folder);folder.mkdir(exist_ok=True)
 vector=struct.pack('<5120f',*[(i%17-8)/16 for i in range(5120)])
 mask=struct.pack('<256f',0,*([-math.inf]*255))
 zero=bytes(128*8*256*2*2)
 data={'input.f32':vector,'x0.f32':vector,'mask.f32':mask,'initial-kv.f16':zero}
 for name,b in data.items():write_exact(folder/name,b)
 shapes={'attn_norm':[5120,1],'vres_in':[5120,1],'Vcur':[128,8,1],'Qcur_normed':[128,80,1],'Kcur_normed':[128,8,1],'Qcur':[128,80,1],'Kcur':[128,8,1],'kq_soft_max':[256,1,80,1],'attn_out':[10240,1],'attn_gate':[10240,1],'attn_gated':[10240,1],'attn_post_norm':[5120,1],'ffn_inp':[5120,1],'ffn_norm':[5120,1],'ffn_swiglu':[40960,1],'ffn_out':[5120,1],'ffn_post_norm':[5120,1],'l_out':[5120,1],'layer-output':[5120,1]}
 outputs={name+'.f32':{'dtype':'F32_LE','shape':shape,'bytes':math.prod(shape)*4,'expected_sha256':None} for name,shape in shapes.items()}
 for name in ('key-written.f16','value-written.f16'):outputs[name]={'dtype':'F16_LE','shape':[128,8,1],'bytes':2048,'expected_sha256':None}
 bundle=json.loads((W/'qies102/bundle-manifest.json').read_text())
 result={'version':1,'status':'CONTRACT_ONLY_NO_GOLDEN_OUTPUTS','model_sha256':MODEL,'stock_source':'240d68fcfe0c2a1f298c8e18ca181368b1914259','layer':0,'position':0,'next_position':1,'sequence_id':0,'tokens':1,'context':256,'window':1024,'attention':'sliding_causal','hidden':5120,'ffn':40960,'heads':80,'kv_heads':8,'head_dimension':128,'rms_epsilon_f32':f32(1e-6),'rope_base_f32':10000.,'rope':'NEOX; position 0','activation_clamp_f32':20.,'attention_scale_f32':f32(1/f32(math.sqrt(128))),'backend':{'kind':'CPU','threads':1,'flash_attention':False,'use_extra_bufts':False,'kv_type':'F16','candidate_ggml':'0.25.3','stock_ggml':'0.29.0'},'input_semantics':'external already-scaled synthetic activation; input equals x0; no embedding lookup or token generation','inputs':{name:{'bytes':len(b),'sha256':sha(b)} for name,b in data.items()},'previous_kv_argument':'-','logical_kv_length':0,'unwritten_kv_comparison':'exclude; initialized zero for deterministic physical buffers','tensor_manifest_sha256':sha((W/'qies102/bundle-manifest.json').read_bytes()),'tensor_plan_sha256':bundle['plan_sha256'],'tensors':[{k:q[k] for k in ('name','shape','type','bytes','sha256')} for q in bundle['tensors']],'outputs':outputs,'comparison':'exact shape/dtype/byte count and every byte, zero tolerance; reject missing/nonfinite computed outputs; layer-output equals l_out','qualification':'reference and live guard pending; never infer qualification from this file'}
 payload=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();write_exact(folder/'contract.json',payload);return result
if __name__=='__main__':
 c=generate(R/'inputs');print('Prepared identical deterministic inputs and 21 output schemas; no golden outputs or inference')
