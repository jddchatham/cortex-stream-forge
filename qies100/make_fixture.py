"""Synthetic-only five-layer GGUF, no downloaded model reads."""
from pathlib import Path
import sys,json,subprocess
R=Path(__file__).resolve().parent
sys.path.insert(0,str(next(R.glob('llama.cpp-*'))/'gguf-py'))
import types
pkg=types.ModuleType("gguf");pkg.__path__=[str(next(R.glob("llama.cpp-*"))/"gguf-py/gguf")];sys.modules["gguf"]=pkg
from gguf.gguf_writer import GGUFWriter
import numpy as np
quantized='--quantized' in sys.argv
out=R/('fixture-quantized' if quantized else 'fixture');out.mkdir(exist_ok=True)
w=GGUFWriter(out/'synthetic.gguf','ricursive')
for fn,arg in [('add_context_length',256),('add_embedding_length',256),('add_feed_forward_length',512),('add_block_count',5),('add_head_count',4),('add_head_count_kv',2),('add_key_length',64),('add_value_length',64),('add_rope_dimension_count',64),('add_rope_freq_base',10000.),('add_layer_norm_rms_eps',1e-6),('add_sliding_window',2),('add_sliding_window_pattern',[True,True,True,True,False]),('add_embedding_scale',1.),('add_vocab_size',32),('add_tokenizer_model','none')]:getattr(w,fn)(arg)
w.add_swiglu_clamp([20.]*5)
spec={}
def add(n,shape,vector=False):
 a=(np.ones(shape,dtype=np.float32) if vector else (np.sin(np.arange(np.prod(shape),dtype=np.float64)*.317+1)*.03).astype(np.float32).reshape(shape))
 if 'lambda' in n:a.fill(0)
 a.tofile(out/(n+'.f32'));spec[n]=list(shape)
 if quantized and len(shape)==2:
  typ=23 if n=='token_embd.weight' else 14 if 'output' in n else 8 if any(k in n for k in ('attn_k.','attn_v.')) else 13 if 'ffn_' in n else 12
  subprocess.run([str(R/'quantize-fixture'),str(typ),str(shape[1]),str(shape[0]),str(out/(n+'.f32')),str(out/(n+'.bin')),str(out/(n+'.decoded'))],check=True)
  raw=np.fromfile(out/(n+'.bin'),dtype=np.uint8).reshape(shape[0],-1)
  from gguf.constants import GGMLQuantizationType
  w.add_tensor(n,raw,raw_dtype=GGMLQuantizationType(typ));(out/(n+'.type')).write_text(str(typ))
 else:
  w.add_tensor(n,a);a.tofile(out/(n+'.bin'))
add('token_embd.weight',(32,256));add('output_norm.weight',(256,),True);add('output.weight',(32,256))
for i in range(5):
 for n,shape in {'attn_norm':(256,),'attn_q':(256,256),'attn_k':(128,256),'attn_v':(128,256),'attn_q_norm':(64,),'attn_k_norm':(64,),'attn_vres_lambda':(1,),'attn_gate':(256,256),'attn_output':(256,256),'post_attention_norm':(256,),'ffn_norm':(256,),'ffn_gate':(512,256),'ffn_up':(512,256),'ffn_down':(256,512),'post_ffw_norm':(256,)}.items():add(f'blk.{i}.{n}.weight',shape,len(shape)==1)
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
(out/'shapes.json').write_text(json.dumps(spec,indent=2))
