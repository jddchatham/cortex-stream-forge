"""One-layer stock-reference transformation, synthetic weights ONLY."""
from pathlib import Path
import types,sys,json
R=Path(__file__).resolve().parent;W=R.parent
pkg=types.ModuleType('gguf');pkg.__path__=[str(W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259/gguf-py/gguf')];sys.modules['gguf']=pkg
from gguf.gguf_writer import GGUFWriter
from gguf.constants import GGMLQuantizationType
import numpy as np
for label in ('fixture','fixture-quantized'):
 source=W/'qies100'/label;out=R/('oneblock-'+label);out.mkdir(exist_ok=True)
 w=GGUFWriter(out/'synthetic.gguf','ricursive')
 for fn,arg in [('add_context_length',256),('add_embedding_length',256),('add_feed_forward_length',512),('add_block_count',1),('add_head_count',4),('add_head_count_kv',2),('add_key_length',64),('add_value_length',64),('add_rope_dimension_count',64),('add_rope_freq_base',10000.),('add_layer_norm_rms_eps',1e-6),('add_sliding_window',2),('add_sliding_window_pattern',[True]),('add_embedding_scale',1.),('add_vocab_size',32),('add_tokenizer_model','none')]:getattr(w,fn)(arg)
 w.add_swiglu_clamp([20.])
 shapes=json.loads((source/'shapes.json').read_text())
 for name,shape in shapes.items():
  if name.startswith('blk.') and not name.startswith('blk.0.'):continue
  file=source/(name+'.bin');typefile=source/(name+'.type')
  if typefile.exists():
   typ=GGMLQuantizationType(int(typefile.read_text()));raw=np.fromfile(file,dtype=np.uint8).reshape(shape[0],-1);w.add_tensor(name,raw,raw_dtype=typ)
  else:w.add_tensor(name,np.fromfile(file,dtype=np.float32).reshape(shape))
 w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
print('Prepared two one-block SYNTHETIC stock fixtures; no real weights accessed')
