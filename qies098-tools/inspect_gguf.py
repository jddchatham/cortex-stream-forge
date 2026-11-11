"""Bounded GGUF metadata/tensor-directory reader. Does not read tensor payloads."""
from pathlib import Path
import argparse,collections,ctypes,hashlib,json,math,struct

class Reader:
 def __init__(self,path):
  self.f=open(path,'rb');self.limit=64*1024*1024
 def read(self,n):
  if n<0 or self.f.tell()+n>self.limit:raise ValueError('Metadata budget exceeded')
  b=self.f.read(n)
  if len(b)!=n:raise EOFError('Header incomplete; retry after more download')
  return b
 def number(self,fmt):return struct.unpack('<'+fmt,self.read(struct.calcsize('<'+fmt)))[0]
 def string(self,keep=True):
  n=self.number('Q')
  if n>8*1024*1024:raise ValueError('String bound')
  if keep:return self.read(n).decode('utf8')
  self.read(n);return None
 def value(self,t,keep,depth=0):
  if depth>1:raise ValueError('Nested array unsupported')
  formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
  if t in formats:return self.number(formats[t])
  if t==8:return self.string(keep)
  if t!=9:raise ValueError('Unknown metadata type')
  subtype=self.number('I');n=self.number('Q')
  if n>1000000:raise ValueError('Array count bound')
  retain=keep and n<=8192
  result=[] if retain else None
  for _ in range(n):
   value=self.value(subtype,retain,depth+1)
   if retain:result.append(value)
  return result


def inspect(path,expected_size,ggml_lib):
 r=Reader(path)
 try:
  if r.read(4)!=b'GGUF':raise ValueError('GGUF magic')
  version=r.number('I')
  if version!=3:raise ValueError('Expected GGUF v3')
  count=r.number('Q');kv=r.number('Q')
  if count>10000 or kv>10000:raise ValueError('Header count bound')
  metadata={}
  for _ in range(kv):
   key=r.string();t=r.number('I')
   if key in metadata:raise ValueError('Duplicate metadata key')
   keep=not key.startswith('tokenizer.') or t!=9
   value=r.value(t,keep)
   metadata[key]=value if keep else '<omitted by bounded metadata reader>'
  lib=ctypes.CDLL(str(ggml_lib));lib.ggml_blck_size.argtypes=[ctypes.c_int];lib.ggml_blck_size.restype=ctypes.c_longlong;lib.ggml_type_size.argtypes=[ctypes.c_int];lib.ggml_type_size.restype=ctypes.c_size_t
  tensors=[];names=set()
  for _ in range(count):
   name=r.string();ndim=r.number('I')
   if name in names or not 1<=ndim<=4:raise ValueError('Tensor identity/dimensions')
   names.add(name);shape=[r.number('Q') for _ in range(ndim)];dtype=r.number('I');offset=r.number('Q')
   if dtype not in (0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30):raise ValueError('Unreviewed quant type: '+str(dtype))
   if any(n<=0 or n>2**32 for n in shape):raise ValueError('Tensor dimension bound')
   block=lib.ggml_blck_size(dtype);unit=lib.ggml_type_size(dtype)
   if not block or not unit or shape[0]%block:raise ValueError('Quantized row alignment')
   size=math.prod(shape)//block*unit
   tensors.append({'name':name,'shape':shape,'type':dtype,'relative_offset':offset,'bytes':size})
  end=r.f.tell();alignment=metadata.get('general.alignment',32)
  if type(alignment) is not int or alignment<1 or alignment>4096 or alignment&(alignment-1):raise ValueError('Invalid alignment')
  start=(end+alignment-1)//alignment*alignment
  intervals=[]
  for t in tensors:
   if t['relative_offset']%alignment:raise ValueError('Tensor offset alignment')
   t['offset']=start+t['relative_offset']
   if t['offset']+t['bytes']>expected_size:raise ValueError('Tensor outside complete file size')
   intervals.append((t['offset'],t['offset']+t['bytes']))
  intervals.sort()
  if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('Overlapping tensor payloads')
  per_layer=collections.Counter();types=collections.Counter()
  for t in tensors:
   types[t['type']]+=1
   if t['name'].startswith('blk.'):per_layer[int(t['name'].split('.')[1])]+=t['bytes']
  return {'source':str(path),'expected_complete_bytes':expected_size,'version':version,'tensor_count':count,'metadata_count':kv,'metadata':metadata,'data_offset':start,'metadata_bytes_read_limit':r.limit,'tensor_payload_read':False,'quant_type_counts':dict(types),'layer_weight_bytes':dict(per_layer),'max_layer_weight_bytes':max(per_layer.values()),'tensors':tensors}
 finally:
  r.f.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('file',type=Path);p.add_argument('--expected-size',type=int,required=True);p.add_argument('--ggml-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=inspect(a.file,a.expected_size,a.ggml_library);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('tensor_count','metadata_count','data_offset','quant_type_counts','max_layer_weight_bytes','tensor_payload_read')}))
