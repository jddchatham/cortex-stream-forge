"""Read-only descriptor compatibility. Does not open any model tensor payload."""
from pathlib import Path
import json,subprocess,os,sys
r=Path(__file__).resolve().parent.parent;checks=r/'qies098-checks';src=r/'runtime-flashnext-layer-exp';env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
index=Path(sys.argv[1]) if len(sys.argv)>1 else checks/'ripple-tensor-index.provisional.json';obj=json.loads(index.read_text())
names=checks/'tensor-names.txt';names.write_text(''.join(t['name']+'\n' for t in obj['tensors']))
command=['c++','-std=c++17','-O2','-I'+str(src/'src'),str(r/'qies098-tools/index_compat.cpp'),str(src/'src/flashnext-tensor-index.cpp'),'-o',str(checks/'index-compat')]
subprocess.run(command,env=env,check=True,capture_output=True)
result=subprocess.check_output([str(checks/'index-compat'),str(index),str(names)],env=env,text=True,timeout=30)
lines=result.splitlines();assert len(lines)==len(obj['tensors'])==1203
for t,line in zip(obj['tensors'],lines):
 name,offset,size,kind,s0,s1=line.split('\t');shape=t['shape']
 assert (name,int(offset),int(size),int(kind),int(s0),int(s1))==(t['name'],t['offset'],t['bytes'],t['type'],shape[0],shape[1] if len(shape)>1 else 0)
(checks/'index-compat-result.json').write_text(json.dumps({'source':str(index),'descriptor_count':len(lines),'all_offsets_lengths_types_shapes_exact':True,'build_command':command,'tensor_payload_read':False,'does_not_validate_expert_stride_for_dense_model':True},indent=2)+'\n')
print('PASS 1203 named descriptors against unchanged baseline parser; no tensor data read')
