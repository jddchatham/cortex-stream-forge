"""Inspect only the header of the fully hash-verified file; never run model math."""
from pathlib import Path
import json,ctypes,os,sys,subprocess,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inspect_gguf import inspect
from layer_contract import plan
r=Path(__file__).resolve().parent.parent;data=r.parent/'qies098-data';checks=r/'qies098-checks'
receipt=json.loads((data/'metadata/download-result.json').read_text())
assert receipt['status']=='VERIFIED' and receipt['sha256']==receipt['expected_sha256']=='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
model=Path(receipt['path']);assert model.stat().st_size==receipt['bytes']==40426079712
identity_path=checks/'model-file-identity.json'
s=model.stat()
if identity_path.exists():
 old=json.loads(identity_path.read_text())
 assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(old['device'],old['inode'],old['bytes'],old['mtime_ns'],old['ctime_ns']),'Model changed since hash verification; reverify SHA256 before accepting'
with model.open('rb') as f:os.fsync(f.fileno())
fd=os.open(model.parent,os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(fd)
finally:os.close(fd)

library=r.parent/'cortex-stream-forge-qies074/build-quortex/bin/libggml-base.so.0.25.3'
obj=inspect(model,receipt['bytes'],library);obj['full_model_sha256_verified']=True
(checks/'gguf-inspection.json').write_text(json.dumps(obj,indent=2)+'\n')
lib=ctypes.CDLL(str(library));lib.ggml_type_name.argtypes=[ctypes.c_int];lib.ggml_type_name.restype=ctypes.c_char_p
index={'schema':'QIES098_RICURSIVE_READONLY_INDEX_v1','architecture':'ricursive','model_dir':str(model.parent),'model_filename':model.name,'model_sha256':receipt['sha256'],'model_revision':receipt['revision'],'model_integrity_status':'SHA256_VERIFIED','tensors':[dict(t,shard=model.name,quantization=lib.ggml_type_name(t['type']).decode()) for t in obj['tensors']]}
(checks/'ripple-tensor-index.json').write_text(json.dumps(index,indent=2)+'\n')
(checks/'single-layer-test-contract.json').write_text(json.dumps({'plans':[plan(obj,0),plan(obj,4)],'status':'PREPARED_NOT_EXECUTABLE; native adapter and qualified oracle missing'},indent=2)+'\n')
(checks/'download-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
subprocess.run(['python3','-I','-B',str(r/'qies098-tools/check_index_compat.py'),str(checks/'ripple-tensor-index.json')],env=env,check=True)
s=model.stat();(checks/'model-file-identity.json').write_text(json.dumps({'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'sha256':receipt['sha256']},indent=2)+'\n')
print('Final metadata/index inspection PASS; no tensor payloads read during inspection; no inference')
