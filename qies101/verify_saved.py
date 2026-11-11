"""Bounded, repeatable saved-artifact verification; no extraction or inference."""
from pathlib import Path
import importlib.util,json,hashlib,os,resource
R=Path(__file__).resolve().parent;W=R.parent
s=importlib.util.spec_from_file_location('q101_ext',R/'extract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
receipt=m.read_json(R/'model-hash-result.json');assert receipt['passed'] and receipt['expected']==receipt['actual']=='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
index=m.read_json(W/'qies098-checks/ripple-tensor-index.json');tensors=m.validate(index,receipt['bytes']);assert len(tensors)==1203
model=W.parent/'qies098-data/models'/index['model_filename'];assert m.identity(model.stat())==receipt['identity']
comparison=m.read_json(R/'byte-comparison.json');assert comparison['all_bytes_equal']
for t in comparison['tensors']:
 a=R/'artifacts/index'/(t['name']+'.bin');b=R/'artifacts/stock'/(t['name']+'.bin');assert a.stat().st_size==b.stat().st_size==tensors[t['name']]['bytes'];h=hashlib.sha256();count=0
 with a.open('rb',buffering=0) as f,b.open('rb',buffering=0) as g:
  while x:=f.read(m.CHUNK):
   y=g.read(len(x));assert x==y;h.update(x);count+=len(x)
  assert not g.read(1)
 assert count==t['bytes'] and h.hexdigest()==t['sha256']==t['stock_sha256']
print('PASS 1203 indexed descriptors and 8 saved complete tensor byte comparisons; peak_rss_kib='+str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
