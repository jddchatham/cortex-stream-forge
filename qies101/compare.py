"""Byte comparison of independent real-tensor extractions; never evaluates tensors."""
from pathlib import Path
import hashlib,json,subprocess,os,time,resource
R=Path(__file__).resolve().parent;W=R.parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
q=R/'artifacts/index';ref=R/'artifacts/stock';q.parent.mkdir(exist_ok=True);ref.mkdir(mode=0o700)
selection=json.loads((R/'selected-tensors.json').read_text());names=[x['name'] for x in selection]
def run(command,log,error=None):
 with log.open('w') as f:
  if error:
   with error.open('w') as e:subprocess.run(list(map(str,command)),env=env,stdout=f,stderr=e,check=True,timeout=300)
  else:subprocess.run(list(map(str,command)),env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
run(['/usr/bin/python3','-I','-B',R/'run_index_extract.py',q],R/'index-extraction.log')
run([R/'reference-extract',W.parent/'qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf',ref,*names],R/'reference-extraction.log',R/'reference-extraction-resources.log')
rows=[];start=time.monotonic()
for t in selection:
 name=t['name'];a=q/(name+'.bin');b=ref/(name+'.bin');assert a.stat().st_size==b.stat().st_size==t['bytes'];ha=hashlib.sha256();hb=hashlib.sha256();count=0
 with a.open('rb',buffering=0) as f,b.open('rb',buffering=0) as g:
  while x:=f.read(1024*1024):
   y=g.read(len(x));assert x==y,(name,count);ha.update(x);hb.update(y);count+=len(x)
  assert not g.read(1)
 rows.append(dict(t,sha256=ha.hexdigest(),stock_sha256=hb.hexdigest(),compared_bytes=count,all_bytes_equal=True))
result={'tensors':rows,'total_compared_bytes':sum(x['compared_bytes'] for x in rows),'all_bytes_equal':True,'compare_seconds':time.monotonic()-start,'compare_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'no_inference':True}
(R/'byte-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),'complete real tensors;',result['total_compared_bytes'],'bytes equal')
