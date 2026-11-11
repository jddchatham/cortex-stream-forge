"""Extract selected tensors with verified index; no mmap or numerical operators."""
from pathlib import Path
import importlib.util,json,resource,time,os,sys
R=Path(__file__).resolve().parent;W=R.parent
s=importlib.util.spec_from_file_location('q101_extract',R/'extract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
receipt=m.read_json(R/'model-hash-result.json');index=m.read_json(W/'qies098-checks/ripple-tensor-index.json');selection=m.read_json(R/'selected-tensors.json')
if not receipt['passed'] or index['model_sha256']!=receipt['actual'] or receipt['actual']!='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc':raise ValueError('model hash identity missing')
out=Path(sys.argv[1]);out.mkdir(mode=0o700)
need=sum(q['bytes'] for q in selection);disk=os.statvfs(out)
if disk.f_bavail*disk.f_frsize-need<30*1024**3:raise ValueError('disk reserve')
rows=[];t=time.monotonic()
with m.Extractor(W.parent/'qies098-data/models'/index['model_filename'],index,receipt['identity']) as reader:
 for q in selection:
  v=os.statvfs(out)
  if v.f_bavail*v.f_frsize-q['bytes']<30*1024**3:raise ValueError('disk reserve')
  row=reader.extract(q['name'],out/(q['name']+'.bin'));rows.append(row)
  print('EXTRACTED',q['name'],row['extracted_bytes'],flush=True)
 read=reader.bytes_read
(R/'index-extraction-result.json').write_text(json.dumps({'tensors':rows,'payload_bytes_read':read,'elapsed_seconds':time.monotonic()-t,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'chunk_bytes':m.CHUNK,'whole_model_mmap':False,'conversion':False},indent=2)+'\n')
