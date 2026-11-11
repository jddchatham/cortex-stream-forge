"""QIES101 extraction -> immutable raw-file bundles; no tensor arithmetic."""
from pathlib import Path
import importlib.util,json,hashlib,subprocess,os
R=Path(__file__).resolve().parent;W=R.parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
s=importlib.util.spec_from_file_location('q101_reader',W/'qies101/extract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
receipt=m.read_json(R/'model-hash-result.json');index=m.read_json(W/'qies098-checks/ripple-tensor-index.json');model=W.parent/'qies098-data/models'/index['model_filename']
assert receipt['passed'] and receipt['actual']==index['model_sha256']=='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc';assert m.identity(model.stat())==receipt['identity']
t=m.validate(index,receipt['bytes']);selected=[x for x in t.values() if x['name'].startswith('blk.0.')];assert len(selected)==15
(R/'bundle').mkdir();(R/'bundle/index').mkdir();(R/'bundle/stock').mkdir();missing=[];locations={};stockpaths={}
for q in selected:
 old=W/'qies101/artifacts/index'/(q['name']+'.bin');oldref=W/'qies101/artifacts/stock'/(q['name']+'.bin')
 if old.exists() and oldref.exists():locations[q['name']]=old;stockpaths[q['name']]=oldref
 else:missing.append(q);locations[q['name']]=R/'bundle/index'/(q['name']+'.bin');stockpaths[q['name']]=R/'bundle/stock'/(q['name']+'.bin')
need=2*sum(q['bytes'] for q in missing);v=os.statvfs(R);assert v.f_bavail*v.f_frsize-need>=30*1024**3
with m.Extractor(model,index,receipt['identity']) as reader:
 for q in missing:reader.extract(q['name'],locations[q['name']])
with (R/'reference-extract.log').open('w') as f:subprocess.run([str(W/'qies101/reference-extract'),str(model),str(R/'bundle/stock'),*[q['name'] for q in missing]],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
rows=[]
for q in selected:
 a=locations[q['name']];b=stockpaths[q['name']];h=hashlib.sha256();count=0
 assert a.stat().st_size==b.stat().st_size==q['bytes']
 with a.open('rb',buffering=0) as f,b.open('rb',buffering=0) as g:
  while x:=f.read(m.CHUNK):y=g.read(len(x));assert x==y;h.update(x);count+=len(x)
  assert not g.read(1)
 rows.append(dict(q,path=str(a.resolve()),reference_path=str(b.resolve()),identity=m.identity(a.stat()),sha256=h.hexdigest(),equal_bytes=count))

def plan(path,kind,layer,entries):
 cfg=[layer,5120,40960,80,8,128,256,1024] if kind=='REAL' else [layer,256,512,4,2,64,256,2]
 lines=['QIES102-BUNDLE-1',kind,' '.join(map(str,cfg))]
 for q in entries:
  shape=q['shape'];lines.append(' '.join([json.dumps(q['name']),str(q['type']),str(shape[0]),str(shape[1] if len(shape)>1 else 1),str(q['offset']),str(q['bytes']),*map(str,q['identity']),q['sha256'],json.dumps(q['path'])]))
 path.write_text('\n'.join(lines)+'\n')
plan(R/'real-layer-00.plan','REAL',0,rows)
(R/'bundle-manifest.json').write_text(json.dumps({'model_sha256':receipt['actual'],'original_model_identity':receipt['identity'],'layer':0,'tensors':rows,'reused_qies101_files':len(selected)-len(missing),'new_tensors':len(missing),'all_bytes_exact':True,'plan_sha256':hashlib.sha256((R/'real-layer-00.plan').read_bytes()).hexdigest()},indent=2)+'\n')
for label in ('fixture','fixture-quantized'):
 folder=W/'qies100'/label;shapes=json.loads((folder/'shapes.json').read_text())
 for layer in range(5):
  entries=[]
  for name,shape in shapes.items():
   if not name.startswith(f'blk.{layer}.'):continue
   file=folder/(name+'.bin');typ=folder/(name+'.type');entries.append({'name':name,'shape':list(reversed(shape)),'type':int(typ.read_text()) if typ.exists() else 0,'offset':0,'bytes':file.stat().st_size,'identity':m.identity(file.stat()),'path':str(file.resolve()),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
  plan(R/f'{label}-{layer}.plan','FIXTURE',layer,entries)
print('PASS 15 real layer-0 tensors independently exact; reused',len(selected)-len(missing),'QIES101 files; emitted ten synthetic plans')
