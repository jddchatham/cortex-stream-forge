"""Real-byte validation ONLY. No inference or GGML computation is invoked."""
from pathlib import Path
import json,subprocess,os,time,hashlib,copy,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from adapter import catalog,layer,manifest,q101
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build';O=R/os.environ.get('QIES110_STORAGE_RESULTS','storage-results');O.mkdir(exist_ok=False);env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
model=W.parent/'qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf';receipt=q101.read_json(W/'qies102/model-hash-result.json');assert receipt['passed'] and q101.identity(model.stat())==receipt['identity']
index,tensors=catalog();rows=[];measurements=[]
def run(cmd,label,success=True):
 log=O/(label+'.log');err=O/(label+'.err');peak={k:0 for k in ('Rss','Pss','Anonymous','Pss_File')};start=time.monotonic()
 with log.open('w') as out,err.open('w') as error:
  p=subprocess.Popen(list(map(str,cmd)),env=env,stdout=out,stderr=error,stdin=subprocess.DEVNULL)
  while p.poll() is None:
   try:
    values={}
    for line in Path(f'/proc/{p.pid}/smaps_rollup').read_text().splitlines()[1:]:
     k,v=line.split(':',1);values[k]=int(v.split()[0])
    for k in peak:peak[k]=max(peak[k],values.get(k,0))
   except (OSError,ValueError):pass
   if time.monotonic()-start>120:p.kill();raise RuntimeError('storage timeout')
   time.sleep(.02)
  p.wait()
 assert (p.returncode==0)==success,(label,p.returncode,err.read_text())
 measurements.append({'label':label,'status':p.returncode,'phase_sampled_kib':peak,'elapsed_seconds':time.monotonic()-start,'real_arithmetic':False});return log
for number in (0,4):
 ref=run([B/'reference-hash',model,number],f'independent-layer{number}')
 parsed={}
 for line in ref.read_text().splitlines():
  name,off,size,typ,h=line.split('\t');t=tensors[name];assert [int(off),int(size),int(typ)]==[t['offset'],t['bytes'],t['type']];parsed[name.split('.',2)[2]]=h
 plan=O/f'real-layer{number}.plan';manifest(plan,model,number,parsed)
 actual=run([B/'storage',plan,'schedule'],f'bounded-layer{number}')
 data=[json.loads(s) for s in actual.read_text().splitlines()]
 hashed=[d for d in data if 'tensor' in d];assert len(hashed)==15 and all(d['sha256']==parsed[d['tensor']] for d in hashed)
 counters=next(d for d in data if 'max_request' in d);assert counters['max_request']<=1024**2 and counters['cache_peak']<=1024**2
 rows.extend(dict(d,layer=number) for d in hashed)
# Entire real-model catalog/profile mapping without payload allocation.
assert all(len(layer(n))==15 for n in range(80))
# C++ descriptor rejection before reads. Only new text manifests are corrupted.
base=(O/'real-layer0.plan').read_text();parts=base.splitlines();faults=[]
mutations={'missing':lambda x:x[:-1], 'bad-count':lambda x:x[:4]+['14']+x[5:], 'bad-mode':lambda x:[x[0],'REALX']+x[2:], 'bad-shape':lambda x:x[:5]+[x[5].replace('5120','5121',1)]+x[6:], 'duplicate':lambda x:x[:6]+[x[5]]+x[7:], 'negative-offset':lambda x:x[:5]+['"attn_gate.weight" 12 5120 10240 -1 29491200 '+x[5].split()[-1]]+x[6:], 'bad-digest':lambda x:x[:-1]+[x[-1][:-64]+'g'*64]}
for name,fn in mutations.items():
 p=O/(name+'.plan');p.write_text('\n'.join(fn(list(parts)))+'\n');run([B/'storage',p,'hash'],'reject-'+name,False);faults.append(name)
# Original QIES101 fault suite uses synthetic files and remains unchanged.
scratch=O/'q101-original';scratch.mkdir()
import shutil
for name in ('extract.py','test_extract.py'):shutil.copyfile(W/'qies101'/name,scratch/name)
run(['/usr/bin/python3','-I','-B',scratch/'test_extract.py'],'qies101-original-faults')
assert q101.identity(model.stat())==receipt['identity']
(O/'summary.json').write_text(json.dumps({'real_tensor_hash_comparisons':rows,'verified_count':30,'all_80_layers_mapped':True,'descriptor_count':len(tensors),'all_layer_tensor_mappings':1200,'descriptor_faults':faults,'original_q101_faults':38,'model_sha256_receipt':receipt['actual'],'full_model_rehashed_this_task':False,'model_stat_bound_to_verified_receipt':True,'measurements':measurements,'storage_only':True,'real_weight_arithmetic':False},indent=2)+'\n');print('PASS storage only: 30 independent real tensor hashes; all 1200 layer mappings; bounds/scheduling and descriptor faults',flush=True)
