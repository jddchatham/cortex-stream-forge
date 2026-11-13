"""Synthetic-only per-layer process orchestration; immutable all-layer token commits.
Real model access is impossible in numerical worker namespaces.
"""
from pathlib import Path
import importlib.util,json,hashlib,subprocess,os,datetime,uuid,struct,sys,shutil,argparse,time
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build'
spec=importlib.util.spec_from_file_location('monitor107',W/'qies107/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.R=R;m.B=B;m.monitor.R=R;m.monitor.B=B
base_namespace=m.monitor.namespace;bindings=[]
def namespace(role,out,args,marker=True):
 cmd=base_namespace(role,out,args,marker);i=cmd.index('--');extra=['--dir','/state']
 for src,dst in bindings:extra+=['--ro-bind',str(src),dst]
 return cmd[:i]+extra+cmd[i:]
m.monitor.namespace=namespace
ENV=os.environ.copy();ENV.pop('LD_LIBRARY_PATH',None)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def jsonwrite(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def state(p,layer,slots,pos):
 b=Path(p).read_bytes();assert len(b)==128+slots*4096,'state length';h=struct.unpack('<8Q',b[:64]);assert h==(0x32564b3930315155,1,layer,slots,pos,slots*1024,slots*1024,109),'state layer/position mismatch'
 assert hashlib.sha256(b[:64]+b[128:]).hexdigest().encode()==b[64:128],'state checksum';return sha(p)
def verified(directory,n,slots,pos):
 assert directory.name.endswith('.COMMITTED'),'unpublished state cannot be reused'
 d=json.loads((directory/'COMMIT.json').read_text());assert d['version']==1 and d['layers']==n and d['next_position']==pos and d['slots']==slots and len(d['states'])==n
 for layer in range(n):
  out=directory/f'layer-{layer}';assert sha(out/'OUTPUT.json')==d['outputs'][str(layer)],'output manifest identity'
  for name,h in json.loads((out/'OUTPUT.json').read_text()).items():assert sha(out/'captures-0'/name)==h,'committed capture integrity'
  assert state(out/'captures-0/state.kv',layer,slots,pos)==d['states'][str(layer)]
 return d

def baseline():
 good=[]
 for p in (W/'qies110/runs').glob('*/summary.json'):
  d=json.loads(p.read_text())
  if d.get('success') and str(d.get('results_root','')).startswith(str(W/'qies110/runs')) and len(d['comparisons'])==312 and len(d['faults'])==78:good.append(p)
 assert good,'Fresh QIES110 numerical/fault replay required'
 p=sorted(good)[-1];d=json.loads(p.read_text())
 for row in d['comparisons']:assert sha(p.parent/row['candidate']/'captures-0'/row['tap'])==row['sha256']==sha(p.parent/row['reference']/'captures-0'/row['tap'])
 return {'summary':str(p),'sha256':sha(p),'comparisons':312,'faults':78}
def fixtures():
 global bindings
 for profile in (0,15):
  out=R/'fixtures'/str(profile)
  if not out.exists():
   out.mkdir(parents=True);bindings=[];cmd=namespace('fixture-writer',out,[str(profile)])
   with (out/'creation.log').open('w') as f:subprocess.run(cmd,env=ENV,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
  assert (out/'manifest').is_file() and (out/'tensors.bin').is_file(),'interrupted fixture retained; use fresh worktree'
 # 80 tiny read-only plan directories share just two synthetic source files.
 for layer in range(80):
  p=R/'fixtures'/('layer-'+str(layer));profile=0 if layer<15 or layer>=65 else 15;source=R/'fixtures'/str(profile)
  if not p.exists():
   p.mkdir();(p/'tensors.bin').symlink_to(source/'tensors.bin');lines=(source/'manifest').read_text().splitlines();lines[2]=str(layer);(p/'manifest').write_text('\n'.join(lines)+'\n')
  # Bind the actual profile file separately: no symlink traversal outside sandbox.

def launch(root,role,layer,slots,pos,previous,input_path,original,fault='none',status=0,marker=True,argv=None):
 global bindings
 out=root/f'layer-{layer}';profile=0 if layer<15 or layer>=65 else 15;plan=R/'fixtures'/('layer-'+str(layer))/'manifest';source=R/'fixtures'/str(profile)/'tensors.bin'
 bindings=[(input_path,'/state/input.f32'),(original,'/state/original.f32')]
 if previous:bindings.append((previous,'/state/previous.kv'))
 if role=='candidate':bindings.extend([(plan,'/fixtures/manifest'),(source,'/fixtures/tensors.bin')])
 # /fixtures is created by bwrap automatically for file bind targets.
 args=argv or ['--synthetic',str(layer),str(slots),str(pos),'resume' if previous else 'seed','contract-v1',fault]
 before=sha(previous) if previous else None
 request={'role':role,'layer':layer,'slots':slots,'position':pos,'previous':before,'input_sha256':sha(input_path),'original_sha256':sha(original),'worker_sha256':sha(B/role),'args':args,'marker':marker}
 if (out/'INPUT.json').exists() and (out/'OUTPUT.json').exists():
  assert json.loads((out/'INPUT.json').read_text())==request,'cached input identity'
  result=json.loads((out/'result.json').read_text());assert result['status']==status and not result['reasons'] and result['rlimits_unchanged']
  for name,h in json.loads((out/'OUTPUT.json').read_text()).items():assert sha(out/'captures-0'/name)==h
  assert result['worker_records'][-1]['owned_resources']==0
 else:
  if out.exists():out.rename(root/(out.name+'.interrupted-'+uuid.uuid4().hex[:8]))
  result=m.job(root,'layer-'+str(layer),role,args,status=status,marker=marker)
  jsonwrite(out/'INPUT.json',request)
  if status==0:jsonwrite(out/'OUTPUT.json',{p.name:sha(p) for p in (out/'captures-0').iterdir()})
 assert not previous or sha(previous)==before
 assert not result['reasons'] and result['rlimits_unchanged']
 assert all(v.get('adapter_open_files_after_cleanup',0)==0 for v in result['worker_records'])
 if status==0:state(out/'captures-0/state.kv',layer,slots,pos+1)
 return result

def transaction(base,role,n,slots,pos,prior,embedding,inject=None,worker_fault=None):
 if prior:
  assert prior.parent==base and prior.name==role+f'-token-{pos-1}.COMMITTED','session/role checkpoint mismatch'
  verified(prior,n,slots,pos)
 stage=base/(role+f'-token-{pos}.UNCOMMITTED');final=base/(role+f'-token-{pos}.COMMITTED');
 if final.exists():verified(final,n,slots,pos+1);return final
 stage.mkdir(exist_ok=True)
 states={};outputs={};current=embedding;peak={k:0 for k in ('rss','pss','anon','file_pss')};controller=[]
 for layer in range(n):
  if inject==('before-layer',layer):raise RuntimeError('injected before layer')
  previous=prior/f'layer-{layer}/captures-0/state.kv' if prior else None
  result=launch(stage,role,layer,slots,pos,previous,current,embedding,fault=worker_fault or 'none',status=1 if worker_fault else 0)
  if worker_fault:raise RuntimeError('injected numerical fault; token remains uncommitted')
  for k,v in result['peak_tree_sampled'].items():peak[k]=max(peak[k],v)
  checkpoint=stage/f'layer-{layer}/captures-0/state.kv';states[str(layer)]=state(checkpoint,layer,slots,pos+1);outputs[str(layer)]=sha(stage/f'layer-{layer}/OUTPUT.json')
  current=stage/f'layer-{layer}/captures-0/final.f32'
  mem=m.monitor.tree_memory(os.getpid());own=mem[0] if mem else {};controller.append({'layer':layer,'rss_kib':own.get('Rss',0),'pss_kib':own.get('Pss',0),'anonymous_kib':own.get('Anonymous',0)})
  print(f'{role} token={pos} layer={layer} complete',flush=True)
  if inject==('after-layer',layer):raise RuntimeError('injected after layer')
 if inject==('before-manifest',0):raise RuntimeError('injected manifest failure')
 commit={'version':1,'layers':n,'slots':slots,'next_position':pos+1,'states':states,'outputs':outputs,'synthetic_only':True}
 if (stage/'COMMIT.json').exists():assert json.loads((stage/'COMMIT.json').read_text())==commit
 else:jsonwrite(stage/'COMMIT.json',commit)
 # Persist capture directory names and nested namespace outputs before publishing.
 for layer in range(n):syncdir(stage/f'layer-{layer}/captures-0');syncdir(stage/f'layer-{layer}')
 syncdir(stage)
 if inject==('before-publication',0):raise RuntimeError('injected before publication')
 assert not final.exists();os.rename(stage,final);syncdir(base);verified(final,n,slots,pos+1)
 jsonwrite(final/'METRICS.json',{'worker_tree_peak_bytes':peak,'controller_samples':controller})
 return final

def comparisons(candidate,reference,n,slots,pos,output):
 count=0
 for layer in range(n):
  a=candidate/f'layer-{layer}/captures-0';b=reference/f'layer-{layer}/captures-0';assert {p.name for p in a.iterdir()}=={p.name for p in b.iterdir()}
  for p in sorted(a.iterdir()):
   q=b/p.name;assert p.read_bytes()==q.read_bytes(),('BYTE EXACT MISMATCH',layer,pos,p.name)
   output.write(json.dumps({'layer':layer,'token_position':pos,'tap':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'byte_exact':True})+'\n');count+=1
  output.flush()
 return count

def campaign(root,n,start=0):
 base=root/f'layers-{n}-start-{start}';base.mkdir(exist_ok=True);slots=1280 if start else 256;cp=rp=None;count=0
 if (base/'comparisons.jsonl').exists():(base/'comparisons.jsonl').rename(base/('comparisons.interrupted-'+uuid.uuid4().hex[:8]+'.jsonl'))
 with (base/'comparisons.jsonl').open('x') as out:
  for pos in (start,start+1):
   emb=base/f'embedding-{pos}.f32';payload=struct.pack('<5120f',*[float((i*13+pos*7+7)%257-128)/128 for i in range(5120)])
   if emb.exists():assert emb.read_bytes()==payload
   else:emb.write_bytes(payload)
   cp=transaction(base,'candidate',n,slots,pos,cp,emb)
   rp=transaction(base,'reference',n,slots,pos,rp,emb)
   count+=comparisons(cp,rp,n,slots,pos,out)
   if start and pos==1024:
    for layer in range(n):
     blob=(cp/f'layer-{layer}/captures-0/state.kv').read_bytes();key0=blob[128:128+2048];v0=blob[128+slots*2048:128+slots*2048+2048]
     if layer%5!=4:assert key0==v0==bytes(2048),'sliding expired state not evicted'
     else:assert key0!=bytes(2048) and v0!=bytes(2048),'global history discarded'
 summary={'success':True,'layers':n,'positions':[start,start+1],'comparisons':count,'candidate_last':str(cp),'reference_last':str(rp),'synthetic_only':True,'sliding_expiration_and_global_retention_verified':bool(start)}
 if (base/'summary.json').exists():assert json.loads((base/'summary.json').read_text())==summary
 else:jsonwrite(base/'summary.json',summary)
 return base

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--layers',type=int,nargs='+',default=[2,4,8],choices=[2,4,8,16,80]);ap.add_argument('--boundary',action='store_true');ap.add_argument('--no-build',action='store_true');ap.add_argument('--resume',type=Path);a=ap.parse_args();base=baseline()
 if not a.no_build and not a.resume:subprocess.run(['/usr/bin/python3','-I','-B',R/'build.py'],env=ENV,check=True)
 fixtures()
 provenance={'baseline':base,'build':json.loads((B/'manifest.json').read_text()),'controller_sha256':sha(Path(__file__)),'layers':a.layers,'boundary':a.boundary}
 if a.resume:
  root=a.resume.resolve();root.relative_to((R/'runs').resolve());assert json.loads((root/'provenance.json').read_text())==provenance,'resume source/build/contract identity changed'
 else:
  root=R/'runs'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True);jsonwrite(root/'provenance.json',provenance)
 for n in a.layers:campaign(root,n)
 if a.boundary:campaign(root,5,1023)
 if not (root/'SUCCESS.json').exists():jsonwrite(root/'SUCCESS.json',{'success':True,'root':str(root),'layers':a.layers,'boundary':a.boundary,'real_weight_arithmetic':False})
 print('PASS QIES111:',root,flush=True)
if __name__=='__main__':main()
