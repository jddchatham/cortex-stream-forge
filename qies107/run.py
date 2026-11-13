"""Complete real-width PROCEDURAL SYNTHETIC FFN validation; no real model input."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,datetime,uuid,resource,importlib.util,tarfile,struct,argparse
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build';G=1024**3
spec=importlib.util.spec_from_file_location('preserved_q106_monitor',W/'qies106/run.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
# Reuse read-only monitoring/process-tree utilities, never its inference guard.
monitor.R=R;monitor.B=B
ENV=os.environ.copy();ENV.pop('LD_LIBRARY_PATH',None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def baseline():
 receipt=json.loads((R/'QIES106-preservation.json').read_text());assert receipt['archive_sha256']==sha(R/'QIES106-preserved.tar.gz')
 with tarfile.open(R/'QIES106-preserved.tar.gz') as t:
  expected=None
  for member in t.getmembers():
   if member.name.endswith('/summary.json'):
    s=json.load(t.extractfile(member))
    if len(s.get('comparisons',[]))==7 and s.get('success'):expected=s
 assert expected is not None
 successful=[]
 for p in (W/'qies106/runs').glob('*/summary.json'):
  d=json.loads(p.read_text())
  if d.get('success') and len(d.get('comparisons',[]))==7:successful.append((p,d))
 assert successful,'QIES106 seven-comparison replay must pass before new arithmetic'
 path,actual=sorted(successful)[-1]
 assert {x['label']:x['output_sha256'] for x in actual['comparisons']}=={x['label']:x['output_sha256'] for x in expected['comparisons']}
 return {'reproduction':str(path),'sha256':sha(path),'comparisons':7,'output_hashes_match_preserved_original':True,'compiler_cpu':actual['build_manifest']}

def job(root,label,role,args,status=0,marker=True,query=False):
 out=root/label;out.mkdir(exist_ok=False);initial=monitor.host();cg0=monitor.cgroups();limits=(resource.getrlimit(resource.RLIMIT_AS),resource.getrlimit(resource.RLIMIT_RSS));start=time.monotonic()
 # A separate synthetic-only bound derived from 3*137.5 MiB full reference
 # weights + activations/workspace. It does NOT modify QIES106 or real policy.
 if initial['MemAvailable']<int((1.75 if role=='reference' else 1.5)*G) or initial['SwapFree']<2*G or initial['disk_free']<20*G:raise RuntimeError('synthetic FFN resource preflight declined')
 peak={k:0 for k in ('rss','pss','anon','file_pss')};minram=initial['MemAvailable'];reasons=[]
 cmd=monitor.namespace(role,out,args,marker)
 with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.log').open('w') as stderr,(out/'monitor.jsonl').open('w') as logfile:
  p=subprocess.Popen(cmd,env=ENV,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
  try:
   while p.poll() is None:
    h=monitor.host();members=monitor.tree_memory(p.pid);minram=min(minram,h['MemAvailable'])
    for key,field in [('rss','Rss'),('pss','Pss'),('anon','Anonymous'),('file_pss','Pss_File')]:peak[key]=max(peak[key],sum(x.get(field,0) for x in members))
    cg=monitor.cgroups()
    for path,v in cg.items():
     if path in cg0:
      if any(v['memory.events'].get(k,0)>cg0[path]['memory.events'].get(k,0) for k in ('oom','oom_kill','oom_group_kill')):reasons.append('ancestor OOM')
    logfile.write(json.dumps({'elapsed':time.monotonic()-start,'host':h,'tree':members,'first_cgroup_current':next(iter(cg.values()))['memory.current'] if cg else None})+'\n');logfile.flush()
    if h['MemAvailable']<G or h['SwapFree']<2*G or h['disk_free']<20*G:reasons.append('synthetic system safety stop')
    if peak['rss']>(256 if role=='candidate' else 640)*1024**2:reasons.append('synthetic worker RSS ceiling')
    if time.monotonic()-start>120:reasons.append('synthetic timeout')
    if reasons:monitor.terminate(p);break
    time.sleep(.05)
   p.wait()
  except BaseException:monitor.terminate(p);raise
 raw=(out/'stdout.jsonl').read_text();data=[]
 if not query:data=[json.loads(s) for s in raw.splitlines()]
 result={'label':label,'role':role,'command':cmd,'status':p.returncode,'elapsed_seconds':time.monotonic()-start,'peak_tree_sampled':peak,'worker_records':data,'minimum_system_MemAvailable':minram,'host_before':initial,'host_after':monitor.host(),'cgroup_before':cg0,'cgroup_after':monitor.cgroups(),'cgroup_attribution':'shared inherited desktop scope; memory.peak historical, not reset or isolated','reasons':reasons,'rlimits_unchanged':limits==(resource.getrlimit(resource.RLIMIT_AS),resource.getrlimit(resource.RLIMIT_RSS)),'real_weights':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 assert not reasons,(label,reasons)
 assert p.returncode==status,(label,p.returncode,(out/'stderr.log').read_text())
 assert result['rlimits_unchanged']
 if not query:
  last=data[-1];assert last['owned_resources']==0,last
  assert data[-2]['phase']==('after-cleanup' if status==0 else 'after-failure-cleanup')
  if status:assert not list(out.glob('captures-*')) and not list(out.glob('staging-*'))
 return result

def compare(root,candidate,reference,repeats=1):
 all_weights=[v for v in candidate['worker_records'] if 'weight' in v];b={v['weight']:(v['sha256'],v['bytes']) for v in reference['worker_records'] if 'weight' in v};assert len(all_weights)==3*repeats and len(b)==3
 for v in all_weights:assert (v['sha256'],v['bytes'])==b[v['weight']],('weight identity',v)
 rows=[]
 for seq in range(repeats):
  for tap,n in [('input-norm',5120),('gate',40960),('up',40960),('active',40960),('down',5120),('post-norm',5120),('final',5120)]:
   x=root/candidate['label']/f'captures-{seq}'/(tap+'.f32');y=root/reference['label']/'captures-0'/(tap+'.f32');av=x.read_bytes();bv=y.read_bytes();assert len(av)==len(bv)==n*4
   if av!=bv:
    af=struct.unpack('<'+str(n)+'f',av);bf=struct.unpack('<'+str(n)+'f',bv);mismatch={'candidate':candidate['label'],'reference':reference['label'],'tap':tap,'different_floats':sum(a!=b for a,b in zip(af,bf)),'max_abs':max(abs(a-b) for a,b in zip(af,bf)),'candidate_sha256':sha(x),'reference_sha256':sha(y)}
    (root/'mismatch.json').write_text(json.dumps(mismatch,indent=2));raise AssertionError(('BYTE MISMATCH; stopped; no tolerance relaxation',mismatch))
   rows.append({'case':candidate['label'],'repeat':seq,'tap':tap,'elements':n,'bytes':len(av),'byte_exact':True,'sha256':sha(x)})
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--build-only',action='store_true');a=p.parse_args();base=baseline()
 subprocess.run(['/usr/bin/python3','-I','-B',R/'build.py'],env=ENV,check=True)
 if a.build_only:return
 root=R/'runs'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True);jobs=[];rows=[];clamps=[];refs={}
 for quant in (['q5'] if a.smoke else ['q5','q4']):
  for vector in ([0] if a.smoke else [0,1,2]):
   ref=job(root,f'reference-{quant}-v{vector}','reference',['--synthetic',quant,str(vector)]);jobs.append(ref);refs[(quant,vector)]=ref
   cand=job(root,f'candidate-{quant}-v{vector}','candidate',['--synthetic',quant,str(vector),'none','1']);jobs.append(cand);rows+=compare(root,cand,ref)
   folder=root/ref['label']/'captures-0';gate=struct.unpack('<40960f',(folder/'gate.f32').read_bytes());up=struct.unpack('<40960f',(folder/'up.f32').read_bytes())
   coverage={'profile':quant,'vector':vector,'gate_above_20':sum(x>20 for x in gate),'up_above_20':sum(x>20 for x in up),'up_below_minus20':sum(x< -20 for x in up),'gate_below_minus20_not_lower_clamped':sum(x< -20 for x in gate)}
   if vector!=2:assert all(coverage[k]>0 for k in ('gate_above_20','up_above_20','up_below_minus20','gate_below_minus20_not_lower_clamped')),coverage
   else:assert all(x==0 for x in struct.unpack('<5120f',(folder/'final.f32').read_bytes()))
   clamps.append(coverage)
 if a.smoke:
  (root/'smoke.json').write_text(json.dumps({'comparisons':rows,'clamp_coverage':clamps,'runs':jobs},indent=2));print('PASS complete FFN smoke:',root,flush=True);return
 repeated=job(root,'candidate-q5-repeat3','candidate',['--synthetic','q5','0','none','3']);jobs.append(repeated);rows+=compare(root,repeated,refs[('q5',0)],3)
 faults=[s.strip().strip('",') for s in (R/'faults.inc').read_text().splitlines()];assert len(faults)==len(set(faults));fault_results=[]
 for fault in faults:
  item=job(root,'fault-'+fault,'candidate',['--synthetic','q5','0',fault,'1'],status=1);jobs.append(item);assert item['worker_records'][-1]['fault_seen'] is True
  assert ('injected '+fault) in (root/item['label']/'stderr.log').read_text();fault_results.append(fault)
 for role in ('candidate','reference'):
  good=['--synthetic','q5','0',*(['none','1'] if role=='candidate' else [])]
  for name,args,marker in [('real',['--real','missing'],True),('no-marker',good,False),('file-input',['--synthetic','/models/model.gguf','0',*(['none','1'] if role=='candidate' else [])],True)]:
   rejected=job(root,'reject-'+role+'-'+name,role,args,status=1,marker=marker);jobs.append(rejected);assert rejected['worker_records'][-1]['peak_weight_bytes']==0
 result={'success':True,'baseline_reproduction':base,'build':json.loads((B/'build-manifest.json').read_text()),'results_root':str(root),'comparisons':rows,'clamp_coverage':clamps,'faults_passed':fault_results,'rejected_entry_cases':6,'runs':jobs,'real_model_access':False,'full_Ricursive_layer_claim':False,'existing_gate_changed':False,'whole_system_4GiB_qualification':False}
 (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),'exact intermediate comparisons,',len(faults),'faults; summary',root/'summary.json',flush=True)
if __name__=='__main__':main()
