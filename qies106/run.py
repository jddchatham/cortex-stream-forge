"""QIES106 fixed, procedural-synthetic CPU experiment. No model-input option.
Runtime monitoring here is an additional synthetic experiment boundary, not a
replacement, override, or qualification of the existing real-inference guards.
"""
from pathlib import Path
import os,sys,json,subprocess,hashlib,time,signal,resource,argparse,datetime,uuid,math,struct
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build';G=1024**3
ENV=os.environ.copy();ENV.pop('LD_LIBRARY_PATH',None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def host():
 mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
 fs=os.statvfs(R);return {'MemTotal':mem['MemTotal'],'MemAvailable':mem['MemAvailable'],'SwapFree':mem['SwapFree'],'disk_free':fs.f_bavail*fs.f_frsize}
def cgroups():
 rel=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));p=Path('/sys/fs/cgroup')/rel.lstrip('/')
 rows={}
 for p in (p,*p.parents):
  if not (p/'memory.events').exists():continue
  d={}
  for name in ('memory.events','memory.current','memory.peak','memory.stat','memory.swap.current','memory.max','memory.high','memory.swap.max'):
   try:
    raw=(p/name).read_text().strip();d[name]={k:int(v) for k,v in (l.split() for l in raw.splitlines())} if name in ('memory.events','memory.stat') else raw
   except OSError:d[name]=None
  rows[str(p)]=d
 return rows
def subtree(pid):
 found=[];pending=[pid]
 while pending:
  p=pending.pop();found.append(p)
  try:pending.extend(map(int,Path(f'/proc/{p}/task/{p}/children').read_text().split()))
  except OSError:pass
 return found
def tree_memory(pid):
 rows=[]
 for p in subtree(pid):
  try:
   values={}
   for line in Path(f'/proc/{p}/smaps_rollup').read_text().splitlines()[1:]:
    key,rest=line.split(':',1);values[key]=int(rest.split()[0])*1024
   values['pid']=p;rows.append(values)
  except (OSError,ProcessLookupError,ValueError):pass
 return rows

def namespace(role,out,args,marker=True):
 cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr','--ro-bind','/etc','/etc','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/home','--dir','/workers','--ro-bind',str(B/role),'/workers/'+role,'--ro-bind',str(B/(role+'-libs')),'/synthetic-sdk','--bind',str(out),'/outputs','--setenv','PATH','/usr/bin:/bin','--setenv','LANG','C.UTF-8','--setenv','OMP_NUM_THREADS','1']
 if marker:cmd+=['--ro-bind',str(B/'marker'),'/qies104.synthetic','--setenv','QIES104_SYNTHETIC','1']
 return cmd+['--','/lib64/ld-linux-x86-64.so.2','/workers/'+role,*args]

def terminate(p):
 pids=subtree(p.pid)
 for pid in reversed(pids):
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 try:p.wait(timeout=2)
 except subprocess.TimeoutExpired:pass
 for pid in reversed(pids):
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 p.wait(timeout=5)

def run_one(base,label,role,args,expect=0,marker=True):
 out=base/label;out.mkdir(exist_ok=False);before=host();cg0=cgroups();limits0=(resource.getrlimit(resource.RLIMIT_AS),resource.getrlimit(resource.RLIMIT_RSS))
 if before['MemAvailable']<int(1.5*G) or before['SwapFree']<2*G or before['disk_free']<20*G:raise RuntimeError('Synthetic experiment resource preflight refused (not the real-inference entry gate)')
 peak={'tree_rss':0,'tree_pss':0,'tree_anonymous':0,'tree_pss_anon':0,'tree_pss_file':0};start=time.monotonic();reasons=[];samples=0;cgmax=0;minavailable=before['MemAvailable']
 command=namespace(role,out,args,marker)
 with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.log').open('w') as stderr,(out/'monitor.jsonl').open('w') as monitor:
  p=subprocess.Popen(command,env=ENV,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
  try:
   while p.poll() is None:
    now=host();minavailable=min(minavailable,now['MemAvailable']);memory=tree_memory(p.pid);sample={'elapsed':time.monotonic()-start,'host':now,'pids':memory};samples+=1
    for key,source in [('tree_rss','Rss'),('tree_pss','Pss'),('tree_anonymous','Anonymous'),('tree_pss_anon','Pss_Anon'),('tree_pss_file','Pss_File')]:peak[key]=max(peak[key],sum(v.get(source,0) for v in memory))
    for path,v in cgroups().items():
     if path in cg0:
      if v.get('memory.current'):cgmax=max(cgmax,int(v['memory.current'])) if path==next(iter(cg0)) else cgmax
      old=cg0[path]['memory.events'];new=v['memory.events']
      if any(new.get(k,0)>old.get(k,0) for k in ('oom','oom_kill','oom_group_kill')):reasons.append('ancestor OOM event')
    monitor.write(json.dumps(sample)+'\n');monitor.flush()
    if now['MemAvailable']<G:reasons.append('MemAvailable below 1 GiB')
    if now['SwapFree']<2*G:reasons.append('SwapFree below 2 GiB')
    if now['disk_free']<20*G:reasons.append('disk below 20 GiB')
    if peak['tree_rss']>(256 if role=='candidate' else 384)*1024**2:reasons.append('synthetic process tree RSS budget')
    if time.monotonic()-start>300:reasons.append('synthetic timeout')
    if reasons:terminate(p);break
    time.sleep(.05)
   p.wait()
  except BaseException:terminate(p);raise
 elapsed=time.monotonic()-start
 data=[]
 for line in (out/'stdout.jsonl').read_text().splitlines():
  try:data.append(json.loads(line))
  except json.JSONDecodeError:raise AssertionError(('non-JSON worker output',line))
 result={'label':label,'role':role,'command':command,'status':p.returncode,'elapsed_seconds_including_synthesis_hashing_io':elapsed,'safety_reasons':reasons,'peak_monitor':peak,'worker_records':data,'host_before':before,'host_after':host(),'minimum_sampled_MemAvailable':minavailable,'samples':samples,'cgroup_before':cg0,'cgroup_after':cgroups(),'cgroup_current_max_sampled':cgmax,'cgroup_scope':'inherited shared desktop cgroup; not isolated worker accounting; memory.peak is historical and not reset','rlimits_unchanged':limits0==(resource.getrlimit(resource.RLIMIT_AS),resource.getrlimit(resource.RLIMIT_RSS)),'real_weight_access':False,'output_sha256':sha(out/'output.f32') if (out/'output.f32').exists() else None}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 if reasons:raise AssertionError(('synthetic safety stop',label,reasons))
 if p.returncode!=expect:raise AssertionError((label,p.returncode,(out/'stderr.log').read_text()))
 if expect==0:
  assert data[-1]['phase']=='after-cleanup' and data[-1]['live_owned_resources']==0
  assert not (out/'output.partial').exists() and (out/'output.f32').is_file()
 else:
  assert not (out/'output.f32').exists() and not (out/'output.partial').exists()
  assert data[-1]['live_owned_resources']==0
 return result

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--case',choices=['attn_q','attn_k','attn_output','ffn_gate','ffn_down']);a.add_argument('--build-only',action='store_true');args=a.parse_args()
 # Source changes require recompilation; no use of stale numerical workers.
 subprocess.run(['/usr/bin/python3','-I','-B',R/'build.py'],env=ENV,check=True)
 if args.build_only:return
 stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8];base=R/'runs'/stamp;base.mkdir(parents=True)
 manifest=json.loads((B/'build-manifest.json').read_text());assert not any('libllama' in n for role in manifest['workers'].values() for n in role['libraries'])
 baseline_hashes={str(p.relative_to(W)):sha(p) for p in [W/'qies104/admission.cpp',W/'qies104/admission.h',W/'qies104/gate.py',W/'qies105/guards/qies077-guard.py']}
 records=[];comparisons=[]
 cases=[args.case] if args.case else ['attn_q','attn_k','attn_output','ffn_gate','ffn_down']
 def compare(label,candidate,reference):
  ar=next(x for x in candidate['worker_records'] if 'weight_sha256' in x);br=next(x for x in reference['worker_records'] if 'weight_sha256' in x)
  assert (ar['weight_sha256'],ar['columns'],ar['rows'],ar['quantized_bytes'])==(br['weight_sha256'],br['columns'],br['rows'],br['quantized_bytes']),('synthetic weight identity differs',label)
  left=base/candidate['label']/'output.f32';right=base/reference['label']/'output.f32';x=left.read_bytes();y=right.read_bytes()
  if x!=y:
   av=struct.unpack('<'+'f'*(len(x)//4),x);bv=struct.unpack('<'+'f'*(len(y)//4),y)
   mismatch={'label':label,'candidate_sha256':sha(left),'reference_sha256':sha(right),'differing_floats':sum(a!=b for a,b in zip(av,bv)),'max_abs':max(abs(a-b) for a,b in zip(av,bv)),'status':'MISMATCH_STOP_NO_TOLERANCE_RELAXATION'}
   (base/'mismatch.json').write_text(json.dumps(mismatch,indent=2));raise AssertionError(mismatch)
  assert len(x)==ar['rows']*4
  comparisons.append({'label':label,'elements':len(x)//4,'bytes':len(x),'byte_exact':True,'output_sha256':sha(left),'weight_sha256':ar['weight_sha256']})
 for case in cases:
  ref=run_one(base,'reference-'+case,'reference',['--synthetic',case]);records.append(ref)
  candidate=run_one(base,'candidate-'+case,'candidate',['--synthetic',case,'128','none']);records.append(candidate);compare(case,candidate,ref)
  if case=='attn_q' and not args.case:
   for tile in (16,64):
    extra=run_one(base,f'candidate-{case}-tile{tile}','candidate',['--synthetic',case,str(tile),'none']);records.append(extra);compare(f'{case}-tile{tile}',extra,ref)
 # Deterministic cleanup faults, after some successful tile work as appropriate.
 for fault in ('after-input','after-load','after-compute','before-publish'):
  result=run_one(base,'fault-'+fault,'candidate',['--synthetic','attn_k','128',fault],expect=1);records.append(result);assert 'injected '+fault in (base/result['label']/'stderr.log').read_text()
 # New explicit synthetic interface never makes arbitrary files or real dispatch
 # available. Existing real guard is compiled unchanged into BOTH workers.
 for role in ('candidate','reference'):
  for name,cmd,marker in [('real',['--real','/models/absent'],True),('no-marker',['--synthetic','attn_k',*(['128','none'] if role=='candidate' else [])],False),('file-disguised-as-case',['--synthetic','/models/absent',*(['128','none'] if role=='candidate' else [])],True)]:
   r=run_one(base,'reject-'+role+'-'+name,role,cmd,expect=1,marker=marker);records.append(r)
   assert all(v.get('weight_bytes',0)==v.get('cpu_tensor_buffer_bytes',0)==0 for v in r['worker_records'])
   assert ('BLOCKED' in (base/r['label']/'stderr.log').read_text()) if name!='file-disguised-as-case' else True
 assert all(sha(W/p)==h for p,h in baseline_hashes.items())
 result={'source_parent':manifest['parent'],'build_manifest':manifest,'build_manifest_sha256':sha(B/'build-manifest.json'),'results_root':str(base),'actual_model_access':False,'real_execution_unlocked':False,'existing_guards_unchanged':True,'comparisons':comparisons,'runs':records,'success':True,'claims':'real-width procedural synthetic matvecs only; no complete layer, real weights, or 4 GiB hardware execution qualification'}
 (base/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS '+str(len(comparisons))+' byte-exact real-width synthetic comparisons; report '+str(base/'summary.json'),flush=True)
if __name__=='__main__':main()
