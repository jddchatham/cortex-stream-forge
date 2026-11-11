from pathlib import Path
import importlib.util,json,sys,hashlib,resource,subprocess,os
R=Path(__file__).resolve().parent;W=R.parent;GUARD='ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=W.parent/'qies077-guard.py';assert sha(p)==GUARD
s=importlib.util.spec_from_file_location('q104_original_guard',p);g=importlib.util.module_from_spec(s);sys.modules[s.name]=g;s.loader.exec_module(g)
host=g.host(R);log=R/'original-guard-refusal'
if not log.exists():
 if host['available']>=g.Policy().entry_ram:raise SystemExit('Unexpected passing host: qualification sequence requires review; no live job launched')
 assert g.run_guard(['/usr/bin/true'],log)!=0
result=json.loads((log/'result.json').read_text());assert not result['ok'] and result['cleanup'] and result['reason']=='GuardAbort: resource preflight failed'
assert not (log/'execution.log').exists()
receipt=json.loads((W/'qies102/model-hash-result.json').read_text());original_model=W.parent/'qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf';st=original_model.stat();assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==receipt['identity'] and receipt['actual']=='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
stock=W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259';sources=json.loads((W/'qies100/stock-source-sha256.json').read_text());assert all(sha(stock/p)==value for p,value in sources.items())
assert sha(W.parent/'cortex-stream-forge-qies074/build-quortex/bin/quortex')=='329ebd647b04a2d6464461ddd94fc07a2691a948b2918d320ac1eb3703f9c101'
env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
# Independent stock metadata reading is data-only. Trap proves no GGML context allocation.
e=env.copy();fd=os.open(R/'allocation-watch.so',os.O_RDONLY|os.O_CLOEXEC);e['LD_PRELOAD']='/proc/self/fd/'+str(fd);model=W.parent/'qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf'
with (R/'reference-1203-descriptors.tsv').open('w') as f,(R/'reference-directory.log').open('w') as error:
 p=subprocess.run([str(W/'qies101/reference-extract'),str(model),'--describe'],env=e,pass_fds=(fd,),stdout=f,stderr=error,timeout=30)
os.close(fd)
assert p.returncode==0
assert 'cannot be preloaded' not in (R/'reference-directory.log').read_text()
assert len((R/'reference-1203-descriptors.tsv').read_text().splitlines())==1203
before=json.loads((R/'preservation.json').read_text());changes=[]
for path,entry in before.items():
 current=W/path
 if entry['kind']=='stock_library_compressed':
  import tarfile
  with tarfile.open(R/entry['archive']) as archive:
   f=archive.extractfile(Path(path).name);h=hashlib.sha256()
   while b:=f.read(1024*1024):h.update(b)
  assert h.hexdigest()==entry['sha256'] and current.stat().st_mode&0o777==0
  changes.append({'path':path,'original_sha256':entry['sha256'],'current_mode':'0o0','preserved_as':'verified compressed SDK archive'});continue
 preserved=R/'preserved'/path;assert sha(preserved)==entry['sha256']
 if entry['kind']=='worker':assert sha(current)==entry['sha256'] and not(current.stat().st_mode&0o111)
 changes.append({'path':path,'old_sha256':entry['sha256'],'current_sha256':sha(current),'current_mode':oct(current.stat().st_mode&0o777)})
branch=subprocess.check_output(['git','branch','--show-current'],cwd=W,env=env,text=True).strip();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,env=env,text=True).strip();assert branch=='experimental/ripple-65b-q4' and head=='8fac93bb2bfc4b2d7069c4044f2f71cd25f80388'
assert not subprocess.check_output(['git','diff'],cwd=W,env=env) and not subprocess.check_output(['git','diff','--cached'],cwd=W,env=env)
limits=[];relative=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'));cg=Path('/sys/fs/cgroup')/relative.lstrip('/')
for path in (cg,*cg.parents):
 row={'path':str(path)}
 for name in ('memory.max','memory.high','memory.swap.max'):
  if (path/name).exists():row[name]=(path/name).read_text().strip()
 if len(row)>1:limits.append(row)
(R/'status.json').write_text(json.dumps({'branch':branch,'HEAD':head,'host':host,'original_policy':vars(g.Policy()),'original_guard_sha256':GUARD,'rlimit_as':resource.getrlimit(resource.RLIMIT_AS),'rlimit_rss':resource.getrlimit(resource.RLIMIT_RSS),'cgroup_limits':limits,'stock_source_files_intact':len(sources),'real_execution_enabled':False,'real_numerical_equivalence_qualified':False,'live_positive_guard_qualified':False,'closure_scope':'approved QIES104 workers and normal retired-legacy dispatch; not a security boundary against owner recompilation/dlopen/permission restoration','preserved_changes':changes,'decision':'NO-GO real execution'},indent=2)+'\n')
print('Verified original guard refusal, 1203 independent descriptors without GGML allocation, preserved baseline/source/limits; real runtime remains locked')
