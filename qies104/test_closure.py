from pathlib import Path
import os,subprocess,json,hashlib,importlib.util,shutil,resource
R=Path(__file__).resolve().parent;W=R.parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
s=importlib.util.spec_from_file_location('q104_sandbox',R/'sandbox.py');sandbox=importlib.util.module_from_spec(s);s.loader.exec_module(sandbox)
subprocess.run(['c++','-shared','-fPIC','-O2',str(R/'allocation_watch.cpp'),'-o',str(R/'allocation-watch.so')],env=env,check=True)
# A space-free inherited-fd preload path is required on this workspace path.
subprocess.run(['c++','-std=c++17','-O2','-I'+str(W/'runtime-flashnext-layer-exp/ggml/include'),str(R/'allocation_control.cpp'),'-Wl,-rpath,'+str(W.parent/'cortex-stream-forge-qies074/build-quortex/bin'),str(W.parent/'cortex-stream-forge-qies074/build-quortex/bin/libggml-base.so'),'-o',str(R/'allocation-control')],env=env,check=True)
fd=os.open(R/'allocation-watch.so',os.O_RDONLY|os.O_CLOEXEC)
try:
 e=env.copy();e['LD_PRELOAD']='/proc/self/fd/'+str(fd)
 control=subprocess.run([str(R/'allocation-control')],env=e,pass_fds=(fd,),capture_output=True,text=True,timeout=30)
 assert control.returncode==97 and 'QIES104_TEST_FAILURE' in control.stderr and 'cannot be preloaded' not in control.stderr,control.stderr
finally:os.close(fd)
rows=[]
def reject(name,cmd,domain=False,token='QIES104 BLOCKED'):
 e=env.copy();e['LD_PRELOAD']='' # Filled below using an inherited descriptor
 if domain:
  e.pop('LD_PRELOAD');e['QIES104_EXPORT']=name
  cmd=sandbox.command(['/usr/bin/env','LD_PRELOAD=/active/allocation-watch.so',*cmd])
 if domain:
  p=subprocess.run(list(map(str,cmd)),env=e,capture_output=True,text=True,timeout=120)
 else:
  fd=os.open(R/'allocation-watch.so',os.O_RDONLY|os.O_CLOEXEC)
  try:
   e['LD_PRELOAD']='/proc/self/fd/'+str(fd)
   p=subprocess.run(list(map(str,cmd)),env=e,pass_fds=(fd,),capture_output=True,text=True,timeout=120)
  finally:os.close(fd)
 if 'cannot be preloaded' in p.stderr:raise AssertionError('allocation detector missing: '+p.stderr)
 if p.returncode==0 or p.returncode==97 or 'QIES104_TEST_FAILURE' in p.stderr or token not in p.stderr+p.stdout:raise AssertionError((name,p.returncode,p.stderr,p.stdout))
 rows.append({'case':name,'returncode':p.returncode,'message':(p.stderr+p.stdout).strip(),'ggml_allocation_trap_triggered':False,'real_arithmetic':False})
for name,args in [('candidate-real',['--real','missing','missing','missing','-','missing']),('candidate-admission',['--admission','missing']),('candidate-storage',['--storage','missing']),('candidate-alternate-run',['--run','missing']),('candidate-unknown',['--different']),('candidate-empty',[])]:reject(name,[R/'worker',*args])
reject('reference-real',[R/'reference-worker','--real','missing'])
reject('reference-unlabelled-real',[R/'reference-worker',str(W.parent/'qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf'),'missing'])
reject('direct-execute',[R/'direct-api','execute'])
reject('direct-file-constructor',[R/'direct-api','file'])
reject('synthetic-outside-domain',[R/'worker','--synthetic','missing','missing'])
reject('synthetic-candidate-outside-domain',[R/'synthetic-candidate','missing','missing'])
reject('unit-layer-outside-domain',[R/'synthetic-layer-tests'])
reject('unit-adapter-outside-domain',[R/'adapter-tests','missing','missing'])
# Forged old/new environment flags never grant admission.
old=env.copy();env.update(QIES102_QUALIFIED_GUARDED_RUN='1',QIES099_RESOURCE_SCOPE='1',QIES104_SYNTHETIC='1')
reject('forged-environment',[R/'worker','--real','missing','missing','missing','-','missing']);env=old
# Even in the synthetic domain real roles stay denied; missing files are never opened.
reject('reference-real-in-sandbox',['/active/reference-worker','--real','missing'],True)
reject('private-reference-direct-dispatch',['/stock-engine/reference-engine','--real','missing'],True)
reject('direct-api-in-sandbox',['/active/direct-api','execute'],True)
reject('fake-fixture-real-geometry',['/active/direct-api','fake-fixture'],True,'QIES104 synthetic geometry budget')
# REAL plan cannot enter synthetic dispatch. Only metadata is exposed here.
shutil.copy2(W/'qies102/real-layer-00.plan',R/'regression/root/q104-real-0.plan')
reject('real-plan-through-synthetic',['/active/worker','--synthetic',str(W/'q104-real-'),str(W/'q104-out')],True,'synthetic mode refuses real bundle')
rogue=R/'regression/root/rogue';rogue.mkdir(exist_ok=True);shutil.copy2(W/'qies103/oneblock-fixture/synthetic.gguf',rogue/'synthetic.gguf')
reject('unregistered-reference-file',['/active/reference-worker',str(W/'rogue/synthetic.gguf'),str(W/'q104-out')],True,'unregistered synthetic fixture')
for path in json.loads((R/'retire-list.json').read_text()):
 p=W/path;assert not(p.stat().st_mode&0o111)
 try:subprocess.run([str(p),'--real'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=10)
 except PermissionError:rows.append({'case':'retired-worker-'+path,'returncode':'EACCES','ggml_allocation_trap_triggered':False,'real_arithmetic':False})
 else:raise AssertionError('retired worker remained executable')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads((R/'preservation.json').read_text())[path]['sha256']
for path in json.loads((R/'builder-list.json').read_text()):reject('retired-builder-'+Path(path).parent.name,['/usr/bin/python3','-I','-B',W/path])
# Namespace cannot reach real models or raw tensor bundles; inherited rlimits stay unchanged.
code='from pathlib import Path;import resource,json;print(json.dumps({"model":Path("../qies098-data/models/Ripple-65B-Preview-AD-Q4_K_M.gguf").exists(),"tensor":Path("qies102/bundle/index/blk.0.attn_norm.weight.bin").exists(),"as":resource.getrlimit(resource.RLIMIT_AS),"rss":resource.getrlimit(resource.RLIMIT_RSS)}))'
p=subprocess.run(sandbox.command(['/usr/bin/python3','-c',code]),env=env,capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stderr;data=json.loads(p.stdout)
assert not data['model'] and not data['tensor'] and tuple(data['as'])==resource.getrlimit(resource.RLIMIT_AS) and tuple(data['rss'])==resource.getrlimit(resource.RLIMIT_RSS)
assert not (W/'missing').exists()
import ctypes
try:ctypes.CDLL(str(W/'qies100/stock-build/bin/libllama.so.0.9.0'))
except OSError as ex:rows.append({'case':'retired-stock-C-ABI','returncode':'EACCES','message':str(ex),'ggml_allocation_trap_triggered':False,'real_arithmetic':False})
else:raise AssertionError('legacy stock C ABI remained loadable')
(R/'closure-tests.json').write_text(json.dumps({'allocation_detector_positive_control_exit':97,'allocation_detector_positive_control_message':control.stderr,'rejections':rows,'passed_rejections':len(rows),'sealed_filesystem':data,'real_execution_enabled':False,'live_positive_guard_qualified':False,'legacy_workers_nonexecutable':True},indent=2)+'\n')
print('PASS',len(rows),'entry/dispatch/API rejections with allocation trap; model filesystem hidden; rlimits unchanged')
