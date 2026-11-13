"""Build ONLY fixed procedural-synthetic workers; never changes frozen files."""
from pathlib import Path
import subprocess,os,json,hashlib,shutil
R=Path(__file__).resolve().parent;W=R.parent;B=R/'build';B.mkdir(exist_ok=True)
env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
statefile=Path(os.environ.get('QIES106_SDK_STATE',str(W.parent/'qies105-publication/.qies105-build/state.json'))).absolute()
state=json.loads(statefile.read_text());stock=Path(state['stock_source'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=json.loads((W/'qies100/stock-source-sha256.json').read_text())
assert all(sha(stock/p)==h for p,h in source.items())
subprocess.run(['/usr/bin/python3','-I','-B',R/'account.py'],env=env,check=True)
manifest={'parent':subprocess.check_output(['git','-C',W,'rev-parse','HEAD'],env=env,text=True).strip(),'stock_source':state['stock'],'stock_source_files_verified':len(source),'compiler':subprocess.check_output(['c++','--version'],env=env,text=True),'workers':{},'real_weights':False,'source_hashes':{str(p.relative_to(W)):sha(p) for p in [R/'stream.cpp',R/'reference.cpp',R/'fixture.h',R/'telemetry.h',R/'candidate-cases.inc',R/'reference-cases.inc',W/'qies104/admission.cpp',W/'qies104/admission.h',W/'qies104/fixture-catalog.inc']}}
for name,key,cpp in [('candidate','cpu_libs','stream.cpp'),('reference','stock_libs','reference.cpp')]:
 libs=B/(name+'-libs');libs.mkdir(exist_ok=True);original=Path(state[key]);hashes={}
 for linkname in ('libggml.so','libggml-cpu.so','libggml-base.so'):
  p=(original/linkname).resolve();target=libs/p.name
  if not target.exists():shutil.copyfile(p,target)
  assert sha(target)==sha(p);hashes[p.name]=sha(p)
  for alias in (linkname,linkname+'.0'):
   t=libs/alias
   if not t.exists():t.symlink_to(p.name)
 headers=W/'runtime-flashnext-layer-exp/ggml/include' if name=='candidate' else stock/'ggml/include'
 command=['c++','-std=c++17','-O2','-Wall','-Wextra','-I'+str(R),'-I'+str(W/'qies104'),'-I'+str(headers),str(R/cpp),str(W/'qies104/admission.cpp'),*[str(libs/n) for n in ('libggml.so','libggml-cpu.so','libggml-base.so')],'-lcrypto','-Wl,-rpath,/synthetic-sdk','-o',str(B/name)]
 with (B/(name+'-build.log')).open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 (B/name).chmod(0o600)
 manifest['workers'][name]={'command':command,'sha256':sha(B/name),'libraries':hashes,'library_source':str(original),'headers':str(headers),'no_llama_library':True,'independent_numerical_source':cpp,'only_shared_source':'unchanged admission.cpp plus input/telemetry headers; no candidate execution code in reference'}
(B/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(B/'marker').write_text('QIES104_SEALED_SYNTHETIC_DOMAIN_V1\n')
print('Built fixed-case procedural synthetic candidate and independent whole-matrix stock observer; host executability disabled; no real-weight loader')
