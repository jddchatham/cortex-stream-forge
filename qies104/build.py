from pathlib import Path
import json,subprocess,os,hashlib
R=Path(__file__).resolve().parent;W=R.parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
libs=W.parent/'cortex-stream-forge-qies074/build-quortex/bin';stock=W/'qies100/stock-build/bin';headers=W/'runtime-flashnext-layer-exp/ggml/include';manifest=json.loads((W/'qies099/checks/build-identity.json').read_text())['library_sha256']
for n,h in manifest.items():assert hashlib.sha256((libs/n).read_bytes()).hexdigest()==h
commands=[]
def compile(name,source,reference=False):
 includes=[R,headers] if not reference else [R,W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259/include',W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259/ggml/include']
 link=[] if name=='reference-worker' else [str(libs/n) for n in manifest] if not reference else [str(stock/n) for n in ('libllama.so','libggml.so','libggml-cpu.so','libggml-base.so')]
 cmd=['c++','-std=c++17','-O2','-Wall','-Wextra',*['-I'+str(p) for p in includes],str(R/'admission.cpp'),*map(str,source),'-Wl,-rpath,'+str(stock if reference else libs),*link,'-lcrypto','-o',str(R/name)]
 with (R/(name+'-build.log')).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 commands.append({'name':name,'command':cmd,'sha256':hashlib.sha256((R/name).read_bytes()).hexdigest()})
compile('worker',[R/'ricursive-layer.cpp',R/'bundle.cpp',R/'candidate.cpp'])
compile('direct-api',[R/'ricursive-layer.cpp',R/'real_api.cpp'])
compile('synthetic-layer-tests',[R/'ricursive-layer.cpp',R/'guarded_layer_tests.cpp'])
compile('synthetic-candidate',[R/'ricursive-layer.cpp',R/'synthetic_candidate.cpp'])
compile('adapter-tests',[R/'ricursive-layer.cpp',R/'bundle.cpp',R/'guarded_adapter_tests.cpp'])
compile('reference-worker',[R/'reference_front.cpp'])
(R/'builds.json').write_text(json.dumps(commands,indent=2)+'\n')
print('Built six workers with shared fail-closed admission; numerical source unchanged')

# Rebuild the private reference backend only when its source/SDK fingerprint changes.
import importlib.util,shutil
spec=importlib.util.spec_from_file_location('q104_build_sandbox',R/'sandbox.py');sandbox=importlib.util.module_from_spec(spec);spec.loader.exec_module(sandbox)
meta=json.loads((R/'engine.json').read_text()) if (R/'engine.json').exists() else {}
expected_sources={name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ('reference.cpp','admission.cpp','admission.h','fixture-catalog.inc')}
if meta.get('source_hashes')!=expected_sources or meta.get('stock_sdk_manifest_sha256')!=hashlib.sha256((R/'stock-sdk-manifest.json').read_bytes()).hexdigest():
 with (R/'engine-build.log').open('w') as f:subprocess.run(sandbox.command(['/usr/bin/python3','-I','-B','/active/build_engine.py']),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 shutil.copy2(R/'results/engine-build/reference-engine.gz',R/'reference-engine.gz');shutil.copy2(R/'results/engine-build/engine.json',R/'engine.json')
