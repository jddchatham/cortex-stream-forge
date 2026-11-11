from pathlib import Path
import subprocess,os,json,hashlib
R=Path(__file__).resolve().parent;W=R.parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
assert subprocess.check_output(['git','branch','--show-current'],cwd=W,env=env).strip()==b'experimental/ripple-65b-q4'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,env=env).strip()==b'8fac93bb2bfc4b2d7069c4044f2f71cd25f80388'
expected=json.loads((W/'qies099/checks/build-identity.json').read_text())['library_sha256'];libs=W.parent/'cortex-stream-forge-qies074/build-quortex/bin'
for n,h in expected.items():assert hashlib.sha256((libs/n).read_bytes()).hexdigest()==h
cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic','-I'+str(W/'runtime-flashnext-layer-exp/ggml/include'),'-I'+str(W/'qies099'),str(W/'qies099/ricursive-layer.cpp'),str(R/'bundle.cpp'),str(R/'probe.cpp'),'-Wl,-rpath,'+str(libs),*[str(libs/n) for n in expected],'-o',str(R/'probe')]
with (R/'build.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
testcmd=cmd.copy();testcmd[testcmd.index(str(R/'probe.cpp'))]=str(R/'tests.cpp');testcmd[-1]=str(R/'test-adapter')
with (R/'test-build.log').open('w') as f:subprocess.run(testcmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'build-identity.json').write_text(json.dumps({'command':cmd,'binary_sha256':hashlib.sha256((R/'probe').read_bytes()).hexdigest(),'libraries':expected,'test_command':testcmd,'test_binary_sha256':hashlib.sha256((R/'test-adapter').read_bytes()).hexdigest(),'qies099_source_sha256':hashlib.sha256((W/'qies099/ricursive-layer.cpp').read_bytes()).hexdigest()},indent=2)+'\n')
print('PASS CPU bundle adapter build')
