from pathlib import Path
import subprocess,os,json,hashlib
r=Path(__file__).resolve().parent.parent;o=r/'qies099/checks';src=r/'runtime-flashnext-layer-exp';libs=r.parent/'cortex-stream-forge-qies074/build-quortex/bin'
env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
assert subprocess.check_output(['git','-C',str(r),'branch','--show-current'],env=env).strip()==b'experimental/ripple-65b-q4'
assert subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],env=env).strip()==b'8fac93bb2bfc4b2d7069c4044f2f71cd25f80388'
expected={'libggml.so.0.25.3':'1f2380dce5820582130cc49e495f72d2f9b4d0e9465233416ef6d6f021c2354c','libggml-cpu.so.0.25.3':'1c56b25a1d87db8beeb150fb7d1c80f3334f3859a2fd6e6262cd417635e39f54','libggml-base.so.0.25.3':'77d90abebddc1467fbcd8a31949a9d056c677382c14e9ea32fa0a1027611f449'}
for name,h in expected.items():assert hashlib.sha256((libs/name).read_bytes()).hexdigest()==h
cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic','-I'+str(src/'ggml/include'),str(r/'qies099/ricursive-layer.cpp'),str(r/'qies099/test-layer.cpp'),'-Wl,-rpath,'+str(libs),*[str(libs/n) for n in expected],'-o',str(o/'test-layer')]
with (o/'build.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
cli=cmd.copy();cli[cli.index(str(r/'qies099/test-layer.cpp'))]=str(r/'qies099/layer-cli.cpp');cli[-1]=str(o/'ricursive-layer')
with (o/'cli-build.log').open('w') as f:subprocess.run(cli,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(o/'build-identity.json').write_text(json.dumps({'commands':[cmd,cli],'library_sha256':expected,'binary_sha256':hashlib.sha256((o/'test-layer').read_bytes()).hexdigest(),'cli_sha256':hashlib.sha256((o/'ricursive-layer').read_bytes()).hexdigest()},indent=2)+'\n')
with (o/'tests.log').open('w') as f:subprocess.run([str(o/'test-layer')],cwd=o,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
print((o/'tests.log').read_text())
