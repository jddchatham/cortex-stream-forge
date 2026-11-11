from pathlib import Path
import json,subprocess,hashlib,os
R=Path(__file__).resolve().parent;W=R.parent;env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
assert subprocess.check_output(['git','branch','--show-current'],cwd=W,env=env).strip()==b'experimental/ripple-65b-q4'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,env=env).strip()==b'8fac93bb2bfc4b2d7069c4044f2f71cd25f80388'
lib=W/'qies100/stock-build/bin/libggml-base.so.0.29.0'
assert hashlib.sha256(lib.read_bytes()).hexdigest()=='f34fb8f64ea4efe5aa82894c3eaaa8a0cee38e63e0c31feafc1ecdb8afc3e519'
src=W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259'
cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-I',str(src/'ggml/include'),str(R/'reference_extract.cpp'),'-Wl,-rpath,'+str(lib.parent),str(lib),'-o',str(R/'reference-extract')]
with (R/'reference-build.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'reference-build-identity.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256((R/'reference_extract.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((R/'reference-extract').read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest()},indent=2)+'\n')
print('PASS independent stock metadata/range reader build')
