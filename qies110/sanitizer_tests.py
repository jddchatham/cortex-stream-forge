"""ASan/UBSan bounds, read-fault and ownership checks; synthetic sparse data only."""
from pathlib import Path
import os,subprocess,json,time
R=Path(__file__).resolve().parent;W=R.parent;O=R/'sanitizer-results-final';O.mkdir(exist_ok=False)
env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
command=['c++','-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+str(R),'-I'+str(W/'runtime-flashnext-layer-exp/ggml/include'),str(R/'adapter_faults.cpp'),'-lcrypto','-Wl,--wrap=pread','-o',str(O/'adapter-faults')]
with (O/'build.log').open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
start=time.monotonic()
with (O/'test.log').open('w') as out,(O/'test.err').open('w') as err:r=subprocess.run([O/'adapter-faults',O/'scratch'],env=env,stdout=out,stderr=err,timeout=60)
assert r.returncode==0,(r.returncode,(O/'test.err').read_text())
last=json.loads((O/'test.log').read_text().splitlines()[-1]);assert last['passed']==20 and last['open_files']==0
(O/'summary.json').write_text(json.dumps(dict(last,exit_status=r.returncode,seconds=time.monotonic()-start,compiler_command=command,asan=True,ubsan=True,leak_detection=True,instrumentation_scope='QIES110 range adapter and synthetic fault caller; not GGML shared libraries'),indent=2)+'\n')
print('PASS ASan/UBSan: 20 synthetic adapter cases, leak detection enabled')
