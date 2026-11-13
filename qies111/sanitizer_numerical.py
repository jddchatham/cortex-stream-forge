"""Instrument QIES111 caller/state code; GGML shared libraries remain uninstrumented."""
from pathlib import Path
import importlib.util,json,subprocess,os,sys,hashlib
R=Path(__file__).resolve().parent;B=R/'build';O=R/'sanitizer-numerical-results';O.mkdir(exist_ok=False);S=R/'sanitizer-build';S.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('q111',R/'run.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
manifest=json.loads((B/'manifest.json').read_text());cmd=list(manifest['workers']['candidate']['command']);cmd[1:1]=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-g'];cmd[cmd.index('-O2')]='-O1';cmd[-1]=str(S/'candidate')
with (O/'build.log').open('w') as f:subprocess.run(cmd,env=a.ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
(S/'candidate').chmod(0o600);(S/'marker').symlink_to(B/'marker');(S/'candidate-libs').symlink_to(B/'candidate-libs')
a.B=S;a.m.B=S;a.m.monitor.B=S
orig=a.m.monitor.namespace
def namespace(role,out,args,marker=True):
 c=orig(role,out,args,marker);i=c.index('--');return c[:i]+['--setenv','ASAN_OPTIONS','detect_leaks=1:halt_on_error=1','--setenv','UBSAN_OPTIONS','halt_on_error=1:print_stacktrace=1']+c[i:]
a.m.monitor.namespace=namespace
source=Path(sys.argv[1]).resolve()/'layers-2-start-0';passed=[]
for label,fault,status in [('continuation','none',0),('kv-update','kv.after-value-update',1),('serialization','state.after-serialization',1),('publication','before-publish',1)]:
 out=O/label;out.mkdir();res=a.launch(out,'candidate',0,256,1,source/'candidate-token-0.COMMITTED/layer-0/captures-0/state.kv',source/'embedding-1.f32',source/'embedding-1.f32',fault=fault,status=status)
 if not status:
  expected=source/'reference-token-1.COMMITTED/layer-0/captures-0';actual=out/'layer-0/captures-0';assert all(p.read_bytes()==(expected/p.name).read_bytes() for p in actual.iterdir())
 err=(out/'layer-0/stderr.log').read_text();assert 'ERROR: AddressSanitizer' not in err and 'runtime error:' not in err and 'LeakSanitizer' not in err
 passed.append({'name':label,'status':res['status'],'passed':True,'peak_tree_sampled':res['peak_tree_sampled']})
(O/'summary.json').write_text(json.dumps({'passed':len(passed),'cases':passed,'command':cmd,'worker_sha256':a.sha(S/'candidate'),'scope':'QIES111 C++ caller, state and adapter; prebuilt GGML libraries are not instrumented','asan':True,'ubsan':True,'leak_detection':True,'real_weights':False},indent=2)+'\n');print('PASS numerical caller ASan/UBSan:',len(passed))
