from pathlib import Path
import sys,json,subprocess,os,struct
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import support as s
B=R/'build';O=R/'sanitizer-results';O.mkdir(exist_ok=False);S=R/'sanitizer-build';S.mkdir(exist_ok=False)
cmd=list(json.loads((B/'manifest.json').read_text())['workers']['candidate']['command']);cmd[1:1]=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-g'];cmd[cmd.index('-O2')]='-O1';cmd[-1]=str(S/'candidate')
with (O/'build.log').open('w') as f:subprocess.run(cmd,env=s.ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
(S/'candidate').chmod(0o600);(S/'candidate-libs').symlink_to(B/'candidate-libs');(S/'marker').symlink_to(B/'marker');s.B=S;s.m.B=S;s.m.monitor.B=S
old=s.m.monitor.namespace
def ns(role,out,args,marker=True):
 c=old(role,out,args,marker);i=c.index('--');return c[:i]+['--setenv','ASAN_OPTIONS','detect_leaks=1:halt_on_error=1','--setenv','UBSAN_OPTIONS','halt_on_error=1:print_stacktrace=1']+c[i:]
s.m.monitor.namespace=ns
rows=[]
for label,op,token,fault,status in [('embedding','embedding',100257,'none',0),('head','head',100257,'none',0),('last-vocab','embedding',100351,'none',0),('bad-ID','embedding',100352,'none',1),('embedding-failure','embedding',0,'embedding.after-compute',1),('head-failure','head',0,'head.after-compute',1),('publish-failure','head',0,'before-publish',1)]:
 inp=R/'endpoint-results/input.f32' if op=='head' else None;result=s.endpoint(O,label,'candidate',op,token,inp,fault=fault,status=status)
 if status==0:
  expected=R/'endpoint-results'/('reference-head' if op=='head' else f'reference-embedding-{token}');s.compare(O/label,expected)
 err=(O/label/'stderr.log').read_text();assert 'ERROR: AddressSanitizer' not in err and 'runtime error:' not in err and 'LeakSanitizer' not in err;rows.append({'label':label,'status':result['status'],'passed':True})
(O/'summary.json').write_text(json.dumps({'passed':len(rows),'tests':rows,'asan':True,'ubsan':True,'leak_detection':True,'scope':'new QIES112 C++ caller/fixtures; prebuilt GGML libraries not instrumented','real_weights':False,'command':cmd},indent=2)+'\n');print('PASS Q112 endpoint sanitizer',len(rows))
