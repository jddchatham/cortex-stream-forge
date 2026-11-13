"""Deliberate SIGTERM of a guarded-namespace synthetic endpoint, never real weights."""
from pathlib import Path
import sys,os,json,time,subprocess,signal
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import support as s
O=R/'termination-result-final';O.mkdir(exist_ok=False);h=s.m.monitor.host();assert h['MemAvailable']>=1.5*1024**3 and h['SwapFree']>=2*1024**3 and h['disk_free']>=20*1024**3
s.bindings=[(R/'endpoint-results/input.f32','/state/input.f32')];command=s.namespace('candidate',O,['--synthetic','head','100257','none']);start=time.monotonic();peak=0;trigger=False;desc=[]
with (O/'stdout.jsonl').open('w') as out,(O/'stderr.log').open('w') as err:
 p=subprocess.Popen(command,env=s.ENV,stdout=out,stderr=err,start_new_session=True)
 try:
  while p.poll() is None:
   rows=s.m.monitor.tree_memory(p.pid);peak=max(peak,sum(x.get('Rss',0) for x in rows));assert peak<=256*1024**2
   assert time.monotonic()-start<30
   if (O/'staging-0/final-norm.f32').exists():
    trigger=True;desc=s.m.monitor.subtree(p.pid);os.killpg(p.pid,signal.SIGTERM);break
   time.sleep(.002)
  p.wait(timeout=5)
 finally:
  if p.poll() is None:s.m.monitor.terminate(p)
assert trigger and p.returncode!=0 and not (O/'captures-0').exists(),'termination boundary not exercised'
deadline=time.monotonic()+2
live=[]
while True:
 live=[]
 for pid in desc:
  try:
   status=Path(f'/proc/{pid}/stat').read_text().split()[2]
   if status!='Z':live.append((pid,status))
  except FileNotFoundError:pass
 if not live or time.monotonic()>deadline:break
 time.sleep(.01)
assert not live,('still-live descendants after bounded teardown',live)
(O/'summary.json').write_text(json.dumps({'passed':True,'command':command,'exit_status':p.returncode,'intentional_signal':'SIGTERM','normalization_stage_reached':trigger,'no_published_capture':True,'peak_subtree_rss_bytes':peak,'real_weights':False,'historical_crash_reproduced':False,'uncommitted_disk_evidence_retained':True},indent=2)+'\n');print('PASS intentional synthetic SIGTERM; exit',p.returncode)
