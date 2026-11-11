"""Read-only original-policy snapshot. Does not run a child or change any limit."""
from pathlib import Path
import argparse,importlib.util,sys,hashlib,json,resource
p=argparse.ArgumentParser();p.add_argument('--storage',type=Path,required=True);a=p.parse_args()
path=Path(__file__).resolve().parent/'guards/qies077-guard.py'
if hashlib.sha256(path.read_bytes()).hexdigest()!='ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2':raise SystemExit('BLOCKED: original guard identity')
s=importlib.util.spec_from_file_location('q105_original_guard',path);g=importlib.util.module_from_spec(s);sys.modules[s.name]=g;s.loader.exec_module(g)
h=g.host(a.storage.resolve());policy=g.Policy();issues=[]
for name,value in [('available',policy.entry_ram),('swap_free',policy.entry_swap),('disk_free',policy.entry_disk)]:
 if h[name]<value:issues.append(name+' below original entry threshold')
if h['psi']>=1:issues.append('memory PSI full avg10 >= 1')
limits=[]
try:
 rel=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
 cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
 for q in (cg,*cg.parents):
  if not (q/'memory.events').exists():continue
  row={'path':str(q)}
  for name,current,needed in [('memory.max','memory.current',7*g.G),('memory.high','memory.current',7*g.G),('memory.swap.max','memory.swap.current',2*g.G)]:
   v=(q/name).read_text().strip();c=int((q/current).read_text());row[name]=v;row[current]=c
   if v!='max' and int(v)-c<needed:issues.append('insufficient ancestor headroom '+str(q/name))
  limits.append(row)
 if not limits:issues.append('no cgroup-v2 memory accounting')
except (OSError,StopIteration) as e:issues.append('cannot inspect cgroups: '+str(e))
print(json.dumps({'host':h,'original_policy':vars(policy),'ancestors':limits,'rlimit_as':resource.getrlimit(resource.RLIMIT_AS),'rlimit_rss':resource.getrlimit(resource.RLIMIT_RSS),'resource_snapshot_pass':not issues,'issues':issues,'real_execution_enabled':False,'note':'Diagnostic only. Actual original run_guard rechecks the user-manager app.slice ancestors; no positive runtime qualification is implied.'},indent=2))
raise SystemExit(1 if issues else 0)
