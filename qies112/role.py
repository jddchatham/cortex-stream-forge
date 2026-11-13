"""One synthetic branch controller; all numerical actors retain QIES111 admission."""
from pathlib import Path
import importlib.util,json,sys,os,signal,ctypes
# Per-process death signal, not a global system setting. Fail closed if parent died.
expected_parent=int(sys.argv[2]);assert os.getppid()==expected_parent
assert ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)==0
assert os.getppid()==expected_parent
R=Path(__file__).resolve().parent;W=R.parent
p=Path(sys.argv[1]);request=json.loads(p.read_text());assert request['synthetic_only'] and request['role'] in ('candidate','reference') and request['layers']==80
s=importlib.util.spec_from_file_location('q111',W/'qies111/run.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
prior=Path(request['prior']) if request['prior'] else None
out=a.transaction(Path(request['base']),request['role'],80,256,request['position'],prior,Path(request['embedding']))
print(json.dumps({'success':True,'checkpoint':str(out),'role':request['role'],'position':request['position']}),flush=True)
