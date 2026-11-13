"""Resume actual synthetic layers after whole-token interruption; no model weights."""
from pathlib import Path
import importlib.util,sys,json,shutil
R=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('q111',R/'run.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
source=Path(sys.argv[1]).resolve()/'layers-2-start-0';O=R/'recovery-results';O.mkdir(exist_ok=False);cases=[]
for name,inject in [('prefix-completed',('before-layer',1)),('before-publication',('before-publication',0))]:
 b=O/name;b.mkdir();prior=b/'candidate-token-0.COMMITTED';shutil.copytree(source/prior.name,prior);original={p:a.sha(p) for p in prior.rglob('*') if p.is_file()};embedding=source/'embedding-1.f32'
 try:a.transaction(b,'candidate',2,256,1,prior,embedding,inject=inject)
 except RuntimeError:pass
 else:raise AssertionError('interruption not injected')
 stage=b/'candidate-token-1.UNCOMMITTED';assert stage.exists() and not (b/'candidate-token-1.COMMITTED').exists()
 completed={p.relative_to(stage):(p.stat().st_mtime_ns,a.sha(p)) for p in stage.glob('*/captures-0/*')}
 result=a.transaction(b,'candidate',2,256,1,prior,embedding)
 for p,(mtime,h) in completed.items():assert (result/p).stat().st_mtime_ns==mtime and a.sha(result/p)==h,'verified layer reevaluated'
 expected=source/'reference-token-1.COMMITTED'
 for layer in range(2):
  for p in (result/f'layer-{layer}/captures-0').iterdir():assert p.read_bytes()==(expected/f'layer-{layer}/captures-0'/p.name).read_bytes()
 snapshot={p:(p.stat().st_mtime_ns,a.sha(p)) for p in result.rglob('*') if p.is_file()}
 assert a.transaction(b,'candidate',2,256,1,prior,embedding)==result
 assert all((p.stat().st_mtime_ns,a.sha(p))==old for p,old in snapshot.items()),'already committed token re-executed'
 assert all(a.sha(p)==h for p,h in original.items())
 cases.append({'name':name,'passed':True,'byte_exact_captures':52,'verified_prefix_not_reevaluated':True,'repeat_commit_is_idempotent':True,'prior_state_unchanged':True})
(O/'summary.json').write_text(json.dumps({'passed':len(cases),'cases':cases,'synthetic_only':True},indent=2)+'\n');print('PASS actual synthetic restart:',len(cases),'cases')
