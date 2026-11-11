from pathlib import Path
import importlib.util,json
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('q100_guard',R/'guard_adapter.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
h=dict(available=6*m.G,swap_free=4*m.G,disk_free=30*m.G,psi=0)
assert m.entry(h)
count=1
for key,value in [('available',6*m.G-1),('swap_free',4*m.G-1),('disk_free',30*m.G-1),('psi',1)]:
 assert not m.entry(dict(h,**{key:value}));count+=1
for args in [(),('missing-model',),('forged-reference',),('gpu',),('missing-checkpoint',)]:
 try:m.real_command(*args)
 except RuntimeError:pass
 else:raise AssertionError('real launch not blocked')
 count+=1
for invalid in ({},{'status':'compiled'},{'status':'RUNTIME_QUALIFIED_RICURSIVE_LAYER','source_revision':'wrong'}):
 try:m.validate_reference_files(invalid)
 except (ValueError,KeyError):pass
 else:raise AssertionError('invalid oracle accepted')
 count+=1
g=m.load_guard();p=g.Policy();assert (p.entry_ram,p.entry_swap,p.entry_disk)==(6*m.G,4*m.G,30*m.G)
assert (p.min_ram,p.min_swap,p.min_disk,p.max_rss)==(m.G,2*m.G,20*m.G,7*m.G);count+=1
result=json.loads((R/'guard-entry-rejection/result.json').read_text());assert not result['ok'] and result['reason']=='GuardAbort: resource preflight failed'
(R/'guard-tests.json').write_text(json.dumps({'pure_contract_checks':count,'actual_default_policy_rejection':result,'live_success_cleanup_qualification':False,'no_policy_overrides':True},indent=2)+'\n')
print('PASS',count,'contract checks + recorded unchanged-policy preflight rejection; no real-model launch')
