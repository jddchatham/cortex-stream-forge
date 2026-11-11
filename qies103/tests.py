"""Model-free contract and original-policy tests. No real-weight arithmetic."""
from pathlib import Path
import importlib.util,json,struct,math,tempfile,hashlib,subprocess,os,ast
R=Path(__file__).resolve().parent;W=R.parent

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=module('q103_contract',R/'contract.py');a=module('q103_audit',R/'audit.py');g=a.load_guard();policy=g.Policy();G=g.G;results=[]
def test(name,fn):fn();results.append({'name':name,'passed':True})
def require(ok):
 if not ok:raise AssertionError('contract failed')
contract=c.generate(R/'inputs')
test('deterministic_regeneration',lambda:require(c.generate(R/'inputs')==contract))
test('no_golden_outputs_fabricated',lambda:require(len(contract['outputs'])==21 and all(q['expected_sha256'] is None for q in contract['outputs'].values())))
x=(R/'inputs/input.f32').read_bytes();x0=(R/'inputs/x0.f32').read_bytes()
test('identical_activation_and_x0',lambda:require(x==x0 and len(x)==20480))
test('exact_dyadic_values',lambda:require(list(struct.unpack('<5120f',x))==[(i%17-8)/16 for i in range(5120)]))
mask=struct.unpack('<256f',(R/'inputs/mask.f32').read_bytes())
test('position_zero_sliding_mask',lambda:require(mask[0]==0 and all(v==-math.inf for v in mask[1:])))
test('empty_zero_f16_cache',lambda:require((R/'inputs/initial-kv.f16').read_bytes()==bytes(1048576) and contract['previous_kv_argument']=='-'))
test('position_advancement_contract',lambda:require(contract['position']==0 and contract['next_position']==1 and contract['tokens']==1))
test('output_dimension_contract',lambda:require(contract['outputs']['l_out.f32']['bytes']==20480 and contract['outputs']['kq_soft_max.f32']['bytes']==81920 and contract['outputs']['key-written.f16']['bytes']==2048))
test('quantized_weight_inventory',lambda:require(len(contract['tensors'])==15 and sum(q['bytes'] for q in contract['tensors'])==545752068))
test('unchanged_policy_values',lambda:require((policy.entry_ram,policy.entry_swap,policy.entry_disk,policy.min_ram,policy.min_swap,policy.min_disk,policy.max_rss,policy.max_seconds)==(6*G,4*G,30*G,G,2*G,20*G,7*G,14400)))
def entry(h):return h['available']>=policy.entry_ram and h['swap_free']>=policy.entry_swap and h['disk_free']>=policy.entry_disk and h['psi']<1
base=dict(available=6*G,swap_free=4*G,disk_free=30*G,psi=0)
test('pure_boundary_pass_no_launch',lambda:require(entry(base)))
for key,value in [('available',6*G-1),('swap_free',4*G-1),('disk_free',30*G-1),('psi',1)]:test('pure_boundary_fail_'+key,lambda key=key,value=value:require(not entry(dict(base,**{key:value}))))
source=(W.parent/'qies077-guard.py').read_text();native=(W/'qies102/probe.cpp').read_text();body=native[native.index('void real_run'):native.index('\nint main')]
test('original_preflight_precedes_launch_source_audit',lambda:require(source.index("raise GuardAbort('resource preflight failed')")<source.index("launch = ['systemd-run'")))
test('candidate_native_admission_precedes_execute_source_audit',lambda:require(body.index('r102::LayerSource source(b)')<body.index('auto result=r99::execute')))
tree=ast.parse((W/'qies102/runtime.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run_guard']
test('candidate_original_guard_no_policy_override',lambda:require(len(calls)==1 and len(calls[0].args)==2 and not calls[0].keywords))
q102=module('q103_q102_launcher',W/'qies102/runtime.py')
def missing():
 try:q102.qualifications(None,None)
 except ValueError:return
 raise AssertionError('missing qualifications accepted')
test('missing_reference_or_guard_receipt_blocks',missing)
# Actual default-policy refusal, once; later runs preserve and inspect that evidence.
def actual_refusal():
 log=R/'guard-preflight-rejection'
 if not log.exists():
  h=g.host(R);require(h['available']<policy.entry_ram)
  rc=g.run_guard(['/usr/bin/true'],log);require(rc!=0)
 result=json.loads((log/'result.json').read_text());require(result['ok'] is False and result['cleanup'] is True and result['reason']=='GuardAbort: resource preflight failed')
 require(not (log/'execution.log').exists())
test('actual_original_guard_default_policy_refusal',actual_refusal)
# Characterize unresolved alternate entrypoints; detecting them is a passing audit, not a qualification.
legacy=(W/'qies099/layer-cli.cpp').read_text();core=(W/'qies099/ricursive-layer.cpp').read_text();core_execute=core[core.index('Result execute('):];stock=(W/'qies100/stock_probe.cpp').read_text()
test('legacy_marker_entry_gap_detected',lambda:require('QIES099_RESOURCE_SCOPE' in legacy and '/proc/self/cgroup' not in legacy))
test('public_core_unguarded_api_gap_detected',lambda:require('mem_available()' not in core_execute))
test('stock_current_capture_is_synthetic_only',lambda:require('llama_model_n_layer(m)!=5' in stock and '16*1024*1024' in stock))
command=json.loads((R/'capsule-build.json').read_text())['command'];test('independent_reference_linkage',lambda:require(all('qies099' not in str(x) and 'qies102' not in str(x) for x in command)))
test('oneblock_synthetic_execution_evidence',lambda:require(json.loads((R/'capsule-comparison.json').read_text())['passed']==144))
# Corrupt only new temporary contract files; do not modify actual input or model files.
with tempfile.TemporaryDirectory(dir=R,prefix='contract-fault-') as t:
 p=Path(t);c.generate(p);(p/'input.f32').write_bytes(b'bad')
 def corruption():
  try:c.generate(p)
  except ValueError:return
  raise AssertionError('corrupted contract input accepted')
 test('corrupt_input_rejected',corruption)
def cleanup_suite():
 env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
 with (R/'adapter-cleanup-tests.log').open('w') as f:subprocess.run([str(W/'qies102/test-adapter'),str(R/'fixture-for-cleanup.plan'),str(W/'qies102/real-layer-00.plan')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
 require('PASS 16' in (R/'adapter-cleanup-tests.log').read_text())
test('existing_adapter_failure_cleanup_suite',cleanup_suite)
(R/'tests.json').write_text(json.dumps({'tests':results,'passed':len(results),'real_weight_arithmetic':False,'live_positive_paths_qualified':False,'alternate_entrypoints_guard_complete':False},indent=2)+'\n')
print('PASS',len(results),'contract/provenance/guard-audit tests; detected entrypoint gaps remain BLOCKERS')
