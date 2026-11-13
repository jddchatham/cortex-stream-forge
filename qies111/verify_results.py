"""Independent post-campaign identity, routing, masks, weight and state audit."""
from pathlib import Path
import json,hashlib,struct,sys,importlib.util
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('run111',R/'run.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
rows=[]
for rootarg in sys.argv[1:]:
 root=Path(rootarg).resolve();assert json.loads((root/'SUCCESS.json').read_text())['success']
 for summary in sorted(root.glob('layers-*/summary.json')):
  d=json.loads(summary.read_text());base=summary.parent;n=d['layers'];start=d['positions'][0];slots=1280 if start else 256;captures=0;weights=0;roles={role:{'rss':0,'pss':0,'anon':0,'file_pss':0,'controller_min_kib':None,'controller_max_kib':0,'controller_first_kib':None,'controller_last_kib':None,'controller_peak_pss_bytes':0,'controller_peak_anon_bytes':0,'after_cleanup_max_rss_kib':0,'weight_staging_max':0,'cpu_buffer_max':0,'workspace_max':0,'source_cache_max':0} for role in ('candidate','reference')}
  for pos in d['positions']:
   cp=base/f'candidate-token-{pos}.COMMITTED';rp=base/f'reference-token-{pos}.COMMITTED'
   for role,commit in [('candidate',cp),('reference',rp)]:
    a.verified(commit,n,slots,pos+1);metric=json.loads((commit/'METRICS.json').read_text());r=roles[role]
    for k,v in metric['worker_tree_peak_bytes'].items():r[k]=max(r[k],v)
    samples=metric['controller_samples'];values=[x['rss_kib']/1024 for x in samples];r['controller_peak_pss_bytes']=max(r['controller_peak_pss_bytes'],max(x['pss_kib'] for x in samples));r['controller_peak_anon_bytes']=max(r['controller_peak_anon_bytes'],max(x['anonymous_kib'] for x in samples));r['controller_min_kib']=min(values+[r['controller_min_kib']] if r['controller_min_kib'] is not None else values);r['controller_max_kib']=max(r['controller_max_kib'],max(values));r['controller_last_kib']=values[-1]
    if r['controller_first_kib'] is None:r['controller_first_kib']=values[0]
    for layer in range(n):
     job=commit/f'layer-{layer}';request=json.loads((job/'INPUT.json').read_text());result=json.loads((job/'result.json').read_text());assert result['status']==0 and not result['reasons'] and result['rlimits_unchanged']
     assert request['layer']==layer and request['position']==pos and request['slots']==slots
     source=commit/f'layer-{layer-1}/captures-0/final.f32' if layer else base/f'embedding-{pos}.f32';assert request['input_sha256']==a.sha(source) and request['original_sha256']==a.sha(base/f'embedding-{pos}.f32')
     if pos>start:assert request['previous']==a.sha(base/f'{role}-token-{pos-1}.COMMITTED/layer-{layer}/captures-0/state.kv')
     else:assert request['previous'] is None
     records=result['worker_records'];assert records[-1]['owned_resources']==0
     assert all(v.get('adapter_open_files_after_cleanup',0)==0 for v in records)
     r['after_cleanup_max_rss_kib']=max(r['after_cleanup_max_rss_kib'],max(v.get('rss_kib',0) for v in records if v.get('phase')=='after-cleanup'))
     for k,field in [('weight_staging_max','peak_weight_bytes'),('cpu_buffer_max','peak_cpu_tensor_buffer_bytes'),('workspace_max','ggml_workspace_bytes'),('source_cache_max','adapter_staging_peak')]:r[k]=max(r[k],max(v.get(field,0) for v in records))
   for layer in range(n):
    ca=cp/f'layer-{layer}/captures-0';re=rp/f'layer-{layer}/captures-0';assert {p.name for p in ca.iterdir()}=={p.name for p in re.iterdir()}
    for p in ca.iterdir():assert a.sha(p)==a.sha(re/p.name);captures+=1
    def inventory(commit):return {v['weight']:(v['sha256'],v['bytes']) for v in json.loads((commit/f'layer-{layer}/result.json').read_text())['worker_records'] if 'weight' in v}
    ci,ri=inventory(cp),inventory(rp);assert len(ci)==8 and ci==ri;weights+=8
    expected_ffn=144179200 if layer<15 or layer>=65 else 117964800;assert ci['ffn-gate'][1]==ci['ffn-up'][1]==ci['ffn-down'][1]==expected_ffn
    mask=struct.unpack('<'+str(slots)+'f',(ca/'mask.f32').read_bytes());expected=[0. if i<=pos and (layer%5==4 or pos-i<1024) else -float('inf') for i in range(slots)];assert list(mask)==expected
    probabilities=struct.unpack('<'+str(slots*80)+'f',(ca/'probabilities.f32').read_bytes());assert all(probabilities[h*slots+i]==0. for h in range(80) for i,v in enumerate(mask) if v==-float('inf'))
    if layer%5==4:assert (ca/'q-current.f32').read_bytes()==(ca/'q-norm.f32').read_bytes() and (ca/'k-current.f32').read_bytes()==(ca/'k-norm.f32').read_bytes()
    elif pos>0:assert (ca/'q-current.f32').read_bytes()!=(ca/'q-norm.f32').read_bytes()
  assert captures==d['comparisons']==n*2*26
  rows.append({'root':str(root),'layers':n,'positions':d['positions'],'capture_comparisons':captures,'weight_hash_comparisons':weights,'metrics':roles,'mask_rope_routing_and_state_checks':True,'kv_payload_per_layer':slots*4096,'persistent_stack_kv_bytes':slots*4096*n})
output=R/'verified-results.json';assert not output.exists(),'preserve previous verification';output.write_text(json.dumps({'success':True,'campaigns':rows,'real_weight_arithmetic':False,'synthetic_weight_profiles_repeated':True,'raw_controller_units_correction':'METRICS controller_samples fields named *_kib actually contain bytes from inherited tree_memory; final RSS KiB values divide by 1024 and controller PSS/anon byte values preserve those raw bytes'},indent=2)+'\n');print('PASS independent result audit:',sum(r['capture_comparisons'] for r in rows),'captures;',sum(r['weight_hash_comparisons'] for r in rows),'weight hashes')
