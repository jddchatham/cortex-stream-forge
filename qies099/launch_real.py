"""Prepared guard adapter. --check never loads model payloads; no policy overrides."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys
R=Path(__file__).resolve().parent.parent
GUARD=R.parent/'qies077-guard.py'
EXPECTED='ecffc24bd040473155445689fa3c8bc63a2e73e79f756af7f66cde3a1d5a95e2'
MODEL='9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc'
REFERENCE='240d68fcfe0c2a1f298c8e18ca181368b1914259'
TAPS=['attn_norm','vres_in','Vcur','Qcur_normed','Kcur_normed','Qcur','Kcur','kq_soft_max','attn_out','attn_gate','attn_gated','attn_post_norm','ffn_inp','ffn_norm','ffn_swiglu','ffn_out','ffn_post_norm','l_out']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_guard():
 if sha(GUARD)!=EXPECTED:raise ValueError('Guard identity changed')
 spec=importlib.util.spec_from_file_location('qies099_original_guard',GUARD)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 return module

def validate_reference(path,layer):
 ref=json.loads(path.read_text())
 required={'status':'RUNTIME_QUALIFIED_RICURSIVE_LAYER','model_sha256':MODEL,'reference_source':REFERENCE,'layer':layer,'context':256,'threads':1,'flash_attention':False,'kv_type':'F16'}
 if any(ref.get(k)!=v for k,v in required.items()):raise ValueError('Missing/unqualified/mismatched independent reference')
 if len(ref.get('reference_binary_sha256',''))!=64 or not ref.get('reference_library_sha256'):raise ValueError('Reference identities incomplete')
 for field in ('input','x0'):
  item=ref[field]
  if sha(item['path'])!=item['sha256'] or Path(item['path']).stat().st_size!=5120*4:raise ValueError('Reference input identity')
 if ref['previous_kv']!='-':
  if sha(ref['previous_kv'])!=ref['previous_kv_sha256']:raise ValueError('Previous KV identity')
 outputs={x+'.f32' for x in TAPS}|{'layer-output.f32','key-written.f16','value-written.f16'}
 if set(ref['outputs'])!=outputs:raise ValueError('Incomplete reference output coverage')
 for name,value in ref['outputs'].items():
  if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Invalid reference checksum')
 return ref

def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--layer',type=int,choices=(0,4),default=0);p.add_argument('--reference',type=Path,default=R/'qies099/reference/BLOCKED.json');p.add_argument('--run-root',type=Path)
 args=p.parse_args();guard=load_guard();host=guard.host(R.parent/'qies098-data')
 ready=host['available']>=6*guard.G and host['swap_free']>=4*guard.G and host['disk_free']>=30*guard.G and host['psi']<1
 error=None;ref=None
 try:ref=validate_reference(args.reference.resolve(),args.layer)
 except (OSError,ValueError,KeyError) as exc:error=str(exc)
 status={'resources_ready':ready,'host':host,'reference_ready':ref is not None,'reference_error':error,'inference_executed':False,'guard_policy_unchanged':True,'runtime_adapter_qualified':False}
 if args.check:
  print(json.dumps(status,indent=2));return 0 if ready and ref else 1
 if not ready or ref is None:raise SystemExit('BLOCKED: resource/reference prerequisites; no inference launched')
 if args.run_root is None or not args.run_root.is_dir() or (args.run_root.resolve()==Path('/tmp') or str(args.run_root.resolve()).startswith('/tmp/')):
  raise SystemExit('Persistent pre-existing SSD run-root required')
 # This adapter still requires independent harmless-child qualification in passing conditions.
 qualification=R/'qies099/reference/guard-adapter-qualified.json'
 if not qualification.exists():raise SystemExit('BLOCKED: guard adapter not qualified; do not fabricate a receipt')
 q=json.loads(qualification.read_text())
 if q.get('guard_sha256')!=EXPECTED or q.get('adapter_sha256')!=sha(__file__) or q.get('status')!='QUALIFIED_HARMLESS_CHILDREN':raise SystemExit('Guard-adapter qualification identity mismatch')
 plan=(R/f'qies099/checks/layer-{args.layer:02d}.plan').resolve();binary=(R/'qies099/checks/ricursive-layer').resolve()
 identity=json.loads((R/'qies099/checks/build-identity.json').read_text())
 if sha(binary)!=identity['cli_sha256']:raise SystemExit('Candidate binary identity changed')
 # No inference was invoked while preparing QIES099. The original monitor owns this job.
 run=args.run_root.resolve();output=run/'layer-output';log=run/'guard-log'
 command=['/usr/bin/env','-i','PATH=/usr/bin:/bin','HOME='+str(Path.home()),'LANG=C.UTF-8','QIES099_RESOURCE_SCOPE=1','OMP_NUM_THREADS=1',str(binary),'--run',str(plan),str(Path(ref['input']['path']).resolve()),str(Path(ref['x0']['path']).resolve()),('-' if ref['previous_kv']=='-' else str(Path(ref['previous_kv']).resolve())),str(output)]
 code=guard.run_guard(command,log)
 if code:return code
 # Reject zero-exit wrappers that did not actually produce the complete layer evidence.
 state=(output/'next.kv').read_bytes();import struct
 header=struct.unpack('<24Q',state[:192]);position=header[11];count=header[12];size=count*2
 if not 1<=position<=256 or count!=128*8*256 or len(state)!=192+2*size:raise ValueError('Wrong candidate KV output')
 written=position*128*8*2
 actual={x:sha(output/x) for x in ref['outputs'] if x.endswith('.f32')}
 actual['key-written.f16']=hashlib.sha256(state[192:192+written]).hexdigest();actual['value-written.f16']=hashlib.sha256(state[192+size:192+size+written]).hexdigest()
 matched=actual==ref['outputs']
 (run/'comparison.json').write_text(json.dumps({'all_bytes_match':matched,'actual':actual,'reference':ref['outputs']},indent=2)+'\n')
 return 0 if matched else 1
if __name__=='__main__':sys.exit(main())
