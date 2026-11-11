"""Relocatable, source-built synthetic replay. Real weights are never mounted.
Frozen milestone sources are copied, never edited in place. Generated wrappers
only add sealed-domain admission; no numerical operators/assertions are changed.
"""
from pathlib import Path
import subprocess,os,json,hashlib,shutil,resource,sys,gzip,tarfile
STOCK='240d68fcfe0c2a1f298c8e18ca181368b1914259'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def selftest(root,out,numpy_python):
 env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
 state=json.loads((out/'state.json').read_text());src=Path(state['stock_source']);cpu=Path(state['cpu_libs']);stock=Path(state['stock_libs'])
 import datetime,uuid
 out=out/'replays'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
 work=out/'synthetic-work';active=out/'active';results=out/'results'
 # Never delete previous test artifacts, and never overwrite existing fixtures.
 if work.exists() or active.exists():raise RuntimeError('Use a new build output directory for each selftest (existing artifacts preserved)')
 for d in (work,active,results,work/'qies100',work/'qies103',work/'qies101',work/'plans'):d.mkdir(parents=True,exist_ok=True)
 marker=active/'marker';marker.write_text('QIES104_SEALED_SYNTHETIC_DOMAIN_V1\n')
 for p in (root/'qies104').glob('*'):
  if p.suffix in ('.h','.cpp'):shutil.copyfile(p,active/p.name)
 (active/'fixture-catalog.inc').write_text('')
 logs=[];builds=[]
 def run(cmd,label,check=True,**kw):
  p=subprocess.run(list(map(str,cmd)),env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=300,**kw)
  (results/(label+'.log')).write_text(p.stdout+p.stderr)
  if check and p.returncode:raise AssertionError((label,p.returncode,p.stdout[-1000:],p.stderr[-3000:]))
  logs.append({'test':label,'status':p.returncode});return p
 def compile(name,sources,role='cpu',admit=True):
  headers=[active,root/'runtime-flashnext-layer-exp/ggml/include'] if role=='cpu' else [active,src/'include',src/'ggml/include']
  libs=cpu if role=='cpu' else stock
  names=['ggml','ggml-cpu','ggml-base'] if role=='cpu' else ['llama','ggml','ggml-cpu','ggml-base']
  target=active/name
  cmd=['c++','-std=c++17','-O2','-Wall','-Wextra',*['-I'+str(h) for h in headers]]
  if admit:cmd+=[active/'admission.cpp']
  cmd += [*sources,'-Wl,-rpath,'+('/cpu-sdk' if role=='cpu' else '/stock-sdk'),*[libs/('lib'+n+'.so') for n in names],'-lcrypto','-o',target]
  run(cmd,'build-'+name);target.chmod(0o600)
  builds.append({'name':name,'sha256':digest(target),'role':role,'command':list(map(str,cmd))});return target
 def namespace(args,write=True,stock_enabled=False):
  cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr','--ro-bind','/etc','/etc','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/home','--ro-bind',active,'/active',('--bind' if write else '--ro-bind'),work,'/work','--ro-bind',cpu,'/cpu-sdk','--ro-bind',marker,'/qies104.synthetic','--setenv','PATH','/usr/bin:/bin','--setenv','LANG','C.UTF-8','--setenv','QIES104_SYNTHETIC','1','--setenv','QIES100_REFERENCE_PLAIN','1','--chdir','/work']
  if stock_enabled:cmd+=['--tmpfs','/stock-sdk','--tmpfs','/stock-engine']
  cmd+=['--','/usr/bin/python3','-I','-B','/active/boot.py',*args] if stock_enabled else ['--',*args]
  return list(map(str,cmd))
 # Quantizer only processes generated arrays, never GGUF/model files.
 quant=(root/'qies100/quantize_fixture.cpp').read_text()
 pos=quant.index('{',quant.index('int main('));quant=quant[:pos+1]+'q104::synthetic();'+quant[pos+1:]
 (active/'quant.cpp').write_text('#include "admission.h"\n'+quant)
 compile('quant',[active/'quant.cpp'])
 # A narrow proxy lets the unchanged fixture generator call the sealed quantizer.
 (work/'qies100'/('llama.cpp-'+STOCK)).symlink_to(src,target_is_directory=True)
 proxy='''#!/usr/bin/python3
import json,subprocess,sys
from pathlib import Path
c=json.loads(Path(__file__).with_name('quant-command.json').read_text());args=sys.argv[1:]
base=Path(__file__).resolve().parents[1]
for i in (3,4,5):
 p=Path(args[i]).resolve();p.relative_to(base);args[i]='/work/'+str(p.relative_to(base))
raise SystemExit(subprocess.run(c+args,stdin=subprocess.DEVNULL).returncode)
'''
 (work/'qies100/quantize-fixture').write_text(proxy);(work/'qies100/quantize-fixture').chmod(0o700)
 (work/'qies100/quant-command.json').write_text(json.dumps(namespace(['/lib64/ld-linux-x86-64.so.2','/active/quant'])))
 for source,dest in [('qies100/make_fixture.py','qies100/make_fixture.py'),('qies103/capsule_fixture.py','qies103/capsule_fixture.py')]:shutil.copyfile(root/source,work/dest)
 run([numpy_python,'-I','-B',work/'qies100/make_fixture.py'],'fixture-f32')
 run([numpy_python,'-I','-B',work/'qies100/make_fixture.py','--quantized'],'fixture-quantized')
 run([numpy_python,'-I','-B',work/'qies103/capsule_fixture.py'],'fixture-oneblock')
 # The source symlink is not needed for numerical execution. Remove to prevent
 # accidental exposure of the host source tree through the synthetic mount.
 (work/'qies100'/('llama.cpp-'+STOCK)).unlink()
 catalog={}
 for folder in [work/'qies100'/x for x in ('fixture','fixture-quantized')]+[work/'qies103'/x for x in ('oneblock-fixture','oneblock-fixture-quantized')]:
  for p in folder.iterdir():
   if p.is_file():catalog['/work/'+str(p.relative_to(work))]=digest(p)
 (active/'fixture-catalog.inc').write_text(''.join('{'+json.dumps(p)+','+json.dumps(h)+'},\n' for p,h in sorted(catalog.items())))
 (results/'fixture-catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
 def plan(path,kind,layer,rows):
  cfg=[layer,5120,40960,80,8,128,256,1024] if kind=='REAL' else [layer,256,512,4,2,64,256,2]
  lines=['QIES102-BUNDLE-1',kind,' '.join(map(str,cfg))]
  for q in rows:
   shape=q['shape'];lines.append(' '.join([json.dumps(q['name']),str(q['type']),str(shape[0]),str(shape[1] if len(shape)>1 else 1),str(q['offset']),str(q['bytes']),*map(str,q['identity']),q['sha256'],json.dumps(q['path'])]))
  path.write_text('\n'.join(lines)+'\n')
 for label in ('fixture','fixture-quantized'):
  folder=work/'qies100'/label;shapes=json.loads((folder/'shapes.json').read_text())
  for layer in range(5):
   rows=[]
   for name,shape in shapes.items():
    if not name.startswith(f'blk.{layer}.'):continue
    p=folder/(name+'.bin');st=p.stat();tp=folder/(name+'.type')
    rows.append(dict(name=name,shape=list(reversed(shape)),type=int(tp.read_text()) if tp.exists() else 0,offset=0,bytes=st.st_size,identity=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns],sha256=digest(p),path='/work/'+str(p.relative_to(work))))
   plan(work/'plans'/f'{label}-{layer}.plan','FIXTURE',layer,rows)
 rows=json.loads((root/'qies102/bundle-manifest.json').read_text())['tensors']
 for row in rows:row['path']='/forbidden/real/'+row['name'];row['identity']=[0,0,row['bytes'],0,0]
 plan(work/'plans/real-layer-00.plan','REAL',0,rows)
 compile('worker',[active/'ricursive-layer.cpp',active/'bundle.cpp',active/'candidate.cpp'])
 compile('direct-api',[active/'ricursive-layer.cpp',active/'real_api.cpp'])
 compile('synthetic-layer-tests',[active/'ricursive-layer.cpp',active/'guarded_layer_tests.cpp'])
 compile('synthetic-candidate',[active/'ricursive-layer.cpp',active/'synthetic_candidate.cpp'])
 compile('adapter-tests',[active/'ricursive-layer.cpp',active/'bundle.cpp',active/'guarded_adapter_tests.cpp'])
 compile('reference-worker',[active/'reference_front.cpp'])
 compile('reference-engine',[active/'reference.cpp'],'stock')
 # Fresh independent five-block stock captures; body/order unchanged, admission
 # wrapper and plain-buffer profile match the already tested QIES100 reference.
 text=(root/'qies100/stock_probe.cpp').read_text();pos=text.index('{',text.index('int main('))
 text=text[:pos+1]+'q104::synthetic();if(argc<3)throw std::runtime_error("arguments");q104::fixture_file(argv[1]);'+text[pos+1:]
 (active/'stock-five.cpp').write_text('#include "admission.h"\n'+text)
 compile('stock-five',[active/'stock-five.cpp'],'stock')
 # Keep reference ELF/SDK private to the sealed namespace, as in QIES104.
 sdkfiles={}
 with tarfile.open(active/'stock-sdk.tar.gz','w:gz') as archive:
  for p in stock.glob('*.so*'):
   archive.add(p,arcname=p.name,recursive=False)
   if not p.is_symlink():sdkfiles[p.name]=digest(p)
 for name in ('reference-engine','stock-five'):
  p=active/name
  with p.open('rb') as f,gzip.open(active/(name+'.gz'),'wb') as g:shutil.copyfileobj(f,g,1024*1024)
  p.unlink()
 (active/'sdk-hashes.json').write_text(json.dumps(sdkfiles))
 (active/'boot.py').write_text('''from pathlib import Path
import tarfile,gzip,shutil,hashlib,json,os,sys
with tarfile.open('/active/stock-sdk.tar.gz') as t:t.extractall('/stock-sdk',filter='data')
for p,h in json.loads(Path('/active/sdk-hashes.json').read_text()).items():assert hashlib.sha256(Path('/stock-sdk',p).read_bytes()).hexdigest()==h
for name in ('reference-engine','stock-five'):
 with gzip.open('/active/'+name+'.gz','rb') as f,Path('/stock-engine',name).open('wb') as g:shutil.copyfileobj(f,g,1024*1024)
 Path('/stock-engine',name).chmod(0o700)
os.execv(sys.argv[1],sys.argv[1:])
''')
 # Data-only independent GGUF extraction: unchanged original four fault tests.
 compile('reference-extract',[root/'qies101/reference_extract.cpp'],'stock',admit=False)
 shutil.copyfile(root/'qies101/test_reference_faults.py',work/'qies101/test_reference_faults.py')
 (work/'qies101/reference-extract').write_text('#!/bin/sh\nexec /lib64/ld-linux-x86-64.so.2 /active/reference-extract "$@"\n')
 (work/'qies101/reference-extract').chmod(0o700)
 run(namespace(['/usr/bin/python3','-I','-B','/work/qies101/test_reference_faults.py'],stock_enabled=True),'qies101-independent-four-faults')
 # The canonical fresh stock C ABI remains unavailable on the host after
 # packaging its verified bytes for synthetic-only namespace use.
 for p in stock.glob('libllama.so.*'):
  if not p.is_symlink():p.chmod(0)
 # Preserve inherited resource limits; no cgroup or threshold edits.
 before_limits={'as':resource.getrlimit(resource.RLIMIT_AS),'rss':resource.getrlimit(resource.RLIMIT_RSS)}
 elf='/lib64/ld-linux-x86-64.so.2'
 run(namespace([elf,'/active/synthetic-layer-tests']),'qies099-14-groups')
 run(namespace([elf,'/active/adapter-tests','/work/plans/fixture-0.plan','/work/plans/real-layer-00.plan']),'qies102-16-checks')
 comparisons=[]
 taps=['attn_norm','vres_in','Vcur','Qcur_normed','Kcur_normed','Qcur','Kcur','kq_soft_max','attn_out','attn_gate','attn_gated','attn_post_norm','ffn_inp','ffn_norm','ffn_swiglu','ffn_out','ffn_post_norm','l_out']
 for label in ('fixture','fixture-quantized'):
  run(namespace(['/stock-engine/stock-five',f'/work/qies100/{label}/synthetic.gguf',f'/work/stock-{label}'],stock_enabled=True),'stock-'+label)
  run(namespace([elf,'/active/worker','--synthetic',f'/work/plans/{label}-',f'/work/candidate-{label}']),'candidate-'+label)
  run(namespace([elf,'/active/reference-worker',f'/work/qies103/oneblock-{label}/synthetic.gguf',f'/work/reference-{label}'],stock_enabled=True),'reference-'+label)
  for role in ('candidate','reference'):
   a=work/(role+'-'+label);b=work/('stock-'+label);count=0
   for p in sorted(a.glob('*.bin')):
    if role=='reference' and not any(p.name==f'p{pos}-{tap}-0.bin' for pos in range(4) for tap in taps):continue
    q=b/p.name
    if not q.exists() and p.name.endswith('-4.bin'):q=q.with_name(p.name.replace('Qcur_normed','Qcur').replace('Kcur_normed','Kcur'))
    assert p.read_bytes()==q.read_bytes(),(label,role,p.name);count+=1
   assert count==(360 if role=='candidate' else 72),(role,count)
   comparisons.append(dict(case=label,role=role,byte_exact=count))
 # QIES101 original 38 tests are data-only, independent of model/gate.
 for name in ('extract.py','test_extract.py'):shutil.copyfile(root/'qies101'/name,work/'qies101'/name)
 run(['/usr/bin/python3','-I','-B',work/'qies101/test_extract.py'],'qies101-original-extraction-faults')
 # Same allocation detector and positive control as frozen guard-closure suite.
 run(['c++','-shared','-fPIC','-O2',root/'qies104/allocation_watch.cpp','-o',active/'allocation-watch.so'],'build-allocation-watch')
 control=compile('allocation-control',[root/'qies104/allocation_control.cpp'],admit=False)
 p=run(namespace(['/usr/bin/env','LD_PRELOAD=/active/allocation-watch.so',elf,'/active/allocation-control']),'allocation-control',check=False)
 assert p.returncode==97 and 'QIES104_TEST_FAILURE' in p.stderr,p.stderr
 cases=[('worker-real',['/active/worker','--real','missing','missing','missing','-','missing']),('worker-admission',['/active/worker','--admission','missing']),('worker-storage',['/active/worker','--storage','missing']),('worker-alternate',['/active/worker','--run','missing']),('worker-unknown',['/active/worker','--other']),('worker-empty',['/active/worker']),('reference-real',['/active/reference-worker','--real','missing']),('direct-execute',['/active/direct-api','execute']),('direct-file',['/active/direct-api','file']),('real-plan-synthetic',['/active/worker','--synthetic','/work/plans/real-layer-0','/work/forbidden-output'])]
 # Use an actual existing REAL plan prefix, so rejection is mode-based.
 shutil.copyfile(work/'plans/real-layer-00.plan',work/'plans/real-plan-0.plan')
 cases[-1]=('real-plan-synthetic',['/active/worker','--synthetic','/work/plans/real-plan-','/work/forbidden-output'])
 rejection=[]
 for label,args in cases:
  p=run(namespace(['/usr/bin/env','LD_PRELOAD=/active/allocation-watch.so',elf,*args]),'reject-'+label,check=False)
  assert p.returncode not in (0,97) and 'QIES104_TEST_FAILURE' not in p.stderr
  assert ('QIES104 BLOCKED' in p.stderr or 'synthetic mode refuses real bundle' in p.stderr),p.stderr
  rejection.append(dict(case=label,returncode=p.returncode,allocation_trap=False,diagnostic=p.stderr.strip()))
 for name in ('worker','direct-api','synthetic-layer-tests','adapter-tests','reference-worker'):
  p=run([elf,active/name], 'outside-domain-'+name,check=False)
  # RPATH is namespace-only; host loading itself must fail before arithmetic.
  assert p.returncode!=0
 # Verify namespace cannot observe home/models and inherits unchanged limits.
 check='import pathlib,resource,json;print(json.dumps({"home_entries":list(map(str,pathlib.Path("/home").iterdir())),"as":resource.getrlimit(resource.RLIMIT_AS),"rss":resource.getrlimit(resource.RLIMIT_RSS)}))'
 p=run(namespace(['/usr/bin/python3','-I','-c',check]),'sealed-filesystem')
 evidence=json.loads(p.stdout);assert not evidence['home_entries']
 assert tuple(evidence['as'])==before_limits['as'] and tuple(evidence['rss'])==before_limits['rss']
 # Deterministic layer-0 contract inputs: no real weights, no fabricated golden.
 import importlib.util,struct,math
 spec=importlib.util.spec_from_file_location('q105_contract',root/'qies103/contract.py');contractmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(contractmod)
 folder=work/'contract';folder.mkdir();contract=contractmod.generate(folder)
 assert contractmod.generate(folder)==contract
 assert len(contract['outputs'])==21 and all(q['expected_sha256'] is None for q in contract['outputs'].values())
 assert (folder/'input.f32').read_bytes()==(folder/'x0.f32').read_bytes()
 assert list(struct.unpack('<5120f',(folder/'input.f32').read_bytes()))==[(i%17-8)/16 for i in range(5120)]
 assert (folder/'initial-kv.f16').read_bytes()==bytes(1048576)
 mask=struct.unpack('<256f',(folder/'mask.f32').read_bytes());assert mask[0]==0 and all(v==-math.inf for v in mask[1:])
 assert (contract['position'],contract['next_position'],contract['tokens'])==(0,1,1)
 assert len(contract['tensors'])==15 and sum(q['bytes'] for q in contract['tensors'])==545752068
 (folder/'input.f32').write_bytes(b'corruption')
 try:contractmod.generate(folder)
 except ValueError:pass
 else:raise AssertionError('corrupt contract input accepted')
 # Archived independent descriptor capture compared with the index; this is
 # metadata evidence replay, not a fresh parse of the external real GGUF.
 descriptors={}
 for line in (root/'qies104/reference-1203-descriptors.tsv').read_text().splitlines():
  name,*values=line.split('\t');descriptors[name]=list(map(int,values))
 index=json.loads((root/'qies098-checks/ripple-tensor-index.json').read_text())
 assert len(descriptors)==len(index['tensors'])==1203
 for q in index['tensors']:
  assert descriptors[q['name']]==[q['offset'],q['bytes'],q['type'],*q['shape'],*[1]*(4-len(q['shape']))]
 # Disabled legacy builders must not silently become host numerical launchers.
 for relative in json.loads((root/'qies104/builder-list.json').read_text()):
  p=run(['/usr/bin/python3','-I','-B',root/relative],'retired-builder-'+relative.replace('/','_'),check=False)
  assert p.returncode!=0 and 'QIES104 BLOCKED' in p.stderr
 # Verify frozen source/evidence bytes, not historical machine-specific paths.
 manifest=json.loads((root/'qies105/frozen-source-manifest.json').read_text())
 entries=manifest.get('files',manifest)
 if isinstance(entries,list):
  for row in entries:assert digest(root/row['path'])==row['sha256']
 elif isinstance(entries,dict):
  for path,row in entries.items():assert digest(root/path)==(row['sha256'] if isinstance(row,dict) else row)
 result={'real_execution':False,'baseline':state['baseline'],'stock':state['stock'],'fresh_source_build':True,'comparisons':comparisons,'candidate_exact_total':sum(r['byte_exact'] for r in comparisons if r['role']=='candidate'),'stock_reference_exact_total':sum(r['byte_exact'] for r in comparisons if r['role']=='reference'),'qies099_groups':14,'qies102_checks':16,'qies101_fault_checks':38,'independent_extraction_fault_checks':4,'archived_descriptor_comparisons':1203,'contract_invariant_groups':9,'guard_rejections':rejection,'allocation_positive_control_exit':97,'namespace':evidence,'builds':builds,'logs':logs,'scope':'Portable replay differs from historical path/binary-hash retirement checks. No real model weight access or arithmetic. Historical evidence remains archived unchanged.'}
 (results/'portable-results.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Evidence directory: '+str(results),flush=True)
 print(json.dumps({k:result[k] for k in ('candidate_exact_total','stock_reference_exact_total','qies099_groups','qies102_checks','qies101_fault_checks','allocation_positive_control_exit','real_execution')},indent=2),flush=True)
