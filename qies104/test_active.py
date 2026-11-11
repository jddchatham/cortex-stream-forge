from pathlib import Path
import importlib.util,subprocess,json,os,hashlib
R=Path(__file__).resolve().parent;W=R.parent;s=importlib.util.spec_from_file_location('q104_sandbox',R/'sandbox.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
def run(label,args):
 os.environ['QIES104_EXPORT']=label;cmd=m.command(args)
 with (R/(label+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 if p.returncode:raise AssertionError((label,p.returncode,(R/(label+'.log')).read_text()))
 return R/'results'/label/'q104-out'
run('new-qies099',['/active/synthetic-layer-tests'])
run('new-qies102',['/active/adapter-tests',str(W/'qies102/fixture-0.plan'),str(W/'qies102/real-layer-00.plan')])
for label,stock in [('fixture','stock-captures'),('fixture-quantized','stock-captures-quantized-plain')]:
 out=run('candidate-'+label,['/active/worker','--synthetic',str(W/'qies102'/(label+'-')),str(W/'q104-out')]);count=0
 for p in sorted(out.glob('*.bin')):
  q=W/'qies100'/stock/p.name
  if not q.exists() and p.name.endswith('-4.bin'):q=q.with_name(p.name.replace('Qcur_normed','Qcur').replace('Kcur_normed','Kcur'))
  assert p.read_bytes()==q.read_bytes(),p.name;count+=1
 assert count==360;rows.append({'role':'candidate','case':label,'exact_comparisons':count})
 out=run('reference-'+label,['/active/reference-worker',str(W/'qies103'/('oneblock-'+label)/'synthetic.gguf'),str(W/'q104-out')]);count=0
 taps=['attn_norm','vres_in','Vcur','Qcur_normed','Kcur_normed','Qcur','Kcur','kq_soft_max','attn_out','attn_gate','attn_gated','attn_post_norm','ffn_inp','ffn_norm','ffn_swiglu','ffn_out','ffn_post_norm','l_out']
 for pos in range(4):
  for tap in taps:
   name=f'p{pos}-{tap}-0.bin';assert (out/name).read_bytes()==(W/'qies100'/stock/name).read_bytes(),name;count+=1
 assert count==72;rows.append({'role':'independent stock reference','case':label,'exact_comparisons':count})
(R/'active-comparisons.json').write_text(json.dumps({'profiles':rows,'candidate_total':720,'reference_total':144,'new_core_qies099_groups':14,'new_adapter_qies102_checks':16,'real_arithmetic':False},indent=2)+'\n')
print('PASS new guarded builds: original 14/16 suites; 720 candidate and 144 independent-reference exact synthetic comparisons')
