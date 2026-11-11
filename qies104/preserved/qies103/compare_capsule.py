from pathlib import Path
import json,os,subprocess,hashlib,resource
R=Path(__file__).resolve().parent;W=R.parent;S=W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259';B=W/'qies100/stock-build/bin';env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-I'+str(S/'include'),'-I'+str(S/'ggml/include'),str(R/'stock_oneblock_probe.cpp'),'-Wl,-rpath,'+str(B),*[str(B/n) for n in ('libllama.so','libggml.so','libggml-cpu.so','libggml-base.so')],'-o',str(R/'stock-oneblock-probe')]
with (R/'capsule-build.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'capsule-build.json').write_text(json.dumps({'command':cmd,'binary_sha256':hashlib.sha256((R/'stock-oneblock-probe').read_bytes()).hexdigest(),'reference_source':'240d68fcfe0c2a1f298c8e18ca181368b1914259','quortex_linked':False},indent=2)+'\n')
rows=[]
for label,original in [('fixture','stock-captures'),('fixture-quantized','stock-captures-quantized-plain')]:
 e=env.copy();e['QIES100_REFERENCE_PLAIN']='1';out=R/('captures-'+label)
 with (R/(label+'-reference.log')).open('w') as f:subprocess.run([str(R/'stock-oneblock-probe'),str(R/('oneblock-'+label)/'synthetic.gguf'),str(out)],env=e,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
 taps=['attn_norm','vres_in','Vcur','Qcur_normed','Kcur_normed','Qcur','Kcur','kq_soft_max','attn_out','attn_gate','attn_gated','attn_post_norm','ffn_inp','ffn_norm','ffn_swiglu','ffn_out','ffn_post_norm','l_out']
 for pos in range(4):
  for tap in taps:
   name=f'p{pos}-{tap}-0.bin';a=(out/name).read_bytes();b=(W/'qies100'/original/name).read_bytes();assert a==b,(label,name);rows.append({'case':label,'name':name,'byte_exact':True,'bytes':len(a),'sha256':hashlib.sha256(a).hexdigest()})
(R/'capsule-comparison.json').write_text(json.dumps({'comparisons':rows,'passed':len(rows),'synthetic_only':True,'real_reference_qualified':False},indent=2)+'\n');print('PASS',len(rows),'one-layer stock vs original five-layer stock layer-0 comparisons; synthetic only')
