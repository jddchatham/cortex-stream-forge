"""Storage/controller tests with simulated states; no numerical execution."""
from pathlib import Path
import importlib.util,json,hashlib,struct,tempfile,os,shutil
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('orchestration111',R/'run.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
O=R/os.environ.get('QIES111_CONTRACT_RESULTS','contract-results-final');O.mkdir(exist_ok=False);passed=[]
def fixture_state(layer,slots,pos):
 h=struct.pack('<8Q',0x32564b3930315155,1,layer,slots,pos,slots*1024,slots*1024,109);payload=bytearray(slots*4096);struct.pack_into('<I',payload,layer*4,layer+pos+1);return h+hashlib.sha256(h+payload).hexdigest().encode()+payload
def fake(root,role,layer,slots,pos,previous,input_path,original,**kwargs):
 out=root/f'layer-{layer}';cap=out/'captures-0';cap.mkdir(parents=True,exist_ok=True)
 (cap/'state.kv').write_bytes(fixture_state(layer,slots,pos+1));(cap/'final.f32').write_bytes(input_path.read_bytes())
 (out/'OUTPUT.json').write_text(json.dumps({p.name:a.sha(p) for p in cap.iterdir()}))
 return {'peak_tree_sampled':dict(rss=0,pss=0,anon=0,file_pss=0)}
a.launch=fake
def check(name,fn,reject=False):
 try:fn();ok=not reject
 except (AssertionError,RuntimeError,FileNotFoundError,ValueError,KeyError):ok=reject
 assert ok,name;passed.append(name)
# A single input inode must be mounted at both activation contract destinations.
a.bindings=[(O/'same-input','/state/input.f32'),(O/'same-input','/state/original.f32')]
cmd=a.namespace('candidate',O,['--real']);assert '/state/input.f32' in cmd and '/state/original.f32' in cmd;passed.append('duplicate-source-distinct-bind-destinations')
emb=O/'embedding.f32';emb.write_bytes(bytes(5120*4))
base=O/'normal';base.mkdir();prior=a.transaction(base,'candidate',8,256,0,None,emb);previous={p:a.sha(p) for p in prior.rglob('*') if p.is_file()}
check('valid all-layer commit',lambda:a.verified(prior,8,256,1))
for boundary in ('before-layer','after-layer'):
 for layer in range(8):
  b=O/(boundary+str(layer));b.mkdir();cp=b/prior.name;shutil.copytree(prior,cp)
  check(boundary+str(layer),lambda:a.transaction(b,'candidate',8,256,1,cp,emb,inject=(boundary,layer)),True)
  assert not (b/'candidate-token-1.COMMITTED').exists()
  for p,h in previous.items():assert a.sha(cp/p.relative_to(prior))==h
for boundary in ('before-manifest','before-publication'):
 b=O/boundary;b.mkdir();cp=b/prior.name;shutil.copytree(prior,cp)
 check(boundary,lambda:a.transaction(b,'candidate',8,256,1,cp,emb,inject=(boundary,0)),True)
 assert not (b/'candidate-token-1.COMMITTED').exists()
 check('unpublished-'+boundary,lambda:a.verified(b/'candidate-token-1.UNCOMMITTED',8,256,2),True)
 # A supported retry safely completes the same token using prior authoritative state.
 final=a.transaction(b,'candidate',8,256,1,cp,emb);check('retry-'+boundary,lambda:a.verified(final,8,256,2))
check('cross-layer checkpoint',lambda:a.state(prior/'layer-0/captures-0/state.kv',1,256,1),True)
check('position drift',lambda:a.state(prior/'layer-0/captures-0/state.kv',0,256,2),True)
check('wrong capacity',lambda:a.state(prior/'layer-0/captures-0/state.kv',0,1280,1),True)
check('cross-session prior',lambda:a.transaction(O,'candidate',8,256,1,prior,emb),True)
check('cross-role prior',lambda:a.transaction(base,'reference',8,256,1,prior,emb),True)
cp=base/'corrupted.kv';cp.write_bytes((prior/'layer-0/captures-0/state.kv').read_bytes()[:-1]);check('truncated state',lambda:a.state(cp,0,256,1),True)
blob=bytearray((prior/'layer-0/captures-0/state.kv').read_bytes());blob[-1]^=1;cp.write_bytes(blob);check('checksum corruption',lambda:a.state(cp,0,256,1),True)
for p,h in previous.items():assert a.sha(p)==h
(O/'summary.json').write_text(json.dumps({'passed':len(passed),'tests':passed,'model_free':True,'numerical_execution':False,'prior_checkpoints_unchanged':True},indent=2)+'\n');print('PASS controller contracts:',len(passed))
