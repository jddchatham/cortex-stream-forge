from pathlib import Path
import importlib.util,sys,json,copy,tempfile,os,resource
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('range_extract',R/'extract.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
results=[]
def base():
 return {'schema':'QIES098_RICURSIVE_READONLY_INDEX_v1','architecture':'ricursive','model_filename':'fixture','tensors':[{'name':'blk.0.attn_q.weight','shape':[256,2],'type':12,'offset':32,'relative_offset':0,'bytes':288,'shard':'fixture','quantization':'q4_K'},{'name':'blk.1.attn_k.weight','shape':[32,2],'type':8,'offset':320,'relative_offset':288,'bytes':68,'shard':'fixture','quantization':'q8_0'}]}
def validate(i):return m.validate(i,512,start=32,expected_count=2)
def test(name,fn):
 fn();results.append({'name':name,'passed':True})
def rejects(fn,classes=(ValueError,KeyError,EOFError,OSError)):
 try:fn()
 except classes:return
 raise AssertionError('unexpected acceptance')
def mutated(fn):
 i=base();fn(i);rejects(lambda:validate(i))
test('valid_descriptors',lambda:validate(base()))
mutations={
 'negative_offset':lambda i:i['tensors'][0].update(offset=-1),
 'offset_overflow':lambda i:i['tensors'][0].update(offset=2**64),
 'shape_overflow':lambda i:i['tensors'][0].update(shape=[2**32,2**32]),
 'shape_zero':lambda i:i['tensors'][0].update(shape=[0]),
 'bool_type':lambda i:i['tensors'][0].update(type=True),
 'unsupported_type':lambda i:i['tensors'][0].update(type=99),
 'wrong_length':lambda i:i['tensors'][0].update(bytes=289),
 'wrong_quantization':lambda i:i['tensors'][0].update(quantization='q5_K'),
 'relative_mismatch':lambda i:i['tensors'][0].update(relative_offset=32),
 'unaligned_offset':lambda i:i['tensors'][0].update(offset=33,relative_offset=1),
 'wrong_shard':lambda i:i['tensors'][0].update(shard='other'),
 'quant_row_misalignment':lambda i:i['tensors'][0].update(shape=[255,2]),
 'overlap':lambda i:i['tensors'][1].update(offset=32,relative_offset=0),
 'past_eof':lambda i:i['tensors'][1].update(offset=512,relative_offset=480),
 'missing_descriptor':lambda i:i['tensors'].pop(),
 'duplicate_name':lambda i:i['tensors'][1].update(name=i['tensors'][0]['name']),
 'unsafe_name':lambda i:i['tensors'][0].update(name='../escape'),
 'corrupt_schema':lambda i:i.update(schema='corrupt')}
for name,fn in mutations.items():test(name,lambda fn=fn:mutated(fn))
test('bool_alignment',lambda:rejects(lambda:m.validate(base(),512,start=32,alignment=True,expected_count=2)))
test('invalid_alignment',lambda:rejects(lambda:m.validate(base(),512,start=32,alignment=3,expected_count=2)))
with tempfile.TemporaryDirectory(dir=R,prefix='synthetic-') as directory:
 d=Path(directory);p=d/'fixture';payload=bytes(range(256))*2;p.write_bytes(payload)
 def make():return m.Extractor(p,base(),m.identity(p.stat()),start=32,expected_count=2)
 before_fds=len(os.listdir('/proc/self/fd'))
 with make() as x:
  out=d/'valid';x.extract('blk.0.attn_q.weight',out);assert out.read_bytes()==payload[32:320];assert x.bytes_read==288
 test('exact_requested_bytes',lambda:None)
 def missing():
  with make() as x:rejects(lambda:x.extract('missing',d/'missing'))
 test('missing_tensor_cleanup',missing)
 def short():
  with make() as x:
   x.extract('blk.0.attn_q.weight',d/'short',reader=lambda fd,n,off:os.pread(fd,min(n,7),off),writer=lambda fd,v:os.write(fd,v[:5]));assert (d/'short').read_bytes()==payload[32:320]
 test('short_reads_and_writes_complete',short)
 for name,kwargs in [('truncated_read',{'reader':lambda *a:b''}),('read_error',{'reader':lambda *a:(_ for _ in ()).throw(OSError('injected'))}),('write_failure',{'writer':lambda *a:0})]:
  def case(name=name,kwargs=kwargs):
   with make() as x:rejects(lambda:x.extract('blk.0.attn_q.weight',d/name,**kwargs))
   assert not (d/name).exists() and not list(d.glob('*.partial-*'))
  test(name,case)
 def existing():
  with make() as x:rejects(lambda:x.extract('blk.0.attn_q.weight',out))
  assert out.read_bytes()==payload[32:320]
 test('no_overwrite',existing)
 def collision():
  import types
  saved=m.uuid.uuid4;m.uuid.uuid4=lambda:types.SimpleNamespace(hex='collision')
  pre=d/'collision.partial-collision';pre.write_bytes(b'preserve')
  try:
   with make() as x:rejects(lambda:x.extract('blk.0.attn_q.weight',d/'collision'))
   assert pre.read_bytes()==b'preserve'
  finally:m.uuid.uuid4=saved;pre.unlink()
 test('preexisting_partial_preserved',collision)
 def truncated_file():
  old=m.identity(p.stat())
  with p.open('r+b') as f:f.truncate(100)
  rejects(lambda:m.Extractor(p,base(),old,start=32,expected_count=2));p.write_bytes(payload)
 test('actual_truncated_file_rejection',truncated_file)

 def changed():
  old=m.identity(p.stat());os.utime(p,ns=(p.stat().st_atime_ns,p.stat().st_mtime_ns+1));rejects(lambda:m.Extractor(p,base(),old,start=32,expected_count=2))
 test('identity_before_open',changed)
 def during():
  with make() as x:rejects(lambda:x.extract('blk.0.attn_q.weight',d/'changed',hook=lambda n:os.utime(p,ns=(p.stat().st_atime_ns,p.stat().st_mtime_ns+1))))
  assert not (d/'changed').exists() and not list(d.glob('*.partial-*'))
 test('identity_during_read',during)
 def replaced():
  with make() as x:
   p.rename(d/'old');p.write_bytes(payload);rejects(lambda:x.extract('blk.0.attn_q.weight',d/'replaced'))
 test('path_inode_replacement',replaced)
 def symlink():
  link=d/'link';link.symlink_to(p);rejects(lambda:m.Extractor(link,base(),m.identity(p.stat()),start=32,expected_count=2))
 test('source_symlink_refusal',symlink)
 def duplicate_json():
  q=d/'index';q.write_text('{"tensors":[],"tensors":[]}');rejects(lambda:m.read_json(q))
 test('duplicate_json_keys',duplicate_json)
 def metadata_budget():
  q=d/'large-index';q.write_bytes(b' '*(2*1024*1024+1));rejects(lambda:m.read_json(q))
 test('metadata_budget',metadata_budget)
 def bounded():
  q=d/'large-source';i=base();i['tensors']=i['tensors'][:1];i['tensors'][0].update(shape=[256,20000],bytes=2880000)
  with q.open('wb') as f:f.truncate(2880032)
  requests=[]
  def read(fd,n,off):requests.append(n);return os.pread(fd,n,off)
  with m.Extractor(q,i,m.identity(q.stat()),start=32,expected_count=1) as x:x.extract('blk.0.attn_q.weight',d/'large-output',reader=read)
  assert max(requests)<=m.CHUNK and len(requests)==3
 test('bounded_multi_chunk_read',bounded)
 # Every extraction context and failed constructor must have closed its descriptor.
 assert len(os.listdir('/proc/self/fd'))==before_fds
 test('descriptor_cleanup_no_leaks',lambda:None)
(R/'fault-results.json').write_text(json.dumps({'tests':results,'passed':len(results),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'synthetic_only':True},indent=2)+'\n')
print('PASS',len(results),'synthetic extraction tests')
