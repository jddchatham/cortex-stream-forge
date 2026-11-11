"""Ephemeral historical-test view; never exports executable files."""
from pathlib import Path
import shutil,os,subprocess,sys,tarfile,json,hashlib,gzip

# SDK bytes are available as ELF only in this private, no-real-model namespace.
sdk=Path('/stock-sdk');sdk.mkdir(exist_ok=True)
manifest=json.loads(Path('/active/stock-sdk-manifest.json').read_text())
archive=Path('/active/stock-sdk.tar.gz');assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
with tarfile.open(archive) as t:t.extractall(sdk,filter='data')
for member in sdk.iterdir():
 dest=Path('/baseline-deps/stock-build/bin')/member.name
 if member.is_symlink():dest.symlink_to(os.readlink(member))
 else:shutil.copyfile(member,dest)
for name,expected in manifest['files'].items():assert hashlib.sha256((sdk/name).read_bytes()).hexdigest()==expected
engine=Path('/active/reference-engine.gz')
if engine.exists() and '/active/build_engine.py' not in sys.argv:
 meta=json.loads(Path('/active/engine.json').read_text());assert all(hashlib.sha256((Path('/active')/name).read_bytes()).hexdigest()==value for name,value in meta['source_hashes'].items());assert hashlib.sha256(Path('/active/stock-sdk-manifest.json').read_bytes()).hexdigest()==meta['stock_sdk_manifest_sha256'];assert hashlib.sha256(engine.read_bytes()).hexdigest()==meta['archive_sha256']
 Path('/stock-engine').mkdir(exist_ok=True)
 with gzip.open(engine,'rb') as f,Path('/stock-engine/reference-engine').open('wb') as g:shutil.copyfileobj(f,g,1024*1024)
 assert hashlib.sha256(Path('/stock-engine/reference-engine').read_bytes()).hexdigest()==meta['binary_sha256']
 Path('/stock-engine/reference-engine').chmod(0o700)

W=Path('/baseline-regression-root.txt').read_text().strip();W=Path(W);src=Path('/baseline-regression')
def ignore(folder,names):
 rel=str(Path(folder).relative_to(src))
 if rel=='qies100':return {'fixture','fixture-quantized'}
 if rel=='qies103':return {'oneblock-fixture','oneblock-fixture-quantized'}
 return set()
shutil.copytree(src,W,dirs_exist_ok=True,symlinks=True,ignore=ignore)
for folder in ('qies099','qies100','qies101','qies102','qies103'):
 for p in (W/folder).rglob('*'):
  if p.is_file() and not p.is_symlink() and '.so' not in p.name:
   with p.open('rb') as f:magic=f.read(4)
   if magic==b'\x7fELF':p.chmod(p.stat().st_mode|0o100)
result=subprocess.run(sys.argv[1:],cwd=W,stdin=subprocess.DEVNULL)
# Export evidence only, never old rebuilt workers. Namespace teardown removes those workers.
label=os.environ.get('QIES104_EXPORT','latest')
if not label.replace('-','').replace('_','').isalnum():raise SystemExit('invalid export label')
out=Path('/q104-results')/label;out.mkdir(exist_ok=True)
for folder in ('q104-out','qies100','qies101','qies103'):
 base=W/folder
 if not base.exists():continue
 candidates=list(base.rglob('*')) if folder=='q104-out' else list(base.glob('*.json'))+list(base.glob('*tests.log'))
 for p in candidates:
  if not p.is_file() or p.is_symlink() or p.stat().st_size>4*1024*1024:continue
  if p.suffix not in ('.json','.log','.tsv','.bin','.f32'):continue
  with p.open('rb') as f:magic=f.read(4)
  if magic==b'\x7fELF':continue
  dest=out/folder/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
raise SystemExit(result.returncode)
