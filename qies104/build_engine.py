"""Runs only in sealed namespace; exports compressed engine, never host-executable ELF."""
from pathlib import Path
import subprocess,json,gzip,hashlib,shutil
assert Path('/qies104.synthetic').is_file()
W=Path('/baseline-regression-root.txt').read_text().strip();W=Path(W);S=W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259'
cmd=['c++','-std=c++17','-O2','-Wall','-Wextra','-I/active','-I'+str(S/'include'),'-I'+str(S/'ggml/include'),'/active/admission.cpp','/active/reference.cpp','-Wl,-rpath,/stock-sdk',*['/stock-sdk/'+n for n in ('libllama.so','libggml.so','libggml-cpu.so','libggml-base.so')],'-lcrypto','-o','/tmp/reference-engine']
subprocess.run(cmd,check=True)
out=Path('/q104-results/engine-build');out.mkdir(exist_ok=True)
with Path('/tmp/reference-engine').open('rb') as f,gzip.open(out/'reference-engine.gz','wb') as g:shutil.copyfileobj(f,g,1024*1024)
(out/'engine.json').write_text(json.dumps({'command':cmd,'binary_sha256':hashlib.sha256(Path('/tmp/reference-engine').read_bytes()).hexdigest(),'archive_sha256':hashlib.sha256((out/'reference-engine.gz').read_bytes()).hexdigest(),'source_hashes':{name:hashlib.sha256((Path('/active')/name).read_bytes()).hexdigest() for name in ('reference.cpp','admission.cpp','admission.h','fixture-catalog.inc')},'stock_sdk_manifest_sha256':hashlib.sha256(Path('/active/stock-sdk-manifest.json').read_bytes()).hexdigest(),'stock_source':'240d68fcfe0c2a1f298c8e18ca181368b1914259','real_enabled':False},indent=2)+'\n')
print('Built independent stock observer inside sealed namespace; only compressed artifact exported')
