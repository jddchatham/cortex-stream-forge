"""Sealed synthetic filesystem. Real model/data directories are never exposed."""
from pathlib import Path
import os,json,subprocess,shutil,hashlib
R=Path(__file__).resolve().parent;W=R.parent;M=R/'regression/root'
def initialize():
 # Only the newly created regression snapshot is edited here.
 for relative in ('qies101/artifacts','qies102/bundle'):
  p=M/relative
  if p.exists():shutil.rmtree(p)
  p.mkdir(parents=True)
 links={'runtime-flashnext-layer-exp':'/baseline-deps/runtime','qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259':'/baseline-deps/stocksrc','qies100/stock-build':'/baseline-deps/stock-build'}
 for rel,dest in links.items():
  p=M/rel
  if p.is_symlink():p.unlink()
  p.symlink_to(dest,target_is_directory=True)
 shutil.copy2(W/'.git',M/'.git')
 deps=R/'deps/cpu';deps.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((W/'qies099/checks/build-identity.json').read_text())['library_sha256'];source=W.parent/'cortex-stream-forge-qies074/build-quortex/bin'
 for name in manifest:
  shutil.copy2(source/name,deps/name)
  for alias in (name.split('.so')[0]+'.so',name.split('.so')[0]+'.so.0'):
   p=deps/alias
   if not p.exists():p.symlink_to(name)


_CACHE={}
CATALOG_SHA='e64d103b97112fc12c0ad13f5d8efc2fdba9bd63abff2ac13da9990f5c54761f'
def validate_fixtures():
 raw=(R/'fixture-catalog.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=CATALOG_SHA:raise ValueError('sealed fixture catalog changed')
 for name,expected in json.loads(raw).items():
  p=Path(name);st=p.stat();ident=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
  if _CACHE.get(name)==ident:continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   while b:=f.read(1024*1024):h.update(b)
  if h.hexdigest()!=expected:raise ValueError('synthetic fixture changed: '+name)
  after=p.stat()
  if ident!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('fixture identity changed during verification')
  _CACHE[name]=ident

def command(args):
 validate_fixtures()
 envdir=W.parent;git=envdir/'cortex-stream-forge-qies074/.git';cpulibs=envdir/'cortex-stream-forge-qies074/build-quortex/bin'
 cmd=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--ro-bind','/usr','/usr','--ro-bind','/etc','/etc','--symlink','usr/bin','/bin','--symlink','usr/bin','/sbin','--symlink','usr/lib','/lib','--symlink','usr/lib','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--tmpfs','/stock-sdk','--tmpfs','/stock-engine','--dir','/home','--dir','/home/user','--dir','/home/user/PandaCode Bots','--dir',str(envdir),'--ro-bind',str(M),'/baseline-regression','--tmpfs',str(W),'--ro-bind',str(R/'workspace-path.txt'),'/baseline-regression-root.txt','--bind',str(R/'results'),'/q104-results','--ro-bind',str(R),'/active','--ro-bind',str(R/'synthetic.marker'),'/qies104.synthetic','--ro-bind',str(W/'runtime-flashnext-layer-exp'),'/baseline-deps/runtime','--ro-bind',str(W/'qies100/llama.cpp-240d68fcfe0c2a1f298c8e18ca181368b1914259'),'/baseline-deps/stocksrc','--ro-bind',str(W/'qies100/stock-build'),'/baseline-deps/stock-build','--tmpfs','/baseline-deps/stock-build/bin','--ro-bind',str(R/'deps/cpu'),str(cpulibs),'--ro-bind',str(git),str(git),'--ro-bind',str(W.parent/'qies077-guard.py'),str(W.parent/'qies077-guard.py'),'--ro-bind',str(W/'qies100/fixture'),str(W/'qies100/fixture'),'--ro-bind',str(W/'qies100/fixture-quantized'),str(W/'qies100/fixture-quantized'),'--ro-bind',str(W/'qies103/oneblock-fixture'),str(W/'qies103/oneblock-fixture'),'--ro-bind',str(W/'qies103/oneblock-fixture-quantized'),str(W/'qies103/oneblock-fixture-quantized'),'--chdir',str(W),'--setenv','HOME',str(Path.home()),'--setenv','PATH','/usr/bin:/bin','--setenv','LANG','C.UTF-8','--setenv','QIES104_SYNTHETIC','1','--setenv','QIES104_EXPORT',os.environ.get('QIES104_EXPORT','latest'),'--','/usr/bin/python3','-I','-B','/active/bootstrap.py',*map(str,args)]
 return cmd
if __name__=='__main__':
 import sys
 initialize();raise SystemExit(subprocess.run(command(sys.argv[1:])).returncode)
