#include "admission.h"
#include "bundle.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
namespace r102 {
namespace {
std::atomic<int> files{0};
void check(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
struct FD{int n;FD(const std::string&p):n(::open(p.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW)){check(n>=0,"tensor open failed");++files;}~FD(){::close(n);--files;}};
std::array<uint64_t,5> identity(const struct stat&s){return {(uint64_t)s.st_dev,(uint64_t)s.st_ino,(uint64_t)s.st_size,(uint64_t)s.st_mtim.tv_sec*1000000000+s.st_mtim.tv_nsec,(uint64_t)s.st_ctim.tv_sec*1000000000+s.st_ctim.tv_nsec};}

}
int open_files(){return files.load();}
std::vector<r99::Spec> specs(const Bundle&b){std::vector<r99::Spec>out;for(auto&e:b.entries)out.push_back(e.spec);return out;}
Bundle read_bundle(const std::string&path){
 std::ifstream f(path);check((bool)f,"cannot open bundle");f.seekg(0,std::ios::end);check(f.tellg()>=0&&f.tellg()<=128*1024,"bundle file budget");f.seekg(0);std::string magic,mode,line;Bundle b;
 std::getline(f,magic);check(magic=="QIES102-BUNDLE-1","bundle version");std::getline(f,mode);check(mode=="REAL"||mode=="FIXTURE","bundle mode");b.real=mode=="REAL";b.config.fixture=!b.real;
 std::getline(f,line);std::istringstream c(line);c>>b.config.layer>>b.config.hidden>>b.config.ffn>>b.config.heads>>b.config.kv_heads>>b.config.head>>b.config.context>>b.config.window;std::string extra;check((bool)c&&!(c>>extra),"bundle config");
 b.config.validate();
 size_t total=0;
 while(std::getline(f,line)){
  check(line.size()<8192&&b.entries.size()<15,"bundle metadata bound");std::istringstream row(line);Entry e;int type;
  row>>std::quoted(e.spec.name)>>type>>e.spec.columns>>e.spec.rows>>e.spec.offset>>e.spec.bytes;for(auto&n:e.identity)row>>n;row>>e.sha256>>std::quoted(e.path);
  check((bool)row&&!(row>>extra)&&e.path.size()<4096&&e.sha256.size()==64&&e.sha256.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid bundle row");
  e.spec.type=(ggml_type)type;check(e.identity[2]==e.spec.bytes,"file length mismatch");check(e.spec.bytes<=768ULL*1024*1024-total,"bundle weight budget");total+=e.spec.bytes;b.entries.push_back(e);
 }
 r99::validate_specs(b.config,specs(b));
 if(!b.real)check(total<8ULL*1024*1024&&b.config.hidden==256&&b.config.ffn==512&&b.config.head==64,"fixture namespace budget");
 return b;
}
void Reader::read(const r99::Spec&s,size_t off,void*destination,size_t n){
 auto it=std::find_if(bundle.entries.begin(),bundle.entries.end(),[&](const Entry&e){return e.spec.name==s.name;});check(it!=bundle.entries.end(),"tensor not in bundle");auto&e=*it;
 check(e.spec.type==s.type&&e.spec.columns==s.columns&&e.spec.rows==s.rows&&e.spec.offset==s.offset&&e.spec.bytes==s.bytes,"tensor descriptor mismatch");
 check(n<=1024*1024&&off<=s.bytes&&n<=s.bytes-off,"bounded tensor read range");
 FD fd(e.path);struct stat before{},named{};check(fstat(fd.n,&before)==0&&S_ISREG(before.st_mode)&&identity(before)==e.identity,"tensor identity changed");
 for(size_t done=0;done<n;){ssize_t count=pread(fd.n,(char*)destination+done,n-done,off+done);if(count<0&&errno==EINTR)continue;check(count>0,"short/truncated tensor read");done+=count;}
 check(fstat(fd.n,&before)==0&&identity(before)==e.identity&&lstat(e.path.c_str(),&named)==0&&identity(named)==e.identity,"tensor changed during read");
 bytes_read+=n;max_read=std::max(max_read,n);
}
LayerSource::LayerSource(const Bundle&b):reader(b),real(b.real){if(real)q104::real("LayerSource constructor");else for(auto&e:b.entries)q104::fixture_file(e.path);}
void LayerSource::read(const r99::Spec&s,size_t o,void*d,size_t n){reader.read(s,o,d,n);}
r99::Result execute_layer(const Bundle&b,const std::vector<float>&x,const std::vector<float>&x0,const r99::KV&kv,bool capture){LayerSource source(b);return r99::execute(b.config,specs(b),source,x,x0,kv,capture);}
}
