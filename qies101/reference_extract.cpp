// Independent stock GGUF parser and POSIX range reader. No QUORTEX parser or kernels.
#include <algorithm>
#include "gguf.h"
#include <sys/stat.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <stdexcept>
void require(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
struct FD{int n=-1;explicit FD(int f):n(f){require(n>=0,"open failed");}~FD(){if(n>=0)::close(n);}FD(const FD&)=delete;};
bool same(const struct stat&a,const struct stat&b){return a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&a.st_mtim.tv_sec==b.st_mtim.tv_sec&&a.st_mtim.tv_nsec==b.st_mtim.tv_nsec&&a.st_ctim.tv_sec==b.st_ctim.tv_sec&&a.st_ctim.tv_nsec==b.st_ctim.tv_nsec;}
int main(int argc,char**argv)try{
 require(argc>=3,"usage: reference-extract MODEL --describe | OUTDIR NAME...");
 FD fd(open(argv[1],O_RDONLY|O_CLOEXEC|O_NOFOLLOW));struct stat before{},after{};require(fstat(fd.n,&before)==0&&S_ISREG(before.st_mode),"regular model file required");
 auto path="/proc/self/fd/"+std::to_string(fd.n);
 std::unique_ptr<gguf_context,decltype(&gguf_free)>g(gguf_init_from_file(path.c_str(),{true,nullptr}),gguf_free);require(g!=nullptr,"stock GGUF parsing failed");
 auto start=gguf_get_data_offset(g.get());auto alignment=gguf_get_alignment(g.get());
 require(start<=size_t(before.st_size),"data offset outside file");
 std::vector<std::pair<size_t,size_t>>ranges;
 for(int64_t i=0;i<gguf_get_n_tensors(g.get());++i){auto offset=gguf_get_tensor_offset(g.get(),i),bytes=gguf_get_tensor_size(g.get(),i);require(offset%alignment==0&&offset<=size_t(before.st_size)-start&&bytes<=size_t(before.st_size)-start-offset,"stock tensor bounds");ranges.emplace_back(start+offset,start+offset+bytes);}
 std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();++i)require(ranges[i-1].second<=ranges[i].first,"stock overlap");
 if(std::string(argv[2])=="--describe"){
  for(int64_t i=0;i<gguf_get_n_tensors(g.get());++i){auto*ne=gguf_get_tensor_ne(g.get(),i);std::cout<<gguf_get_tensor_name(g.get(),i)<<'\t'<<start+gguf_get_tensor_offset(g.get(),i)<<'\t'<<gguf_get_tensor_size(g.get(),i)<<'\t'<<int(gguf_get_tensor_type(g.get(),i));for(int d=0;d<4;++d)std::cout<<'\t'<<ne[d];std::cout<<'\n';}
 }else{
  require(argc>3,"no selected tensors");std::vector<char>buffer(1024*1024);
  for(int a=3;a<argc;++a){std::string name=argv[a];require(name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")==std::string::npos,"unsafe tensor name");auto id=gguf_find_tensor(g.get(),name.c_str());require(id>=0,"stock tensor missing");size_t length=gguf_get_tensor_size(g.get(),id),offset=start+gguf_get_tensor_offset(g.get(),id),done=0;
   std::string dest=std::string(argv[2])+"/"+name+".bin",tmp=dest+".partial";bool created=false;
   try{FD out(open(tmp.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));created=true;
    while(done<length){size_t wanted=std::min(buffer.size(),length-done);ssize_t n=pread(fd.n,buffer.data(),wanted,offset+done);if(n<0&&errno==EINTR)continue;require(n>0,"stock short/truncated read");size_t sent=0;while(sent<size_t(n)){auto written=write(out.n,buffer.data()+sent,n-sent);if(written<0&&errno==EINTR)continue;require(written>0,"stock output failure");sent+=written;}done+=n;}
    require(fsync(out.n)==0,"stock sync failed");require(fstat(fd.n,&after)==0&&same(before,after),"stock file identity changed");require(link(tmp.c_str(),dest.c_str())==0,"stock output publication failed");require(unlink(tmp.c_str())==0,"stock partial cleanup");created=false;std::cout<<name<<'\t'<<done<<'\n';
   }catch(...){if(created)unlink(tmp.c_str());throw;}
  }
 }
 require(fstat(fd.n,&after)==0&&same(before,after),"stock identity changed");struct stat named{};require(lstat(argv[1],&named)==0&&same(before,named),"stock path identity changed");
 rusage usage{};getrusage(RUSAGE_SELF,&usage);std::cerr<<"peak_rss_kib="<<usage.ru_maxrss<<" tensor_payload_chunk_limit=1048576 no_mmap=true\n";return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}
