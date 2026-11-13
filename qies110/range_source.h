#pragma once
// Read-only data infrastructure. No GGML compute/backend APIs are used here.
#include "ggml.h"
#include <array>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <openssl/evp.h>
namespace io110 {
constexpr size_t CHUNK=1024*1024;
inline int open_files=0;
inline void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
inline std::array<uint64_t,5> id(const struct stat&s){return{{uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),uint64_t(s.st_mtim.tv_sec)*1000000000+s.st_mtim.tv_nsec,uint64_t(s.st_ctim.tv_sec)*1000000000+s.st_ctim.tv_nsec}};}
struct Entry{std::string name,hash;int type=0;uint64_t columns=0,rows=0,offset=0,bytes=0;};
inline size_t row_size(int type,uint64_t columns){uint64_t block=0,unit=0;switch(type){case 0:block=1;unit=4;break;case 8:block=32;unit=34;break;case 12:block=256;unit=144;break;case 13:block=256;unit=176;break;case 14:block=256;unit=210;break;default:throw std::runtime_error("unsupported layer tensor type");}require(columns>0&&columns<=40960&&columns%block==0,"row dimensions/alignment");return columns/block*unit;}
inline std::map<std::string,std::array<uint64_t,3>> schema(int layer){return{{"attn_q.weight",{12,5120,10240}},{"attn_k.weight",{8,5120,1024}},{"attn_v.weight",{8,5120,1024}},{"attn_gate.weight",{12,5120,10240}},{"attn_output.weight",{14,10240,5120}},{"attn_norm.weight",{0,5120,1}},{"post_attention_norm.weight",{0,5120,1}},{"attn_q_norm.weight",{0,128,1}},{"attn_k_norm.weight",{0,128,1}},{"attn_vres_lambda.weight",{0,1,1}},{"ffn_norm.weight",{0,5120,1}},{"post_ffw_norm.weight",{0,5120,1}},{"ffn_gate.weight",{uint64_t(layer<15||layer>=65?13:12),5120,40960}},{"ffn_up.weight",{uint64_t(layer<15||layer>=65?13:12),5120,40960}},{"ffn_down.weight",{uint64_t(layer<15||layer>=65?13:12),40960,5120}}};}
class File {
 int fd=-1;std::string path,cache_name;std::array<uint64_t,5>expected{};std::vector<uint8_t>cache;uint64_t cached_first=0,cached_rows=0;bool real_=false;
 void exact(uint64_t off,void*data,size_t n){size_t done=0;while(done<n){ssize_t got=pread(fd,(char*)data+done,n-done,off+done);if(got<0&&errno==EINTR)continue;require(got>0,"short/truncated tensor read");done+=size_t(got);}bytes_read+=n;max_request=std::max(max_request,n);}
public:
 int layer=0;std::map<std::string,Entry>entries;uint64_t bytes_read=0;size_t max_request=0,cache_peak=0;
 File(const std::string&plan,bool expected_real){try{
  std::ifstream f(plan);require(bool(f),"source manifest open");f.seekg(0,std::ios::end);require(f.tellg()>0&&f.tellg()<32768,"source manifest budget");f.seekg(0);std::string magic,mode,line;std::getline(f,magic);std::getline(f,mode);require(magic=="QIES110-SOURCE-1"&&(mode=="REAL"||mode=="SYNTHETIC"),"source manifest version/mode");real_=mode=="REAL";require(real_==expected_real,"source mode mismatch");f>>layer>>std::quoted(path);for(auto&v:expected)f>>v;require(bool(f)&&layer>=0&&layer<80&&path.size()<4096,"manifest identity");size_t count=0;f>>count;require(count==15,"complete layer requires 15 tensors");std::getline(f,line);auto required=schema(layer);std::vector<std::pair<uint64_t,uint64_t>>ranges;
  for(size_t i=0;i<count;++i){require(bool(std::getline(f,line))&&line.size()<2048,"tensor manifest row");std::istringstream s(line);Entry e;std::string extra;s>>std::quoted(e.name)>>e.type>>e.columns>>e.rows>>e.offset>>e.bytes>>e.hash;require(bool(s)&&!(s>>extra),"malformed tensor descriptor");auto it=required.find(e.name);require(it!=required.end()&&entries.count(e.name)==0,"missing/duplicate/unknown tensor");auto spec=it->second;require(e.type==int(spec[0])&&e.columns==spec[1]&&e.rows==spec[2],"tensor type/shape mismatch");size_t rb=row_size(e.type,e.columns);require(e.rows<=40960&&e.bytes==rb*e.rows,"tensor byte count");require(e.offset>=(real_?3615712ULL:64ULL)&&e.offset%32==0&&e.offset<=expected[2]&&e.bytes<=expected[2]-e.offset,"tensor range/offset overflow");require(e.hash.size()==64&&e.hash.find_first_not_of("0123456789abcdef")==std::string::npos,"tensor digest syntax");ranges.emplace_back(e.offset,e.offset+e.bytes);entries.emplace(e.name,e);
  }
  require(!(f>>line),"trailing manifest data");std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();++i)require(ranges[i-1].second<=ranges[i].first,"overlapping tensor descriptors");
  fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"source file open");++open_files;check();char head[64]{};exact(0,head,sizeof(head));if(real_){require(expected[2]==40426079712ULL&&!memcmp(head,"GGUF",4),"real GGUF identity");}else{require(expected[2]<600ULL*1024*1024&&!memcmp(head,"QIES110_SYNTHETIC_TENSOR_DATA_V1",30),"synthetic-only file identity");}
 }catch(...){close_file();throw;}}
 ~File(){close_file();}File(const File&)=delete;File&operator=(const File&)=delete;
 void close_file(){if(fd>=0){close(fd);fd=-1;--open_files;}}
 bool real()const{return real_;}
 void check()const{struct stat a{},b{};require(fstat(fd,&a)==0&&S_ISREG(a.st_mode)&&lstat(path.c_str(),&b)==0&&id(a)==expected&&id(b)==expected,"source identity changed");}
 const Entry&get(const std::string&name)const{auto it=entries.find(name);require(it!=entries.end(),"tensor missing");return it->second;}
 void read(const std::string&name,uint64_t relative,void*dst,size_t n){auto&e=get(name);require(n<=CHUNK&&relative<=e.bytes&&n<=e.bytes-relative,"bounded read range");check();exact(e.offset+relative,dst,n);check();}
 void row(const std::string&name,int r,int columns,void*dst){auto&e=get(name);require(r>=0&&uint64_t(r)<e.rows&&columns==int(e.columns),"requested row mismatch");size_t rb=row_size(e.type,e.columns);if(cache_name!=name||uint64_t(r)<cached_first||uint64_t(r)>=cached_first+cached_rows){cached_first=r;cached_rows=std::min<uint64_t>(e.rows-r,CHUNK/rb);cache_name.clear();cache.resize(cached_rows*rb);read(name,cached_first*rb,cache.data(),cache.size());cache_name=name;cache_peak=std::max(cache_peak,cache.size());}else check();memcpy(dst,cache.data()+(r-cached_first)*rb,rb);}
 std::vector<float>floats(const std::string&name){auto&e=get(name);require(e.type==0&&e.rows==1&&e.bytes<=20480,"bounded F32 parameter");std::vector<float>v(e.columns);read(name,0,v.data(),e.bytes);return v;}
};
struct RowSource {File&file;std::string name;ggml_type type;RowSource(File&f,const std::string&n):file(f),name(n),type(ggml_type(f.get(n).type)){}bool real()const{return file.real();}void row(int r,int columns,void*dst)const{file.row(name,r,columns,dst);}};
inline std::string hash_tensor(File&file,const std::string&name){EVP_MD_CTX*c=EVP_MD_CTX_new();require(c,"digest allocation");try{require(EVP_DigestInit_ex(c,EVP_sha256(),nullptr)==1,"digest init");std::vector<uint8_t>b(CHUNK);auto&e=file.get(name);for(uint64_t off=0;off<e.bytes;){size_t n=std::min<uint64_t>(b.size(),e.bytes-off);file.read(name,off,b.data(),n);require(EVP_DigestUpdate(c,b.data(),n)==1,"digest update");off+=n;}unsigned char out[32];unsigned n=0;require(EVP_DigestFinal_ex(c,out,&n)==1&&n==32,"digest final");EVP_MD_CTX_free(c);c=nullptr;std::ostringstream s;for(auto v:out)s<<std::hex<<std::setfill('0')<<std::setw(2)<<int(v);return s.str();}catch(...){if(c)EVP_MD_CTX_free(c);throw;}}
}
