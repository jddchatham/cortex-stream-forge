#pragma once
// Input/state/ownership infrastructure only; no shared attention computation.
#include "support.h"
#include <array>
#include <fstream>
#include <limits>
using namespace infra107;
namespace a109 {
constexpr int H=5120,Q=10240,K=1024,D=128,NH=80,NKV=8;
struct Config{std::string name;int layer,slots,start;bool sliding()const{return layer%5!=4;}};
inline Config config(const std::string&s){
 if(s=="swa-start")return{s,0,256,0};
 if(s=="global-start")return{s,4,256,0};
 if(s=="swa-boundary")return{s,0,1280,1023};
 if(s=="global-boundary")return{s,4,1280,1023};
 throw std::runtime_error("only fixed synthetic cases accepted");
}
// Attention fixtures include Q8_0 (32-value blocks): a 256-value template
// holds eight of those blocks. The full packed row size, not type_size, is used.
struct InputBlocks {
 ggml_type type;int projection;size_t block_bytes;std::vector<uint8_t>data;
 InputBlocks(ggml_type t,int p):type(t),projection(p),block_bytes(ggml_row_size(t,256)),data(fixture107::TEMPLATES*block_bytes){
  need(t==GGML_TYPE_Q4_K||t==GGML_TYPE_Q5_K||t==GGML_TYPE_Q8_0||t==GGML_TYPE_Q6_K,"attention fixture type");std::vector<float>x(256),importance(256,1.f);
  for(int code=0;code<fixture107::TEMPLATES;++code){for(int i=0;i<256;++i)x[i]=fixture107::block_value(projection,code,i);need(ggml_quantize_chunk(type,x.data(),data.data()+code*block_bytes,0,1,256,importance.data())==block_bytes,"attention fixture byte count");}++live;
 }
 ~InputBlocks(){--live;}
 void row(int r,int columns,void*dst)const{auto*p=static_cast<uint8_t*>(dst);for(int b=0;b<columns/256;++b)memcpy(p+b*block_bytes,data.data()+fixture107::code(projection,r,b)*block_bytes,block_bytes);}
};
struct State{int position=0;std::vector<ggml_fp16_t>key,value;};
inline std::array<uint64_t,8>header(const Config&c,const State&s){return{{0x32564b3930315155ULL,1,(uint64_t)c.layer,(uint64_t)c.slots,(uint64_t)s.position,s.key.size(),s.value.size(),109}};}
inline std::string identity(const Config&c,const State&s){Digest h;auto a=header(c,s);h.add(a.data(),sizeof(a));h.add(s.key.data(),s.key.size()*2);h.add(s.value.data(),s.value.size()*2);return h.finish();}
inline State initial(const Config&c){State s;s.position=c.start;s.key.resize(size_t(K)*c.slots);s.value.resize(s.key.size());for(int p=0;p<c.start;++p)for(int i=0;i<K;++i){s.key[size_t(p)*K+i]=ggml_fp32_to_fp16(float((p*7+i*3)%33-16)/64.f);s.value[size_t(p)*K+i]=ggml_fp32_to_fp16(float((p*11+i*5)%65-32)/64.f);}return s;}
inline State load(const Config&c){
 std::ifstream f("/state/previous.kv",std::ios::binary|std::ios::ate);need(bool(f),"previous synthetic checkpoint missing");size_t n=size_t(K)*c.slots;need(f.tellg()==std::streamoff(128+n*4),"state length");f.seekg(0);std::array<uint64_t,8>a{};std::string h(64,' ');f.read((char*)a.data(),64);f.read(h.data(),64);
 need(a[0]==0x32564b3930315155ULL&&a[1]==1&&a[2]==uint64_t(c.layer)&&a[3]==uint64_t(c.slots)&&a[4]<uint64_t(c.slots)&&a[5]==n&&a[6]==n&&a[7]==109,"state identity/position/shape mismatch");
 State s;s.position=int(a[4]);s.key.resize(n);s.value.resize(n);f.read((char*)s.key.data(),n*2);f.read((char*)s.value.data(),n*2);need(bool(f)&&h==identity(c,s),"state checksum mismatch");return s;
}
inline void raw(Captures&out,const std::string&name,const void*data,size_t bytes){
 out.files.push_back(name);int fd=open((out.staging+"/"+name).c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);need(fd>=0,"exclusive raw capture");++live;
 try{const char*p=(const char*)data;while(bytes){auto n=write(fd,p,bytes);if(n<0&&errno==EINTR)continue;need(n>0,"raw capture write");p+=n;bytes-=n;}need(fsync(fd)==0,"raw capture sync");}catch(...){close(fd);--live;throw;}
 int status=close(fd);--live;need(status==0,"raw capture close");
}
inline void checkpoint(Captures&out,const Config&c,const State&s){auto a=header(c,s);std::string hash=identity(c,s);std::vector<uint8_t>b(128+s.key.size()*4);memcpy(b.data(),a.data(),64);memcpy(b.data()+64,hash.data(),64);memcpy(b.data()+128,s.key.data(),s.key.size()*2);memcpy(b.data()+128+s.key.size()*2,s.value.data(),s.value.size()*2);raw(out,"state.kv",b.data(),b.size());}
inline std::vector<float>activation(int position,bool original){std::vector<float>v(H);for(int i=0;i<H;++i)v[i]=original?float((i*29+position*3+31)%509-254)/256.f:float((i*13+position*7+7)%257-128)/128.f;return v;}
inline std::vector<float>gamma(int n,int kind){std::vector<float>v(n);for(int i=0;i<n;++i)v[i]=fixture107::gamma(kind,i);return v;}
inline std::vector<float>mask(const Config&c,int pos){std::vector<float>v(c.slots,-std::numeric_limits<float>::infinity());for(int i=0;i<=pos;++i)if(!c.sliding()||pos-i<1024)v[i]=0;return v;}
inline void get(Captures&out,const std::string&name,ggml_tensor*t){std::vector<float>v(ggml_nelements(t));need(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"capture shape/type");ggml_backend_tensor_get(t,v.data(),0,v.size()*4);out.write(name,v);}
inline void set(ggml_tensor*t,const std::vector<float>&v){need(ggml_nbytes(t)==v.size()*4,"activation shape");ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
inline std::vector<float>read(ggml_tensor*t){std::vector<float>v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,v.size()*4);return v;}
inline void state_record(const Config&c,const State&s,const char*phase){std::cout<<"{\"state_phase\":\""<<phase<<"\",\"position\":"<<s.position<<",\"kv_bytes\":"<<s.key.size()*4<<",\"sha256\":\""<<identity(c,s)<<"\"}\n"<<std::flush;}
inline std::unique_ptr<State> authoritative;inline std::string before_state;inline std::unique_ptr<Config> active_config;
inline int failed(const std::exception&e){bool unchanged=!authoritative||identity(*active_config,*authoritative)==before_state;std::cout<<"{\"authoritative_state_unchanged\":"<<(unchanged?"true":"false")<<"}\n";mark("after-failure-cleanup");summary();std::cerr<<e.what()<<"\n";return live==0&&unchanged?1:2;}
inline bool known_fault(const std::string&s){if(s=="none")return true;const char*names[]={
#include "faults.inc"
};for(auto*n:names)if(s==n)return true;return false;}
inline std::vector<float> supplied(const std::string&name){std::ifstream f("/state/"+name,std::ios::binary|std::ios::ate);need(bool(f)&&f.tellg()==H*4,"activation input length");f.seekg(0);std::vector<float>v(H);f.read((char*)v.data(),H*4);need(bool(f),"activation input read");for(float x:v)need(std::isfinite(x),"activation input finite");return v;}
inline std::pair<Config,State>admit(int argc,char**argv){if(argc<2||std::string(argv[1])!="--synthetic")q104::real("QIES111 real model interface locked");q104::synthetic();need(argc==8,"--synthetic LAYER SLOTS POSITION seed|resume FAULT");
 auto integer=[](const char*s){std::string t=s;need(!t.empty()&&t.size()<=4&&t.find_first_not_of("0123456789")==std::string::npos,"integer contract");return std::stoi(t);};int layer=integer(argv[2]),slots=integer(argv[3]),pos=integer(argv[4]);need(layer<80&&(slots==256||slots==1280)&&pos<slots,"layer/capacity/position contract");Config c{"layer-"+std::to_string(layer),layer,slots,pos};need(known_fault(argv[7]),"known synthetic fault");need(std::string(argv[5])=="seed"||std::string(argv[5])=="resume","state mode");need(std::string(argv[6])=="contract-v1","input contract version");State s=std::string(argv[5])=="seed"?initial(c):load(c);need(s.position==pos,"checkpoint position mismatch");active_config=std::make_unique<Config>(c);authoritative=std::make_unique<State>(std::move(s));before_state=identity(c,*authoritative);state_record(c,*authoritative,"before");return{c,State{}};}
}
