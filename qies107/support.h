#pragma once
// Ownership/telemetry/input/output infrastructure only. No FFN numerical graph
// or streaming scheduler is shared with the independent reference.
#include "admission.h"
#include "fixture.h"
#include "telemetry.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <vector>
#include <string>
#include <atomic>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <openssl/evp.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
namespace infra107 {
inline std::atomic<int> live{0};
inline size_t current_weights=0,peak_weights=0,current_buffers=0,peak_buffers=0;
inline void need(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
inline std::string requested_fault;
inline bool fault_seen=false;
inline void fault(const std::string&s){if(s==requested_fault){fault_seen=true;throw std::runtime_error("injected "+s);}}
inline void mark(const std::string&phase,size_t workspace=0,size_t meta=0,size_t used=0,size_t host=0,size_t source=0){diag106::sample(phase.c_str(),current_weights,current_buffers,workspace,meta,used,host,source,live);}
struct Backend {
 ggml_backend_t p=nullptr;Backend(){p=ggml_backend_cpu_init();need(p,"CPU backend");ggml_backend_cpu_set_n_threads(p,1);++live;}
 ~Backend(){if(p){ggml_backend_free(p);--live;}}Backend(const Backend&)=delete;
};
struct Context {
 ggml_context*p=nullptr;static constexpr size_t bytes=1024*1024;
 Context(){p=ggml_init({bytes,nullptr,true});need(p,"GGML metadata");++live;}
 ~Context(){if(p){ggml_free(p);--live;}}Context(const Context&)=delete;
};
struct Buffer {
 ggml_backend_buffer_t p=nullptr;size_t bytes=0,weights=0;
 Buffer(ggml_context*c,ggml_backend_t backend,size_t wb):weights(wb){p=ggml_backend_alloc_ctx_tensors(c,backend);need(p,"CPU tensor buffer");bytes=ggml_backend_buffer_get_size(p);++live;current_buffers+=bytes;current_weights+=weights;peak_buffers=std::max(peak_buffers,current_buffers);peak_weights=std::max(peak_weights,current_weights);}
 ~Buffer(){if(p){ggml_backend_buffer_free(p);current_buffers-=bytes;current_weights-=weights;--live;}}Buffer(const Buffer&)=delete;
};
struct Workspace {
 static constexpr size_t limit=1024*1024;std::vector<uint8_t>bytes;
 Workspace():bytes(limit){++live;}~Workspace(){--live;}Workspace(const Workspace&)=delete;
 ggml_cplan plan(ggml_cgraph*g){auto p=ggml_graph_plan(g,1,nullptr);need(p.work_size<=bytes.size(),"CPU workspace budget");p.work_data=bytes.data();return p;}
};
struct Digest {
 EVP_MD_CTX*p=nullptr;
 Digest(){p=EVP_MD_CTX_new();need(p,"digest allocation");if(EVP_DigestInit_ex(p,EVP_sha256(),nullptr)!=1){EVP_MD_CTX_free(p);p=nullptr;throw std::runtime_error("digest init");}++live;}
 ~Digest(){if(p){EVP_MD_CTX_free(p);--live;}}
 void add(const void*b,size_t n){need(EVP_DigestUpdate(p,b,n)==1,"digest update");}
 std::string finish(){unsigned char b[32];unsigned n=0;need(EVP_DigestFinal_ex(p,b,&n)==1&&n==32,"digest final");std::ostringstream s;for(auto v:b)s<<std::hex<<std::setw(2)<<std::setfill('0')<<int(v);return s.str();}
};
// Fixture-only acceleration: templates define every quantized matrix block.
// This does NOT assume real GGUF weights repeat and is not a model optimization.
struct Blocks {
 ggml_type type;int projection;size_t block_bytes;std::vector<uint8_t>data;
 Blocks(ggml_type t,int p):type(t),projection(p),block_bytes(ggml_type_size(t)),data(fixture107::TEMPLATES*block_bytes){
  std::vector<float>x(256),importance(256,1.f);
  for(int code=0;code<fixture107::TEMPLATES;++code){for(int i=0;i<256;++i)x[i]=fixture107::block_value(projection,code,i);need(ggml_quantize_chunk(type,x.data(),data.data()+code*block_bytes,0,1,256,importance.data())==block_bytes,"fixture block quantization");}
  ++live;
 }
 ~Blocks(){--live;}
 void row(int row_index,int columns,void*dst)const{auto*out=static_cast<uint8_t*>(dst);for(int b=0;b<columns/256;++b)std::memcpy(out+b*block_bytes,data.data()+fixture107::code(projection,row_index,b)*block_bytes,block_bytes);}
};
struct Captures {
 std::string staging,final;std::vector<std::string>files;bool published=false;
 explicit Captures(int sequence):staging("/outputs/staging-"+std::to_string(sequence)),final("/outputs/captures-"+std::to_string(sequence)){need(std::filesystem::create_directory(staging),"exclusive staging directory");++live;}
 ~Captures(){if(!published){for(const auto&name:files)unlink((staging+"/"+name).c_str());rmdir(staging.c_str());}--live;}
 void write(const std::string&name,const std::vector<float>&v){
  fault("capture."+name+".before");for(float f:v)need(std::isfinite(f),"nonfinite capture");
  std::string filename=name+".f32";files.push_back(filename);int fd=open((staging+"/"+filename).c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);need(fd>=0,"exclusive capture create");++live;
  try{const auto*p=reinterpret_cast<const char*>(v.data());size_t left=v.size()*4;while(left){auto n=::write(fd,p,left);if(n<0&&errno==EINTR)continue;need(n>0,"capture write");p+=n;left-=n;}need(fsync(fd)==0,"capture sync");}
  catch(...){close(fd);--live;throw;}
  int status=close(fd);--live;need(status==0,"capture close");fault("capture."+name+".after");
 }
 void publish(){fault("before-publish");need(syscall(SYS_renameat2,AT_FDCWD,staging.c_str(),AT_FDCWD,final.c_str(),1)==0,"exclusive capture publication");published=true;}
};
inline void summary(){std::cout<<"{\"peak_weight_bytes\":"<<peak_weights<<",\"peak_cpu_tensor_buffer_bytes\":"<<peak_buffers<<",\"owned_resources\":"<<live<<",\"fault_seen\":"<<(fault_seen?"true":"false")<<",\"real_weights\":false}\n"<<std::flush;}
inline ggml_type type_from(const std::string&s){need(s=="q5"||s=="q4","only verified fixed FFN quantization profiles accepted");return s=="q5"?GGML_TYPE_Q5_K:GGML_TYPE_Q4_K;}
inline int vector_from(const std::string&s){need(s=="0"||s=="1"||s=="2","only built-in activation vectors accepted; filenames forbidden");return s[0]-'0';}
inline void weight_identity(const char*name,Digest&d,size_t n){std::cout<<"{\"weight\":\""<<name<<"\",\"sha256\":\""<<d.finish()<<"\",\"bytes\":"<<n<<"}\n"<<std::flush;}
}
