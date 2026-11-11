#include "admission.h"
#include "bundle.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sys/resource.h>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <sstream>
#include <algorithm>
namespace {
struct Memory{long rss=0,pss=0;};
Memory memory(){std::ifstream f("/proc/self/smaps_rollup");std::string line,key,unit;Memory m;long value;while(std::getline(f,line)){std::istringstream s(line);s>>key>>value>>unit;if(key=="Rss:")m.rss=value;if(key=="Pss:")m.pss=value;}return m;}
void synthetic(const std::string&prefix,const std::string&out){
 std::filesystem::create_directory(out);std::vector<r99::KV>state(5);
 for(int p=0;p<4;++p){std::vector<float>x(256);for(int i=0;i<256;++i)x[i]=float((i+p)%11-5)*.1f;auto x0=x;
  for(int l=0;l<5;++l){auto b=r102::read_bundle(prefix+std::to_string(l)+".plan");if(b.real)throw std::runtime_error("synthetic mode refuses real bundle");auto result=r102::execute_layer(b,x,x0,state[l]);state[l]=result.next;x=result.output;
   for(auto&[name,v]:result.captures){std::ofstream f(out+"/p"+std::to_string(p)+"-"+name+"-"+std::to_string(l)+".bin",std::ios::binary);f.write((char*)v.data(),v.size()*4);if(!f)throw std::runtime_error("capture failure");}
   if(r99::live_resources()!=0||r102::open_files()!=0)throw std::runtime_error("resource leak");
   auto current=memory();std::cout<<"synthetic_memory\t"<<p<<'\t'<<l<<'\t'<<result.weight_bytes<<'\t'<<result.graph_bytes<<'\t'<<result.kv_bytes<<'\t'<<current.rss<<'\t'<<current.pss<<"\tresources=0\n";
  }
 }
 std::cout<<"PASS synthetic layer-source integration, four positions, five layers\n";
}
void storage(const std::string&plan){
 auto b=r102::read_bundle(plan);if(!b.real)throw std::runtime_error("storage test expects real bundle");r102::Reader reader(b);auto base=memory();std::cout<<"baseline\t"<<base.rss<<'\t'<<base.pss<<'\n';
 auto backend=ggml_backend_cpu_init();if(!backend)throw std::runtime_error("CPU backend failed");ggml_backend_cpu_set_n_threads(backend,1);
 std::vector<unsigned char>input(1024*1024),output(1024*1024);
 try{for(auto&e:b.entries){ggml_context*ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;
  try{ctx=ggml_init({1024*1024,nullptr,true});if(!ctx)throw std::runtime_error("metadata allocation failed");auto*t=ggml_new_tensor_2d(ctx,e.spec.type,e.spec.columns,e.spec.rows);buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)throw std::runtime_error("CPU weight storage failed");
   for(size_t off=0;off<e.spec.bytes;){size_t n=std::min(input.size(),(size_t)e.spec.bytes-off);reader.read(e.spec,off,input.data(),n);ggml_backend_tensor_set(t,input.data(),off,n);ggml_backend_tensor_get(t,output.data(),off,n);if(std::memcmp(input.data(),output.data(),n))throw std::runtime_error("CPU storage altered quantized bytes");off+=n;}
   auto peak=memory();auto bytes=ggml_backend_buffer_get_size(buffer);ggml_backend_buffer_free(buffer);buffer=nullptr;ggml_free(ctx);ctx=nullptr;auto after=memory();
   if(r102::open_files()!=0)throw std::runtime_error("reader fd leak");
   std::cout<<e.spec.name<<'\t'<<e.spec.bytes<<'\t'<<bytes<<'\t'<<peak.rss<<'\t'<<peak.pss<<'\t'<<after.rss<<'\t'<<after.pss<<'\n';
  }catch(...){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);throw;}
 }}catch(...){ggml_backend_free(backend);throw;}
 ggml_backend_free(backend);rusage usage{};getrusage(RUSAGE_SELF,&usage);std::cout<<"summary\tbytes_read="<<reader.bytes_read<<"\tmax_read="<<reader.max_read<<"\tpeak_rss_kib="<<usage.ru_maxrss<<"\tcompute_workspace=0\topen_files="<<r102::open_files()<<"\tnumeric_execution=false\n";
}
void real_run(const std::string&plan,const std::string&input,const std::string&original,const std::string&previous,const std::string&out){
 auto b=r102::read_bundle(plan);if(!b.real)throw std::runtime_error("real mode refuses fixture");
 auto read=[&](const std::string&path){std::vector<float>v(b.config.hidden);std::ifstream f(path,std::ios::binary);f.read((char*)v.data(),v.size()*4);if(!f||f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("input F32 length");return v;};
 auto x=read(input),x0=read(original);r99::KV state;if(previous!="-")state=r99::load_kv(previous,b.config);
 r102::LayerSource source(b); // Entry gate once; original guard owns runtime thresholds.
 if(!std::filesystem::create_directory(out))throw std::runtime_error("new output directory required");
 auto result=r99::execute(b.config,r102::specs(b),source,x,x0,state,true);
 auto write=[&](const std::string&name,const std::vector<float>&v){std::ofstream f(out+"/"+name,std::ios::binary);f.write((char*)v.data(),v.size()*4);f.close();if(!f)throw std::runtime_error("output failure");};
 write("layer-output.f32",result.output);r99::save_kv(out+"/next.kv",b.config,result.next);for(auto&[name,v]:result.captures)write(name+".f32",v);
 std::cout<<"input_position="<<state.position<<" next_position="<<result.next.position<<" weight_bytes="<<result.weight_bytes<<" graph_buffer_bytes="<<result.graph_bytes<<" kv_bytes="<<result.kv_bytes<<"\n";
}

}
int main(int argc,char**argv)try{
 if(argc<2)q104::real("missing dispatch");
 if(std::string(argv[1])=="--synthetic")q104::synthetic();
 else if(std::string(argv[1])!="--describe")q104::real("candidate CLI dispatch");
 if(argc==7&&std::string(argv[1])=="--real"){real_run(argv[2],argv[3],argv[4],argv[5],argv[6]);return 0;}
 if(argc==4&&std::string(argv[1])=="--synthetic"){synthetic(argv[2],argv[3]);return 0;}
 if(argc==3&&std::string(argv[1])=="--storage"){storage(argv[2]);return 0;}
 if(argc==3&&std::string(argv[1])=="--admission"){auto b=r102::read_bundle(argv[2]);r102::LayerSource source(b);throw std::runtime_error("admission probe never executes a graph");}
 if(argc==3&&std::string(argv[1])=="--describe"){auto b=r102::read_bundle(argv[2]);std::cout<<"layer="<<b.config.layer<<" real="<<b.real<<" tensors="<<b.entries.size()<<'\n';return 0;}
 throw std::runtime_error("only --synthetic PREFIX OUT, --storage PLAN, --describe PLAN, --admission PLAN supported");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
