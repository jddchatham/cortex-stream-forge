#include "admission.h"
#include "fixture.h"
#include "telemetry.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <array>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <openssl/evp.h>
#include <iomanip>
using synthetic106::require;
namespace {
std::atomic<int> live{0};
const synthetic106::Case cases[]={
#include "candidate-cases.inc"
};
struct Context {
 ggml_context*p=nullptr;size_t capacity=1024*1024;
 Context(){p=ggml_init({capacity,nullptr,true});require(p,"metadata allocation failed");++live;}
 ~Context(){if(p){ggml_free(p);--live;}}Context(const Context&)=delete;
};
struct Backend {
 ggml_backend_t p=nullptr;
 Backend(){p=ggml_backend_cpu_init();require(p,"CPU init failed");ggml_backend_cpu_set_n_threads(p,1);++live;}
 ~Backend(){if(p){ggml_backend_free(p);--live;}}Backend(const Backend&)=delete;
};
struct CpuBuffer {
 ggml_backend_buffer_t p=nullptr;
 CpuBuffer(ggml_context*c,ggml_backend_t b){
  p=ggml_backend_alloc_ctx_tensors(c,b);require(p,"CPU tensor allocation failed");++live;
  if(ggml_backend_buffer_get_size(p)>5*1024*1024){ggml_backend_buffer_free(p);p=nullptr;--live;throw std::runtime_error("candidate CPU tensor budget exceeded");}
 }
 ~CpuBuffer(){if(p){ggml_backend_buffer_free(p);--live;}}CpuBuffer(const CpuBuffer&)=delete;
};
struct Hash {
 EVP_MD_CTX*p=nullptr;
 Hash(){p=EVP_MD_CTX_new();require(p,"digest context");if(EVP_DigestInit_ex(p,EVP_sha256(),nullptr)!=1){EVP_MD_CTX_free(p);p=nullptr;throw std::runtime_error("digest init");}++live;}
 ~Hash(){if(p){EVP_MD_CTX_free(p);--live;}}
 void update(const void*b,size_t n){require(EVP_DigestUpdate(p,b,n)==1,"digest update");}
 std::string final(){unsigned char b[32];unsigned n;require(EVP_DigestFinal_ex(p,b,&n)==1&&n==32,"digest final");std::ostringstream s;for(auto v:b)s<<std::hex<<std::setfill('0')<<std::setw(2)<<int(v);return s.str();}
};
struct Output {
 int fd=-1;bool done=false;
 Output(){fd=open("/outputs/output.partial",O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);require(fd>=0,"exclusive output create failed");++live;}
 ~Output(){if(fd>=0){close(fd);--live;}if(!done)unlink("/outputs/output.partial");}
 void write(const void*b,size_t n){auto*p=static_cast<const char*>(b);while(n){ssize_t v=::write(fd,p,n);if(v<0&&errno==EINTR)continue;require(v>0,"output write failed");p+=v;n-=v;}}
 void publish(){require(fsync(fd)==0,"output sync failed");require(link("/outputs/output.partial","/outputs/output.f32")==0,"exclusive output publish failed");unlink("/outputs/output.partial");done=true;}
};
// No file/source callback is accepted here. This interface can only construct
// its built-in deterministic synthetic weights; real file-backed adaptation is
// explicitly NOT present. The QIES104 real gate and geometry budget are unchanged.
void execute(const synthetic106::Case&c,int tile,const std::string&fail){
 const auto type=static_cast<ggml_type>(c.type);size_t row_bytes=ggml_row_size(type,c.columns);
 require(c.rows%tile==0&&tile>=16&&tile<=128,"fixed synthetic tile budget");
 diag106::sample("before",0,0,0,0,0,0,0,live);
 {
  Backend backend;Context context;
  auto*w=ggml_new_tensor_2d(context.p,type,c.columns,tile);
  auto*x=ggml_new_tensor_2d(context.p,GGML_TYPE_F32,c.columns,1);
  auto*y=ggml_mul_mat(context.p,w,x);
  auto*g=ggml_new_graph_custom(context.p,32,false);ggml_build_forward_expand(g,y);
  CpuBuffer buffer(context.p,backend.p);
  auto plan=ggml_graph_plan(g,1,nullptr);std::cout<<"{\"planned_workspace_bytes\":"<<plan.work_size<<"}\n"<<std::flush;require(plan.work_size<=1024*1024,"workspace budget");
  std::vector<uint8_t>workspace(plan.work_size);plan.work_data=workspace.data();
  std::vector<float>input(c.columns),row(c.columns),imatrix(c.columns,1.0f),output(tile);
  std::vector<uint8_t>quantized(row_bytes);Hash weights;
  for(int i=0;i<c.columns;++i)input[i]=synthetic106::activation(i);
  ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
  if(fail=="after-input")throw std::runtime_error("injected after-input");
  Output stream;
  for(int start=0;start<c.rows;start+=tile){
   for(int r=0;r<tile;++r){
    for(int i=0;i<c.columns;++i)row[i]=synthetic106::weight(start+r,i);
    auto n=ggml_quantize_chunk(type,row.data(),quantized.data(),0,1,c.columns,imatrix.data());
    require(n==row_bytes,"quantized byte count");weights.update(quantized.data(),quantized.size());
    ggml_backend_tensor_set(w,quantized.data(),size_t(r)*row_bytes,row_bytes);
   }
   if(start==tile*2&&fail=="after-load")throw std::runtime_error("injected after-load");
   if(start==0||start==c.rows-tile||start%(tile*64)==0)
    diag106::sample("weights-live",row_bytes*tile,ggml_backend_buffer_get_size(buffer.p),plan.work_size,context.capacity,ggml_used_mem(context.p),(input.size()+output.size())*4,row.size()*4+imatrix.size()*4+quantized.size(),live,start/tile);
   require(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"CPU graph compute failed");
   if(start==tile*2&&fail=="after-compute")throw std::runtime_error("injected after-compute");
   ggml_backend_tensor_get(y,output.data(),0,output.size()*4);
   for(float v:output)require(std::isfinite(v),"nonfinite output");
   stream.write(output.data(),output.size()*4);
  }
  std::cout<<"{\"weight_sha256\":\""<<weights.final()<<"\",\"quantized_bytes\":"<<row_bytes*c.rows<<",\"columns\":"<<c.columns<<",\"rows\":"<<c.rows<<",\"tile_rows\":"<<tile<<",\"threads\":1,\"real_weights\":false}\n"<<std::flush;
  if(fail=="before-publish")throw std::runtime_error("injected before-publish");
  stream.publish();
 }
 diag106::sample("after-cleanup",0,0,0,0,0,0,0,live);
 require(live==0,"resource leak after successful execution");
}
}
int main(int argc,char**argv){try{
 if(argc<2||std::string(argv[1])!="--synthetic")q104::real("QIES106 prototype: real model interface absent");
 q104::synthetic();require(argc==5,"usage: --synthetic FIXED_CASE TILE FAULT");
 const synthetic106::Case*selected=nullptr;for(const auto&c:cases)if(std::string(c.name)==argv[2])selected=&c;
 require(selected,"unrecognized built-in synthetic case; file paths forbidden");
 const std::string tilearg=argv[3];require(tilearg=="16"||tilearg=="64"||tilearg=="128","unvalidated tile shape");
 const std::string failure=argv[4];require(failure=="none"||failure=="after-input"||failure=="after-load"||failure=="after-compute"||failure=="before-publish","unknown fault");
 execute(*selected,std::stoi(tilearg),failure);return 0;
}catch(const std::exception&e){diag106::sample("after-failure-cleanup",0,0,0,0,0,0,0,live);std::cerr<<"QIES106 FAILURE: "<<e.what()<<" live="<<live<<"\n";return live==0?1:2;}}
