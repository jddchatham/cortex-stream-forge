// Independent stock-GGML observer: one whole-matrix mul_mat, no candidate
// executor, tiled graph or candidate numerical source is compiled/linked here.
#include "admission.h"
#include "fixture.h"
#include "telemetry.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <openssl/evp.h>
#include <memory>
#include <iomanip>
#include <cmath>
#include <fcntl.h>
#include <cerrno>
using synthetic106::require;
namespace {
int owned=0;
const synthetic106::Case reference_cases[]={
#include "reference-cases.inc"
};
struct Owner {
 ggml_backend_t backend=nullptr;ggml_context*ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;EVP_MD_CTX*hash=nullptr;int output=-1;bool complete=false;
 ~Owner(){if(output>=0){close(output);--owned;}if(!complete)unlink("/outputs/output.partial");if(hash){EVP_MD_CTX_free(hash);--owned;}if(buffer){ggml_backend_buffer_free(buffer);--owned;}if(ctx){ggml_free(ctx);--owned;}if(backend){ggml_backend_free(backend);--owned;}}
};
void observe(const synthetic106::Case&c){
 diag106::sample("before",0,0,0,0,0,0,0,owned);
 {
  Owner o;o.backend=ggml_backend_cpu_init();require(o.backend,"stock CPU init");++owned;ggml_backend_cpu_set_n_threads(o.backend,1);
  constexpr size_t meta=1024*1024;o.ctx=ggml_init({meta,nullptr,true});require(o.ctx,"stock metadata");++owned;
  auto type=static_cast<ggml_type>(c.type);auto*matrix=ggml_new_tensor_2d(o.ctx,type,c.columns,c.rows);auto*vector=ggml_new_tensor_2d(o.ctx,GGML_TYPE_F32,c.columns,1);
  auto*product=ggml_mul_mat(o.ctx,matrix,vector);auto*graph=ggml_new_graph_custom(o.ctx,32,false);ggml_build_forward_expand(graph,product);
  o.buffer=ggml_backend_alloc_ctx_tensors(o.ctx,o.backend);require(o.buffer,"stock CPU allocation");++owned;
  require(ggml_backend_buffer_get_size(o.buffer)<=160*1024*1024,"stock single-weight budget");
  auto plan=ggml_graph_plan(graph,1,nullptr);std::cout<<"{\"planned_workspace_bytes\":"<<plan.work_size<<"}\n"<<std::flush;require(plan.work_size<=1024*1024,"stock workspace budget");std::vector<uint8_t>work(plan.work_size);plan.work_data=work.data();
  std::vector<float>input(c.columns),unquantized(c.columns),importance(c.columns,1.0f),result(c.rows);
  size_t row_size=ggml_row_size(type,c.columns);std::vector<uint8_t>packed(row_size);
  for(int j=0;j<c.columns;++j)input[j]=synthetic106::activation(j);
  ggml_backend_tensor_set(vector,input.data(),0,input.size()*sizeof(float));
  o.hash=EVP_MD_CTX_new();require(o.hash,"stock digest");++owned;require(EVP_DigestInit_ex(o.hash,EVP_sha256(),nullptr)==1,"stock digest init");
  // Independent reference fills a full single tensor once, then evaluates its
  // unpartitioned upstream graph exactly once. No whole layer/model is loaded.
  for(int row=0;row<c.rows;++row){
   for(int j=0;j<c.columns;++j)unquantized[j]=synthetic106::weight(row,j);
   require(ggml_quantize_chunk(type,unquantized.data(),packed.data(),0,1,c.columns,importance.data())==row_size,"stock quantized length");
   require(EVP_DigestUpdate(o.hash,packed.data(),packed.size())==1,"stock digest update");
   ggml_backend_tensor_set(matrix,packed.data(),row*row_size,row_size);
  }
  diag106::sample("weights-live",ggml_nbytes(matrix),ggml_backend_buffer_get_size(o.buffer),plan.work_size,meta,ggml_used_mem(o.ctx),(input.size()+result.size())*4,unquantized.size()*4+importance.size()*4+packed.size(),owned);
  require(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS,"stock compute");
  ggml_backend_tensor_get(product,result.data(),0,result.size()*4);for(auto f:result)require(std::isfinite(f),"stock nonfinite output");
  unsigned char hash[32];unsigned length=0;require(EVP_DigestFinal_ex(o.hash,hash,&length)==1&&length==32,"stock digest final");std::ostringstream hex;for(auto v:hash)hex<<std::hex<<std::setfill('0')<<std::setw(2)<<int(v);
  std::cout<<"{\"weight_sha256\":\""<<hex.str()<<"\",\"quantized_bytes\":"<<ggml_nbytes(matrix)<<",\"columns\":"<<c.columns<<",\"rows\":"<<c.rows<<",\"stock_unpartitioned_graph\":true,\"real_weights\":false}\n"<<std::flush;
  o.output=open("/outputs/output.partial",O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);require(o.output>=0,"stock exclusive output");++owned;
  size_t remaining=result.size()*4;auto*p=reinterpret_cast<const char*>(result.data());
  while(remaining){auto n=write(o.output,p,remaining);if(n<0&&errno==EINTR)continue;require(n>0,"stock output write");p+=n;remaining-=n;}
  require(fsync(o.output)==0&&link("/outputs/output.partial","/outputs/output.f32")==0,"stock output publication");unlink("/outputs/output.partial");o.complete=true;
 }
 diag106::sample("after-cleanup",0,0,0,0,0,0,0,owned);require(owned==0,"stock cleanup");
}
}
int main(int argc,char**argv){try{
 if(argc<2||std::string(argv[1])!="--synthetic")q104::real("QIES106 stock observer: real model interface absent");
 q104::synthetic();require(argc==3,"stock observer accepts only --synthetic FIXED_CASE");
 const synthetic106::Case*c=nullptr;for(const auto&q:reference_cases)if(std::string(q.name)==argv[2])c=&q;
 require(c,"unknown synthetic reference case; filenames forbidden");observe(*c);return 0;
}catch(const std::exception&e){diag106::sample("after-failure-cleanup",0,0,0,0,0,0,0,owned);std::cerr<<"QIES106 reference failure: "<<e.what()<<" live="<<owned<<"\n";return owned==0?1:2;}}
