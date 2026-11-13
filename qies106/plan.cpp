// Planning ONLY: no tensor buffer allocation, work-buffer allocation, compute,
// model I/O or GGUF loading. Even the advertised context is metadata only.
#include "admission.h"
#include "telemetry.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <stdexcept>
namespace {
struct Metadata {
 ggml_context*c=nullptr;
 Metadata(){c=ggml_init({2*1024*1024,nullptr,true});if(!c)throw std::runtime_error("metadata allocation");}
 ~Metadata(){ggml_free(c);}
};
void measure(const char*label,int context){
 Metadata meta;auto*c=meta.c;auto*g=ggml_new_graph_custom(c,256,false);
 auto input=[&](ggml_type type,int n,int m=1){auto*t=ggml_new_tensor_2d(c,type,n,m);ggml_set_input(t);ggml_set_output(t);return t;};
 ggml_tensor*out=nullptr;
 if(std::string(label)=="hidden-rms")out=ggml_mul(c,ggml_rms_norm(c,input(GGML_TYPE_F32,5120),1e-6f),input(GGML_TYPE_F32,5120));
 else if(std::string(label)=="q-k-rms"){
  for(int heads:{80,8}){
   auto*x=ggml_new_tensor_3d(c,GGML_TYPE_F32,128,heads,1);ggml_set_input(x);ggml_set_output(x);
   auto*y=ggml_mul(c,ggml_rms_norm(c,x,1e-6f),input(GGML_TYPE_F32,128));ggml_set_output(y);ggml_build_forward_expand(g,y);
  }
 }else if(std::string(label)=="value-residual"){
  auto*a=input(GGML_TYPE_F32,5120);auto*x0=input(GGML_TYPE_F32,5120);auto*lambda=input(GGML_TYPE_F32,1);
  out=ggml_add(c,x0,ggml_mul(c,ggml_sub(c,a,x0),ggml_sigmoid(c,lambda)));
 }else if(std::string(label)=="attention-sigmoid-gating")out=ggml_mul(c,input(GGML_TYPE_F32,10240),ggml_sigmoid(c,input(GGML_TYPE_F32,10240)));
 else if(std::string(label)=="swiglu-clamp")out=ggml_swiglu_clamp(c,input(GGML_TYPE_F32,40960),input(GGML_TYPE_F32,40960),20.f);
 else if(std::string(label)=="attention"){
  auto*q=ggml_new_tensor_3d(c,GGML_TYPE_F32,128,80,1);auto*k=ggml_new_tensor_3d(c,GGML_TYPE_F16,128,8,context);auto*v=ggml_new_tensor_3d(c,GGML_TYPE_F16,128,8,context);for(auto*t:{q,k,v}){ggml_set_input(t);ggml_set_output(t);}
  auto*qp=ggml_permute(c,q,0,2,1,3);auto*kp=ggml_permute(c,k,0,2,1,3);auto*vp=ggml_permute(c,v,0,2,1,3);
  auto*kq=ggml_mul_mat(c,kp,qp);if(!ggml_prec_set_acc(kq,GGML_PREC_F32))throw std::runtime_error("precision");
  auto*probability=ggml_soft_max_ext(c,kq,input(GGML_TYPE_F32,context),1.0f/std::sqrt(128.f),0);
  vp=ggml_cont(c,ggml_transpose(c,vp));out=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vp,probability),0,2,1,3),10240,1);
 }else throw std::runtime_error("unknown plan");
 if(out){ggml_set_output(out);ggml_build_forward_expand(g,out);}
 auto allocator=ggml_gallocr_new(ggml_backend_cpu_buffer_type());if(!allocator)throw std::runtime_error("planner metadata");size_t bytes=0;
 // *_size uses no_alloc=true. Never substitute reserve()/alloc_graph().
 ggml_gallocr_reserve_n_size(allocator,g,nullptr,nullptr,&bytes);ggml_gallocr_free(allocator);
 auto plan=ggml_graph_plan(g,1,nullptr);
 std::cout<<"{\"stage\":\""<<label<<"\",\"context\":"<<context<<",\"ggml_planned_tensor_buffer_bytes\":"<<bytes<<",\"ggml_planned_workspace_bytes\":"<<plan.work_size<<",\"graph_metadata_used_bytes\":"<<ggml_used_mem(c)<<",\"tensor_buffers_allocated\":false,\"arithmetic_executed\":false}\n";
}
}
int main(int argc,char**argv)try{
 if(argc!=1)q104::real("QIES106 planning utility: only metadata-only fixed plans supported");q104::synthetic();
 diag106::sample("before-planning",0,0,0,0,0,0,0,0);
 for(auto*s:{"hidden-rms","q-k-rms","value-residual","attention-sigmoid-gating","swiglu-clamp"})measure(s,0);
 for(int context:{256,1024,4096,16384,32768,65536,2097152})measure("attention",context);
 diag106::sample("after-planning",0,0,0,0,0,0,0,0);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
