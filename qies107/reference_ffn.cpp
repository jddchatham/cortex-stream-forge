// Independent whole-branch stock-GGML graph. No streamed executor, candidate
// graph builder or candidate projection function is compiled into this worker.
#include "support.h"
using namespace infra107;using namespace fixture107;
namespace {
void observe(ggml_type type,int vector){
 mark("before");
 {
  Backend backend;Workspace scratch;Blocks gate(type,0),up(type,1),down(type,2);Context c;
  auto*input_tensor=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,H,1);auto*gamma=ggml_new_tensor_1d(c.p,GGML_TYPE_F32,H);auto*postgamma=ggml_new_tensor_1d(c.p,GGML_TYPE_F32,H);
  auto*wg=ggml_new_tensor_2d(c.p,type,H,F);auto*wu=ggml_new_tensor_2d(c.p,type,H,F);auto*wd=ggml_new_tensor_2d(c.p,type,F,H);
  // Derived from stock src/models/ricursive.cpp: ffn_norm, gate/up,
  // swiglu_clamp, down, ffn_post_norm, then add ffn_inp (not the pre-attn x).
  auto*normalized=ggml_mul(c.p,ggml_rms_norm(c.p,input_tensor,EPS),gamma);
  auto*gated=ggml_mul_mat(c.p,wg,normalized);auto*raised=ggml_mul_mat(c.p,wu,normalized);
  auto*activated=ggml_swiglu_clamp(c.p,gated,raised,CLAMP);
  auto*projected=ggml_mul_mat(c.p,wd,activated);
  auto*post=ggml_mul(c.p,ggml_rms_norm(c.p,projected,EPS),postgamma);
  auto*result=ggml_add(c.p,post,input_tensor);
  std::vector<std::pair<std::string,ggml_tensor*>>taps={{"input-norm",normalized},{"gate",gated},{"up",raised},{"active",activated},{"down",projected},{"post-norm",post},{"final",result}};
  for(auto&tap:taps)ggml_set_output(tap.second);
  auto*graph=ggml_new_graph_custom(c.p,64,false);ggml_build_forward_expand(graph,result);
  const size_t weights=ggml_nbytes(wg)+ggml_nbytes(wu)+ggml_nbytes(wd);need(weights<=432537600,"fixed reference weight bound");
  Buffer buffer(c.p,backend.p,weights);need(buffer.bytes<=448*1024*1024,"whole reference FFN allocation bound");auto plan=scratch.plan(graph);
  std::vector<float>values(H),scale(H),postscale(H);
  for(int i=0;i<H;++i){values[i]=input(vector,i);scale[i]=fixture107::gamma(0,i);postscale[i]=fixture107::gamma(1,i);}
  ggml_backend_tensor_set(input_tensor,values.data(),0,H*4);ggml_backend_tensor_set(gamma,scale.data(),0,H*4);ggml_backend_tensor_set(postgamma,postscale.data(),0,H*4);
  auto fill=[&](const char*name,ggml_tensor*t,const Blocks&generator,int cols,int rows){
   std::vector<uint8_t>row(ggml_row_size(type,cols));Digest digest;
   for(int i=0;i<rows;++i){generator.row(i,cols,row.data());digest.add(row.data(),row.size());ggml_backend_tensor_set(t,row.data(),size_t(i)*row.size(),row.size());}
   weight_identity(name,digest,ggml_nbytes(t));
  };
  fill("gate",wg,gate,H,F);fill("up",wu,up,H,F);fill("down",wd,down,F,H);
  mark("reference.all-weights-live",plan.work_size,Context::bytes,ggml_used_mem(c.p),H*12,3*TEMPLATES*ggml_type_size(type));
  need(ggml_graph_compute(graph,&plan)==GGML_STATUS_SUCCESS,"independent stock FFN compute");mark("reference.graph-computed",plan.work_size,Context::bytes,ggml_used_mem(c.p));
  Captures captures(0);
  for(auto&tap:taps){std::vector<float>data(ggml_nelements(tap.second));ggml_backend_tensor_get(tap.second,data.data(),0,data.size()*4);captures.write(tap.first,data);}
  captures.publish();
 }
 mark("after-cleanup");need(live==0&&current_buffers==0&&current_weights==0,"stock FFN cleanup");summary();
}
}
int main(int argc,char**argv)try{
 if(argc<2||std::string(argv[1])!="--synthetic"){q104::real("QIES107 stock FFN observer: no real model interface");}
 q104::synthetic();need(argc==4,"usage: --synthetic q5|q4 VECTOR");observe(type_from(argv[2]),vector_from(argv[3]));return 0;
}catch(const std::exception&e){mark("after-failure-cleanup");summary();std::cerr<<"QIES107 reference failure: "<<e.what()<<" live="<<live<<"\n";return live==0?1:2;}
