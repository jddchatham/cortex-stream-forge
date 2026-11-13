#include "common.h"
using namespace a109;
namespace {
std::vector<float> projected(Backend&backend,Workspace&scratch,const InputBlocks&source,const std::vector<float>&input,int columns,int rows,const std::string&stage){
 need(input.size()==size_t(columns),"projection input size");fault(stage+".before-context");Context c;fault(stage+".after-context");
 auto*w=ggml_new_tensor_2d(c.p,source.type,columns,fixture107::TILE);auto*x=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,columns,1);auto*y=ggml_mul_mat(c.p,w,x);auto*g=ggml_new_graph_custom(c.p,32,false);ggml_build_forward_expand(g,y);
 fault(stage+".before-buffer");Buffer memory(c.p,backend.p,ggml_nbytes(w));need(memory.bytes<=5*1024*1024,"bounded tile allocation");fault(stage+".after-buffer");
 auto plan=scratch.plan(g);std::vector<uint8_t>row(ggml_row_size(source.type,columns));fault(stage+".before-output-allocation");std::vector<float>result(rows);fault(stage+".after-output-allocation");Digest hash;
 ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
 for(int first=0;first<rows;first+=fixture107::TILE){
  if(first==0)fault(stage+".before-load");
  for(int r=0;r<fixture107::TILE;++r){source.row(first+r,columns,row.data());hash.add(row.data(),row.size());ggml_backend_tensor_set(w,row.data(),size_t(r)*row.size(),row.size());}
  if(first==0){fault(stage+".after-load");mark(stage+".weights-live",plan.work_size,Context::bytes,ggml_used_mem(c.p),(input.size()+result.size())*4,row.size());fault(stage+".before-compute");}
  need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"streamed projection compute");if(first==0)fault(stage+".after-compute");
  ggml_backend_tensor_get(y,result.data()+first,0,fixture107::TILE*4);
  if(first==rows-fixture107::TILE)mark(stage+".last-tile",plan.work_size,Context::bytes,ggml_used_mem(c.p),(input.size()+result.size())*4,row.size());
 }
 weight_identity(stage.c_str(),hash,ggml_row_size(source.type,columns)*rows);fault(stage+".after-output");return result;
}
std::vector<float> pointwise(Backend&b,Workspace&w,const std::vector<float>&a,const std::vector<float>&z,const std::string&mode){Context c;auto*x=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,a.size(),1);auto*y=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,z.size(),1);ggml_tensor*out=nullptr;
 if(mode=="norm")out=ggml_mul(c.p,ggml_rms_norm(c.p,x,1e-6f),y);else if(mode=="gate")out=ggml_mul(c.p,x,ggml_sigmoid(c.p,y));else if(mode=="add")out=ggml_add(c.p,x,y);else {auto*l=ggml_new_tensor_1d(c.p,GGML_TYPE_F32,1);out=ggml_add(c.p,y,ggml_mul(c.p,ggml_sub(c.p,x,y),ggml_sigmoid(c.p,l)));auto*g=ggml_new_graph_custom(c.p,32,false);ggml_build_forward_expand(g,out);Buffer buffer(c.p,b.p,0);float lambda=-.375f;ggml_backend_tensor_set(l,&lambda,0,4);set(x,a);set(y,z);auto plan=w.plan(g);need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"value mixing");return read(out);}
 auto*g=ggml_new_graph_custom(c.p,32,false);ggml_build_forward_expand(g,out);Buffer buffer(c.p,b.p,0);set(x,a);set(y,z);auto plan=w.plan(g);need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"pointwise compute");return read(out);
}
std::pair<std::vector<float>,std::vector<float>>qk(Backend&b,Workspace&w,const Config&cfg,int pos,const std::vector<float>&q,const std::vector<float>&k,Captures&out){Context c;auto*q0=ggml_new_tensor_3d(c.p,GGML_TYPE_F32,D,NH,1);auto*k0=ggml_new_tensor_3d(c.p,GGML_TYPE_F32,D,NKV,1);auto*qg=ggml_new_tensor_1d(c.p,GGML_TYPE_F32,D);auto*kg=ggml_new_tensor_1d(c.p,GGML_TYPE_F32,D);auto*p=ggml_new_tensor_1d(c.p,GGML_TYPE_I32,1);
 auto*qn=ggml_mul(c.p,ggml_rms_norm(c.p,q0,1e-6f),qg);auto*kn=ggml_mul(c.p,ggml_rms_norm(c.p,k0,1e-6f),kg);auto*qr=qn;auto*kr=kn;
 if(cfg.sliding()){qr=ggml_rope_ext(c.p,qn,p,nullptr,128,GGML_ROPE_TYPE_NEOX,2097152,10000.f,1,0,1,32,1);kr=ggml_rope_ext(c.p,kn,p,nullptr,128,GGML_ROPE_TYPE_NEOX,2097152,10000.f,1,0,1,32,1);}
 auto*g=ggml_new_graph_custom(c.p,64,false);ggml_build_forward_expand(g,qr);ggml_build_forward_expand(g,kr);Buffer buffer(c.p,b.p,0);set(q0,q);set(k0,k);set(qg,gamma(D,1));set(kg,gamma(D,0));ggml_backend_tensor_set(p,&pos,0,4);auto plan=w.plan(g);fault("qk.before-compute");need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"QK norm/RoPE");fault("qk.after-compute");get(out,"q-norm",qn);get(out,"k-norm",kn);get(out,"q-current",qr);get(out,"k-current",kr);return{read(qr),read(kr)};
}
std::pair<std::vector<float>,State>attention(Backend&b,Workspace&w,const Config&cfg,const State&previous,const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,Captures&out){
 fault("kv.before-copy");State next=previous;fault("kv.after-copy");Context c;auto*qt=ggml_new_tensor_3d(c.p,GGML_TYPE_F32,D,NH,1);auto*kt=ggml_new_tensor_3d(c.p,GGML_TYPE_F32,D,NKV,1);auto*vt=ggml_new_tensor_3d(c.p,GGML_TYPE_F32,D,NKV,1);auto*keys=ggml_new_tensor_3d(c.p,GGML_TYPE_F16,D,NKV,cfg.slots);auto*values=ggml_new_tensor_3d(c.p,GGML_TYPE_F16,D,NKV,cfg.slots);auto*m=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,cfg.slots,1);
 auto*ks=ggml_view_3d(c.p,keys,D,NKV,1,keys->nb[1],keys->nb[2],previous.position*keys->nb[2]);auto*vs=ggml_view_3d(c.p,values,D,NKV,1,values->nb[1],values->nb[2],previous.position*values->nb[2]);auto*ck=ggml_cpy(c.p,kt,ks);auto*cv=ggml_cpy(c.p,vt,vs);
 auto*scores=ggml_mul_mat(c.p,ggml_permute(c.p,keys,0,2,1,3),ggml_permute(c.p,qt,0,2,1,3));need(ggml_prec_set_acc(scores,GGML_PREC_F32),"F32 attention accumulation");auto*prob=ggml_soft_max_ext(c.p,scores,m,1.f/std::sqrt(128.f),0);auto*vp=ggml_cont(c.p,ggml_transpose(c.p,ggml_permute(c.p,values,0,2,1,3)));auto*result=ggml_cont_2d(c.p,ggml_permute(c.p,ggml_mul_mat(c.p,vp,prob),0,2,1,3),Q,1);
 auto*gk=ggml_new_graph_custom(c.p,32,false);ggml_build_forward_expand(gk,ck);auto*gv=ggml_new_graph_custom(c.p,32,false);ggml_build_forward_expand(gv,cv);auto*g=ggml_new_graph_custom(c.p,64,false);ggml_build_forward_expand(g,result);
 fault("kv.before-buffer");Buffer buffer(c.p,b.p,0);need(buffer.bytes<=16*1024*1024,"attention workspace tensor bound");fault("kv.after-buffer");set(qt,q);set(kt,k);set(vt,v);set(m,mask(cfg,previous.position));ggml_backend_tensor_set(keys,previous.key.data(),0,previous.key.size()*2);ggml_backend_tensor_set(values,previous.value.data(),0,previous.value.size()*2);
 auto pk=w.plan(gk);need(ggml_graph_compute(gk,&pk)==GGML_STATUS_SUCCESS,"K write");fault("kv.after-key-update");auto pv=w.plan(gv);need(ggml_graph_compute(gv,&pv)==GGML_STATUS_SUCCESS,"V write");fault("kv.after-value-update");auto plan=w.plan(g);mark("attention.live",plan.work_size,Context::bytes,ggml_used_mem(c.p));fault("attention.before-compute");need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"attention compute");fault("attention.after-compute");get(out,"scores",scores);get(out,"probabilities",prob);get(out,"attention",result);
 ggml_backend_tensor_get(keys,next.key.data(),0,next.key.size()*2);ggml_backend_tensor_get(values,next.value.data(),0,next.value.size()*2);++next.position;fault("kv.after-download");return{read(result),std::move(next)};
}
std::vector<float> elementwise(Backend&backend,Workspace&scratch,const std::vector<float>&a,const std::vector<float>&b,const std::string&stage){
 fault(stage+".before-context");Context c;fault(stage+".after-context");
 const int n=int(a.size());auto*x=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,n,1);auto*z=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,n,1);ggml_tensor*y=nullptr;
 if(stage=="ffn-input-norm"||stage=="ffn-post-norm")y=ggml_mul(c.p,ggml_rms_norm(c.p,x,1e-6f),z);
 else if(stage=="ffn-active")y=ggml_swiglu_clamp(c.p,x,z,20.f);
 else{need(stage=="ffn-residual","unknown stage");y=ggml_add(c.p,x,z);}
 auto*g=ggml_new_graph_custom(c.p,32,false);ggml_build_forward_expand(g,y);
 fault(stage+".before-buffer");Buffer buffer(c.p,backend.p,0);need(buffer.bytes<=1024*1024,"activation tensor budget");fault(stage+".after-buffer");
 ggml_backend_tensor_set(x,a.data(),0,n*4);need(b.size()==a.size(),"elementwise shape");ggml_backend_tensor_set(z,b.data(),0,n*4);auto plan=scratch.plan(g);
 fault(stage+".before-compute");need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"elementwise compute");fault(stage+".after-compute");
 fault(stage+".before-output-allocation");std::vector<float>out(n);fault(stage+".after-output-allocation");ggml_backend_tensor_get(y,out.data(),0,n*4);mark(stage+".computed",plan.work_size,Context::bytes,ggml_used_mem(c.p),(a.size()+b.size()+out.size())*4);return out;
}
void execute(const Config&cfg,const State&old,Captures&out){
 Backend b;Workspace w;State pending;std::vector<float>residual;
 fault("after-backend");
 {InputBlocks qw(GGML_TYPE_Q4_K,0),kw(GGML_TYPE_Q8_0,1),vw(GGML_TYPE_Q8_0,2),gw(GGML_TYPE_Q4_K,3),ow(GGML_TYPE_Q6_K,4);auto x=activation(old.position,false),x0=activation(old.position,true);auto a=pointwise(b,w,x,gamma(H,0),"norm");out.write("input-norm",a);
 auto q=projected(b,w,qw,a,H,Q,"q");out.write("q-projection",q);auto k=projected(b,w,kw,a,H,K,"k");out.write("k-projection",k);auto vi=pointwise(b,w,a,x0,"mix");out.write("value-input",vi);std::vector<float>().swap(x0);auto v=projected(b,w,vw,vi,H,K,"v");out.write("v-current",v);std::vector<float>().swap(vi);auto rotated=qk(b,w,cfg,old.position,q,k,out);std::vector<float>().swap(q);std::vector<float>().swap(k);auto core=attention(b,w,cfg,old,rotated.first,rotated.second,v,out);std::vector<float>().swap(rotated.first);std::vector<float>().swap(rotated.second);std::vector<float>().swap(v);
 auto gate=projected(b,w,gw,a,H,Q,"gate");out.write("gate-projection",gate);std::vector<float>().swap(a);auto gated=pointwise(b,w,core.first,gate,"gate");out.write("gated",gated);std::vector<float>().swap(gate);std::vector<float>().swap(core.first);auto o=projected(b,w,ow,gated,Q,H,"output");out.write("output-projection",o);std::vector<float>().swap(gated);auto post=pointwise(b,w,o,gamma(H,1),"norm");out.write("post-norm",post);std::vector<float>().swap(o);auto final=pointwise(b,w,post,x,"add");out.write("attention-residual",final);fault("attention-residual.after");std::vector<float>().swap(post);residual=std::move(final);pending=std::move(core.second);}
 need(current_weights==0&&current_buffers==0,"attention resources retained across FFN boundary");mark("attention-released.before-ffn");fault("between-branches");
 {
  InputBlocks fg(GGML_TYPE_Q5_K,0),fu(GGML_TYPE_Q5_K,1),fd(GGML_TYPE_Q5_K,2);
  auto normalized=elementwise(b,w,residual,gamma(H,0),"ffn-input-norm");out.write("ffn-input-norm",normalized);
  auto gate=projected(b,w,fg,normalized,H,fixture107::F,"ffn-gate");out.write("ffn-gate",gate);
  auto up=projected(b,w,fu,normalized,H,fixture107::F,"ffn-up");out.write("ffn-up",up);std::vector<float>().swap(normalized);
  auto active=elementwise(b,w,gate,up,"ffn-active");out.write("ffn-active",active);std::vector<float>().swap(gate);std::vector<float>().swap(up);
  auto down=projected(b,w,fd,active,fixture107::F,H,"ffn-down");out.write("ffn-down",down);std::vector<float>().swap(active);
  auto post=elementwise(b,w,down,gamma(H,1),"ffn-post-norm");out.write("ffn-post-norm",post);std::vector<float>().swap(down);
  auto final=elementwise(b,w,post,residual,"ffn-residual");out.write("final",final);std::vector<float>().swap(post);std::vector<float>().swap(residual);std::vector<float>().swap(final);
 }
 need(current_weights==0&&current_buffers==0,"FFN resources retained before publication");mark("ffn-released.before-checkpoint");
 auto masks=mask(cfg,old.position);raw(out,"mask.f32",masks.data(),masks.size()*4);fault("state.before-serialization");checkpoint(out,cfg,pending);fault("state.after-serialization");state_record(cfg,pending,"next");
}
}
int main(int argc,char**argv)try{auto admitted=admit(argc,argv);requested_fault=argv[4];mark("before");{Captures out(0);execute(admitted.first,*authoritative,out);fault("after-cleanup-before-publish");out.publish();}need(identity(*active_config,*authoritative)==before_state,"authoritative state modified");mark("after-cleanup");summary();return 0;}catch(const std::exception&e){return failed(e);}
