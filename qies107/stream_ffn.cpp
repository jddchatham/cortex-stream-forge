#include "support.h"
using namespace infra107;using namespace fixture107;
namespace {
std::vector<float> projected(Backend&backend,Workspace&scratch,const Blocks&source,const std::vector<float>&input,int columns,int rows,const std::string&stage){
 need(input.size()==size_t(columns),"projection input size");fault(stage+".before-context");Context c;fault(stage+".after-context");
 auto*w=ggml_new_tensor_2d(c.p,source.type,columns,TILE);auto*x=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,columns,1);auto*y=ggml_mul_mat(c.p,w,x);auto*g=ggml_new_graph_custom(c.p,32,false);ggml_build_forward_expand(g,y);
 fault(stage+".before-buffer");Buffer memory(c.p,backend.p,ggml_nbytes(w));need(memory.bytes<=5*1024*1024,"bounded tile allocation");fault(stage+".after-buffer");
 auto plan=scratch.plan(g);std::vector<uint8_t>row(ggml_row_size(source.type,columns));fault(stage+".before-output-allocation");std::vector<float>result(rows);fault(stage+".after-output-allocation");Digest hash;
 ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
 for(int first=0;first<rows;first+=TILE){
  if(first==0)fault(stage+".before-load");
  for(int r=0;r<TILE;++r){source.row(first+r,columns,row.data());hash.add(row.data(),row.size());ggml_backend_tensor_set(w,row.data(),size_t(r)*row.size(),row.size());}
  if(first==0){fault(stage+".after-load");mark(stage+".weights-live",plan.work_size,Context::bytes,ggml_used_mem(c.p),(input.size()+result.size())*4,row.size());fault(stage+".before-compute");}
  need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"streamed projection compute");if(first==0)fault(stage+".after-compute");
  ggml_backend_tensor_get(y,result.data()+first,0,TILE*4);
  if(first==rows-TILE)mark(stage+".last-tile",plan.work_size,Context::bytes,ggml_used_mem(c.p),(input.size()+result.size())*4,row.size());
 }
 weight_identity(stage.c_str(),hash,ggml_row_size(source.type,columns)*rows);fault(stage+".after-output");return result;
}
std::vector<float> elementwise(Backend&backend,Workspace&scratch,const std::vector<float>&a,const std::vector<float>&b,const std::string&stage){
 fault(stage+".before-context");Context c;fault(stage+".after-context");
 const int n=int(a.size());auto*x=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,n,1);auto*z=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,n,1);ggml_tensor*y=nullptr;
 if(stage=="input-norm"||stage=="post-norm")y=ggml_mul(c.p,ggml_rms_norm(c.p,x,EPS),z);
 else if(stage=="active")y=ggml_swiglu_clamp(c.p,x,z,CLAMP);
 else{need(stage=="residual","unknown stage");y=ggml_add(c.p,x,z);}
 auto*g=ggml_new_graph_custom(c.p,32,false);ggml_build_forward_expand(g,y);
 fault(stage+".before-buffer");Buffer buffer(c.p,backend.p,0);need(buffer.bytes<=1024*1024,"activation tensor budget");fault(stage+".after-buffer");
 ggml_backend_tensor_set(x,a.data(),0,n*4);need(b.size()==a.size(),"elementwise shape");ggml_backend_tensor_set(z,b.data(),0,n*4);auto plan=scratch.plan(g);
 fault(stage+".before-compute");need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"elementwise compute");fault(stage+".after-compute");
 fault(stage+".before-output-allocation");std::vector<float>out(n);fault(stage+".after-output-allocation");ggml_backend_tensor_get(y,out.data(),0,n*4);mark(stage+".computed",plan.work_size,Context::bytes,ggml_used_mem(c.p),(a.size()+b.size()+out.size())*4);return out;
}
void execute(ggml_type type,int vector,int repeats){
 mark("before");
 {
  fault("before-backend");Backend backend;fault("after-backend");fault("before-workspace");Workspace scratch;fault("after-workspace");
  fault("before-fixtures");fault("gate-fixture.before");Blocks gate(type,0);fault("gate-fixture.after");fault("up-fixture.before");Blocks up(type,1);fault("up-fixture.after");fault("down-fixture.before");Blocks down(type,2);fault("down-fixture.after");fault("after-fixtures");
  for(int sequence=0;sequence<repeats;++sequence){
   fault("before-host-activations");std::vector<float>residual(H),gamma(H),postgamma(H);for(int i=0;i<H;++i){residual[i]=input(vector,i);gamma[i]=fixture107::gamma(0,i);postgamma[i]=fixture107::gamma(1,i);}fault("after-host-activations");
   fault("before-capture-owner");Captures captures(sequence);fault("after-capture-owner");
   auto normalized=elementwise(backend,scratch,residual,gamma,"input-norm");captures.write("input-norm",normalized);std::vector<float>().swap(gamma);
   auto g=projected(backend,scratch,gate,normalized,H,F,"gate");captures.write("gate",g);fault("gate.after-phase");
   auto u=projected(backend,scratch,up,normalized,H,F,"up");captures.write("up",u);fault("up.after-phase");std::vector<float>().swap(normalized);
   auto active=elementwise(backend,scratch,g,u,"active");captures.write("active",active);std::vector<float>().swap(g);std::vector<float>().swap(u);fault("active.after-phase");
   auto d=projected(backend,scratch,down,active,F,H,"down");captures.write("down",d);std::vector<float>().swap(active);fault("down.after-phase");
   auto post=elementwise(backend,scratch,d,postgamma,"post-norm");captures.write("post-norm",post);std::vector<float>().swap(d);std::vector<float>().swap(postgamma);
   auto final=elementwise(backend,scratch,post,residual,"residual");captures.write("final",final);std::vector<float>().swap(post);std::vector<float>().swap(residual);std::vector<float>().swap(final);captures.publish();
   mark("branch-complete");
  }
 }
 mark("after-cleanup");need(live==0&&current_buffers==0&&current_weights==0,"successful FFN cleanup");summary();
}
const char*faults[]={
#include "faults.inc"
};
}
int main(int argc,char**argv)try{
 if(argc<2||std::string(argv[1])!="--synthetic"){q104::real("QIES107 complete FFN prototype: no real model interface");}
 q104::synthetic();
 if(argc==3&&std::string(argv[2])=="--list-faults"){for(auto*s:faults)std::cout<<s<<"\n";return 0;}
 need(argc==6,"usage: --synthetic q5|q4 VECTOR FAULT REPEATS");auto type=type_from(argv[2]);int v=vector_from(argv[3]);requested_fault=argv[4];bool known=requested_fault=="none";for(auto*f:faults)if(requested_fault==f)known=true;need(known,"unknown fault");need(std::string(argv[5])=="1"||std::string(argv[5])=="3","bounded repeats only");execute(type,v,argv[5][0]-'0');return 0;
}catch(const std::exception&e){mark("after-failure-cleanup");summary();std::cerr<<"QIES107 FAILURE: "<<e.what()<<" live="<<live<<"\n";return live==0?1:2;}
