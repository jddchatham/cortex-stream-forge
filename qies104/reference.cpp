#include "admission.h"
#include "llama.h"
#include "ggml-backend.h"
#include <fstream>
#include <sys/resource.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
struct Capture { std::string root;int position; };
bool callback(ggml_tensor *t,bool ask,void *data) {
 auto*c=(Capture*)data;
 bool wanted=t->type==GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nbytes(t)<8*1024*1024;
 if(ask)return wanted;
 if(wanted){std::vector<char>b(ggml_nbytes(t));ggml_backend_tensor_get(t,b.data(),0,b.size());std::ofstream f(c->root+"/p"+std::to_string(c->position)+"-"+t->name+".bin",std::ios::binary);f.write(b.data(),b.size());}
 return true;
}
int main(int argc,char**argv)try{
 if(argc>1&&std::string(argv[1])=="--real")q104::real("stock reference CLI");
 q104::synthetic();
 if(argc!=3)return 2;
 if(std::string(argv[1])=="--metadata" || std::string(argv[1])=="--structure"){
 llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.vocab_only=std::string(argv[1])=="--metadata";mp.no_alloc=!mp.vocab_only;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.use_extra_bufts=false;
 auto*m=llama_model_load_from_file(argv[2],mp);if(!m)return 10;
 char desc[512];llama_model_desc(m,desc,sizeof(desc));std::cout<<"STOCK_METADATA_ONLY "<<desc<<" layers="<<llama_model_n_layer(m)<<" hidden="<<llama_model_n_embd(m)<<" no_decode=true\n";
 bool ok=llama_model_n_layer(m)==80 && llama_model_n_embd(m)==5120;llama_model_free(m);llama_backend_free();return ok?0:11;
 }
 std::string file=argv[1];q104::fixture_file(file);
 if(file.find("synthetic.gguf")==std::string::npos)return 3;
 std::error_code ec;auto bytes=std::filesystem::file_size(file,ec);if(!ec && bytes>16*1024*1024)return 3;
 // This executable intentionally cannot serve as a real-model runner.
 llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.check_tensors=true;mp.use_extra_bufts=false;
 auto*m=llama_model_load_from_file(argv[1],mp);if(!m)return 4;
 if(llama_model_n_embd(m)!=256 || llama_model_n_layer(m)!=1){llama_model_free(m);return 5;}
 Capture cap{argv[2],0};std::filesystem::create_directory(cap.root);
 auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=1;cp.n_ubatch=1;cp.n_threads=1;cp.n_threads_batch=1;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=false;cp.op_offload=false;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.cb_eval=callback;cp.cb_eval_user_data=&cap;
 auto*ctx=llama_init_from_model(m,cp);if(!ctx){llama_model_free(m);return 6;}
 auto b=llama_batch_init(1,256,1);b.n_tokens=1;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=true;
 for(int p=0;p<4;++p){cap.position=p;b.pos[0]=p;for(int i=0;i<256;++i)b.embd[i]=float((i+p)%11-5)*.1f;std::ofstream x(cap.root+"/input-"+std::to_string(p)+".bin",std::ios::binary);x.write((char*)b.embd,256*4);if(llama_decode(ctx,b)!=0)return 7;}
 llama_batch_free(b);llama_free(ctx);llama_model_free(m);llama_backend_free();rusage usage{};getrusage(RUSAGE_SELF,&usage);std::cerr<<"reference_peak_rss_kib="<<usage.ru_maxrss<<"\n";std::cout<<"PASS stock synthetic one layer, four positions\n";
}

catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
