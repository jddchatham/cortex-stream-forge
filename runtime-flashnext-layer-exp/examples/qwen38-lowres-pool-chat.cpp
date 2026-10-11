#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static void set_pos(llama_batch & b, llama_pos p) { b.pos[0]=p; b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.n_tokens=1; }
static llama_context * create_ctx(llama_model * m, int il) {
    char s[32],e[32]; std::snprintf(s,sizeof(s),"%d",il); std::snprintf(e,sizeof(e),"%d",il+1);
    setenv("CORTEX_QWEN35_SEGMENT_START",s,1); setenv("CORTEX_QWEN35_SEGMENT_END",e,1);
    if(!llama_cortex_rebind_layer(m,il)) return nullptr;
    llama_context_params cp=llama_context_default_params(); cp.n_ctx=2; cp.n_batch=1;
    llama_context * c=llama_init_from_model(m,cp); if(c) llama_set_embeddings_layer_inp(c,il+1,true); return c;
}
static llama_token step(llama_model * m, std::vector<llama_context*> & pool, int pool_n,
                        std::vector<std::vector<unsigned char>> & state, llama_token tok,
                        llama_pos p, std::vector<float> & h) {
    int nl=llama_model_n_layer(m), ne=llama_model_n_embd(m);
    for(int il=0;il<nl;++il) {
        int pi=il%pool_n; char s[32],e[32]; std::snprintf(s,sizeof(s),"%d",il); std::snprintf(e,sizeof(e),"%d",il+1);
        setenv("CORTEX_QWEN35_SEGMENT_START",s,1); setenv("CORTEX_QWEN35_SEGMENT_END",e,1);
        if(!llama_cortex_rebind_layer(m,il)) return -1;
        if(!pool[pi]) { pool[pi]=create_ctx(m,il); if(!pool[pi]) return -1; }
        llama_context * c=pool[pi];
        if(!state[il].empty() && !llama_state_set_data(c,state[il].data(),state[il].size())) return -1;
        llama_batch b;
        if(il==0) { b=llama_batch_init(1,0,1); b.token[0]=tok; } else { b=llama_batch_init(1,ne,1); std::copy(h.begin(),h.end(),b.embd); }
        set_pos(b,p); if(llama_decode(c,b)!=0) { llama_batch_free(b); return -1; }
        if(il+1<nl) { float * out=llama_get_embeddings_layer_inp(c,il+1); if(!out){llama_batch_free(b);return -1;} std::copy(out,out+ne,h.begin()); }
        else { float * l=llama_get_logits(c); int nv=llama_vocab_n_tokens(llama_model_get_vocab(m)); tok=(llama_token)(std::max_element(l,l+nv)-l); }
        llama_batch_free(b); size_t sz=llama_state_get_size(c); state[il].resize(sz); if(llama_state_get_data(c,state[il].data(),sz)!=sz)return -1;
    }
    return tok;
}
int main(int argc,char ** argv) {
    if(argc<3||argc>4)return 2; int pn=std::atoi(argv[2]); if(pn<1||pn>4)return 2; int maxgen=argc==4?std::atoi(argv[3]):2;
    setenv("CORTEX_LOW_RESIDENCY_LAYER","0",1); llama_backend_init(); llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    llama_model * m=llama_model_load_from_file(argv[1],mp); if(!m)return 3; int nl=llama_model_n_layer(m),ne=llama_model_n_embd(m);
    std::vector<llama_context*> pool(pn); std::vector<std::vector<unsigned char>> state(nl); std::vector<float> h(ne); const llama_vocab * v=llama_model_get_vocab(m); std::string line; llama_pos p=0;
    std::cout<<"pool="<<pn<<" ready\n> "<<std::flush;
    while(std::getline(std::cin,line)) { if(line=="/exit")break; int n=-llama_tokenize(v,line.c_str(),(int)line.size(),nullptr,0,p==0,true); std::vector<llama_token> q(n); if(llama_tokenize(v,line.c_str(),(int)line.size(),q.data(),n,p==0,true)<0)return 4; llama_token next=-1; for(auto t:q){next=step(m,pool,pn,state,t,p++,h);if(next<0)return 5;} for(int i=0;i<maxgen;++i){char piece[256];int z=llama_token_to_piece(v,next,piece,sizeof(piece),0,false);if(z>0)std::cout.write(piece,z)<<std::flush;next=step(m,pool,pn,state,next,p++,h);if(next<0)return 6;} std::cout<<"\n> "<<std::flush;}
    for(auto c:pool)if(c)llama_free(c); llama_model_free(m); llama_backend_free(); return 0;
}
