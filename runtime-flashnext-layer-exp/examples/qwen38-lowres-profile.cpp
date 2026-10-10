#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using forge_clock = std::chrono::steady_clock;
static double ms(forge_clock::time_point a, forge_clock::time_point b) { return std::chrono::duration<double, std::milli>(b-a).count(); }
static void pos(llama_batch & b, llama_pos p) { b.pos[0]=p; b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.n_tokens=1; }

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1); llama_backend_init();
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    auto t0=forge_clock::now(); llama_model * model=llama_model_load_from_file(argv[1],mp); auto t1=forge_clock::now();
    if (!model) return 3;
    int nl=llama_model_n_layer(model), ne=llama_model_n_embd(model); std::vector<float> h(ne); std::vector<std::vector<unsigned char>> states(nl);
    llama_token tok=9707; llama_pos p=0; double rebind=0, create=0, decode=0, save=0, restore=0;
    for (int il=0; il<nl; ++il) {
        char s[32],e[32]; std::snprintf(s,sizeof(s),"%d",il); std::snprintf(e,sizeof(e),"%d",il+1);
        setenv("CORTEX_QWEN35_SEGMENT_START",s,1); setenv("CORTEX_QWEN35_SEGMENT_END",e,1);
        auto a=forge_clock::now(); bool ok=llama_cortex_rebind_layer(model,il); auto b=forge_clock::now(); rebind+=ms(a,b); if(!ok) return 4;
        llama_context_params cp=llama_context_default_params(); cp.n_ctx=2; cp.n_batch=1;
        a=forge_clock::now(); llama_context * ctx=llama_init_from_model(model,cp); b=forge_clock::now(); create+=ms(a,b); if(!ctx) return 5;
        llama_set_embeddings_layer_inp(ctx,il+1,true);
        if(!states[il].empty()) { a=forge_clock::now(); if(!llama_state_set_data(ctx,states[il].data(),states[il].size())) return 6; b=forge_clock::now(); restore+=ms(a,b); }
        llama_batch batch;
        if(il==0) { batch=llama_batch_init(1,0,1); batch.token[0]=tok; }
        else { batch=llama_batch_init(1,ne,1); std::copy(h.begin(),h.end(),batch.embd); }
        pos(batch,p); a=forge_clock::now(); int rc=llama_decode(ctx,batch); b=forge_clock::now(); decode+=ms(a,b); if(rc) return 7;
        if(il+1<nl) { float * out=llama_get_embeddings_layer_inp(ctx,il+1); if(!out)return 8; std::copy(out,out+ne,h.begin()); }
        else { float * logits=llama_get_logits(ctx); int nv=llama_vocab_n_tokens(llama_model_get_vocab(model)); tok=(llama_token)(std::max_element(logits,logits+nv)-logits); }
        llama_batch_free(batch); size_t sz=llama_state_get_size(ctx); states[il].resize(sz); a=forge_clock::now(); if(llama_state_get_data(ctx,states[il].data(),sz)!=sz)return 9; b=forge_clock::now(); save+=ms(a,b); llama_free(ctx);
    }
    auto t2=forge_clock::now(); std::printf("profile load_ms=%.1f total_ms=%.1f rebind_ms=%.1f context_create_ms=%.1f decode_ms=%.1f state_restore_ms=%.1f state_save_ms=%.1f token=%d\n",ms(t0,t1),ms(t0,t2),rebind,create,decode,restore,save,(int)tok);
    llama_model_free(model); llama_backend_free();
}
