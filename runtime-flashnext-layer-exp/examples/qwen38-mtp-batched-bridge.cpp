#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

static uint64_t fnv(const void * p, size_t n) { uint64_t h=1469598103934665603ull; auto b=(const unsigned char*)p; for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;} return h; }

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    llama_backend_init();
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    llama_model * m=llama_model_load_from_file(argv[1],mp); if(!m) return 3;
    const int L=llama_model_n_layer(m), E=llama_model_n_embd(m), V=llama_vocab_n_tokens(llama_model_get_vocab(m));
    const std::vector<llama_token> tok={248045,846,198,12675,248046,198,248045,74455,198}; const int K=tok.size();
    std::vector<std::vector<uint8_t>> states(L); std::vector<float> h((size_t)K*E), hn;
    for(int il=0;il<L;++il){
        char a[16],b[16]; snprintf(a,sizeof(a),"%d",il); snprintf(b,sizeof(b),"%d",il+1);
        setenv("CORTEX_QWEN35_SEGMENT_START",a,1); setenv("CORTEX_QWEN35_SEGMENT_END",b,1);
        if(!llama_cortex_rebind_layer(m,il)) return 10;
        auto cp=llama_context_default_params(); cp.n_ctx=256; cp.n_batch=K;
        llama_context * c=llama_init_from_model(m,cp); if(!c) return 11;
        if(il+1<L) llama_set_embeddings_layer_inp(c,il+1,true);
        llama_set_embeddings_nextn(c,il==L-1,false);
        if(!states[il].empty() && llama_state_set_data(c,states[il].data(),states[il].size())==0) return 12;
        llama_batch x=il==0?llama_batch_init(K,0,1):llama_batch_init(K,E,1);
        for(int i=0;i<K;++i){ if(il==0)x.token[i]=tok[i]; else std::copy(h.begin()+(size_t)i*E,h.begin()+(size_t)(i+1)*E,x.embd+(size_t)i*E); x.pos[i]=i; x.n_seq_id[i]=1; x.seq_id[i][0]=0; x.logits[i]=(i==K-1); }
        x.n_tokens=K; if(llama_decode(c,x)!=0) return 13;
        if(il+1<L){ float * out=llama_get_embeddings_layer_inp(c,il+1); if(!out)return 14; std::copy(out,out+(size_t)K*E,h.begin()); }
        else { float * out=llama_get_embeddings_nextn_ith(c,K-1); if(!out)return 15; hn.assign(out,out+E); }
        size_t sz=llama_state_get_size(c); states[il].resize(sz); if(llama_state_get_data(c,states[il].data(),sz)!=sz)return 16;
        llama_batch_free(x); llama_free(c);
    }
    FILE * f=fopen("qwen38-mtp-hnextn.f32","wb"); if(!f||fwrite(hn.data(),sizeof(float),hn.size(),f)!=hn.size())return 20; fclose(f);
    double sum=0,norm=0; float mn=hn[0],mx=hn[0]; for(float x:hn){mn=std::min(mn,x);mx=std::max(mx,x);sum+=x;norm+=double(x)*x;}
    printf("HNEXTN shape=[%d] dtype=f32 bytes=%zu position=8 fnv=%016llx min=%g max=%g mean=%g norm=%g\n",E,hn.size()*sizeof(float),(unsigned long long)fnv(hn.data(),hn.size()*sizeof(float)),mn,mx,sum/hn.size(),sqrt(norm));
    // Independent MTP context. The target model remains the provider of the
    // shared embedding/output tensors; no target layers are decoded here.
    // Rebind block 64 before constructing the MTP context: graph reservation
    // validates the NextN tensor descriptors during context creation.
    if(!llama_cortex_rebind_layer(m,64))return 30;
    auto mp2=llama_context_default_params(); mp2.ctx_type=LLAMA_CONTEXT_TYPE_MTP; mp2.n_ctx=256; mp2.n_batch=1;
    llama_context * mtp=llama_init_from_model(m,mp2); if(!mtp)return 31; llama_set_embeddings_nextn(mtp,true,true);
    llama_batch y=llama_batch_init(1,E,1); y.token[0]=248068; std::copy(hn.begin(),hn.end(),y.embd); y.pos[0]=9; y.n_seq_id[0]=1; y.seq_id[0][0]=0; y.logits[0]=1; y.n_tokens=1;
    if(llama_decode(mtp,y)!=0)return 32; float * z=llama_get_logits(mtp); if(!z)return 33;
    std::vector<int> ids(V);std::iota(ids.begin(),ids.end(),0);std::partial_sort(ids.begin(),ids.begin()+10,ids.end(),[z](int a,int b){return z[a]>z[b];});
    printf("MTP_TOP");for(int i=0;i<10;++i)printf(" %d=%.8g",ids[i],z[ids[i]]);printf("\nMTP_LOGIT_FNV %016llx EOS246=%g\n",(unsigned long long)fnv(z,(size_t)V*sizeof(float)),z[248046]);
    llama_batch_free(y);llama_free(mtp);llama_model_free(m);llama_backend_free();return 0;
}
