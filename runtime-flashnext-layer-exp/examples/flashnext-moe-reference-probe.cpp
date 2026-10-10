#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_CAPTURE_MOE", "1", 1);
    unsetenv("CORTEX_FLASHNEXT_ROUTER_ONLY");
    llama_backend_init();
    llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    llama_model * model=llama_model_load_from_file(argv[1],mp); if(!model) return 3;
    llama_context_params cp=llama_context_default_params(); cp.n_ctx=256; cp.n_batch=1;
    llama_context * ctx=llama_init_from_model(model,cp); if(!ctx) return 4;
    llama_batch b=llama_batch_init(1,0,1); b.token[0]=248045;b.pos[0]=0;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;b.n_tokens=1;
    if(llama_decode(ctx,b)!=0)return 5; const float * p=llama_get_logits_ith(ctx,0); if(!p)return 6;
    FILE * f=fopen(argv[2],"wb"); if(!f)return 7; fwrite(p,4,2560,f);fclose(f);
    uint64_t h=1469598103934665603ull; for(int i=0;i<2560;++i){uint32_t u;memcpy(&u,p+i,4);h^=u;h*=1099511628211ull;}
    printf("FLASHNEXT_REFERENCE_MOE width=2560 hash=%016llx output=%s\n",(unsigned long long)h,argv[2]);
    llama_batch_free(b);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
