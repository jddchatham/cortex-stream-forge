#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <vector>
#include <unistd.h>

static bool read_exact(int fd, void * p, size_t n, off_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t got = pread(fd, (uint8_t *) p + done, n - done, off + (off_t) done);
        if (got <= 0) return false;
        done += (size_t) got;
    }
    return true;
}
static uint64_t hash_f32(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { uint32_t u; std::memcpy(&u, p+i, 4); h ^= u; h *= 1099511628211ull; }
    return h;
}
int main(int argc, char ** argv) {
    // shard expert_id gate_off gate_bytes up_off up_bytes down_off down_bytes hidden.f32 output.f32
    if (argc != 11) return 2;
    const int expert_id = atoi(argv[2]);
    const char * shard = argv[1];
    uint64_t off[3], bytes[3];
    for (int i=0;i<3;++i) { off[i]=strtoull(argv[3+i*2],nullptr,0); bytes[i]=strtoull(argv[4+i*2],nullptr,0); }
    FILE * hf=fopen(argv[9],"rb"); if(!hf) return 3;
    std::vector<float> hidden(2560); if(fread(hidden.data(),4,hidden.size(),hf)!=hidden.size()) return 4; fclose(hf);
    int fd=open(shard,O_RDONLY|O_CLOEXEC); if(fd<0) return 5;
    std::vector<uint8_t> gate(bytes[0]), up(bytes[1]), down(bytes[2]);
    for(int i=0;i<3;++i) { std::vector<uint8_t> * dst = i==0?&gate:i==1?&up:&down; if(!read_exact(fd,dst->data(),dst->size(),off[i])) return 6; }
    close(fd);
    ggml_backend_load_all();
    ggml_backend_t backend=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr); if(!backend) return 7;
    ggml_init_params ip={128*1024*1024,nullptr,true}; ggml_context * ctx=ggml_init(ip); if(!ctx) return 8;
    // Qwen4Exp layer-0 expert slices: gate/up [640,2560], down [2560,640].
    ggml_tensor * wg=ggml_new_tensor_2d(ctx,(ggml_type)23,2560,640);
    ggml_tensor * wu=ggml_new_tensor_2d(ctx,(ggml_type)23,2560,640);
    ggml_tensor * wd=ggml_new_tensor_2d(ctx,(ggml_type)20,640,2560);
    ggml_tensor * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2560,1);
    fprintf(stderr, "EXPERT_SHAPES wg=[%lld,%lld] wu=[%lld,%lld] wd=[%lld,%lld] x=[%lld,%lld]\n",
        (long long)wg->ne[0],(long long)wg->ne[1],(long long)wu->ne[0],(long long)wu->ne[1],
        (long long)wd->ne[0],(long long)wd->ne[1],(long long)x->ne[0],(long long)x->ne[1]);
    ggml_tensor * g=ggml_mul_mat(ctx,wg,x), * u=ggml_mul_mat(ctx,wu,x);
    g=ggml_silu(ctx,g); ggml_tensor * gu=ggml_mul(ctx,g,u); ggml_tensor * y=ggml_mul_mat(ctx,wd,gu);
    ggml_cgraph * graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,y);
    ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(ctx,backend); if(!buf) return 9;
    wg->data=gate.data(); wu->data=up.data(); wd->data=down.data();
    ggml_backend_tensor_set(x,hidden.data(),0,hidden.size()*sizeof(float));
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 10;
    std::vector<float> out(2560); ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float));
    FILE * of=fopen(argv[10],"wb"); if(of){fwrite(out.data(),4,out.size(),of);fclose(of);}
    double norm=0; float mn=out[0],mx=out[0]; for(float v:out){mn=std::min(mn,v);mx=std::max(mx,v);norm+=(double)v*v;}
    printf("FLASHNEXT_SINGLE_EXPERT id=%d hidden_hash=%016llx output_hash=%016llx min=%.9g max=%.9g l2=%.9g bytes=%llu\n",
        expert_id,
        (unsigned long long)hash_f32(hidden.data(),hidden.size()),(unsigned long long)hash_f32(out.data(),out.size()),mn,mx,std::sqrt(norm),(unsigned long long)(bytes[0]+bytes[1]+bytes[2]));
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend); return 0;
}
