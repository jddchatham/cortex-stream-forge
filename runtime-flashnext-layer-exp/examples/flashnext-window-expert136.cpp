#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <vector>
#include <unistd.h>

static bool pread_all(int fd, void * dst, size_t n, off_t off) {
    for (size_t done=0; done<n;) { ssize_t r=pread(fd,(uint8_t*)dst+done,n-done,off+(off_t)done); if(r<=0)return false; done+=(size_t)r; }
    return true;
}
static uint64_t hash_f32(const float * p,size_t n) { uint64_t h=1469598103934665603ull; for(size_t i=0;i<n;++i){uint32_t u;memcpy(&u,p+i,4);h^=u;h*=1099511628211ull;} return h; }

static bool tiled_rows(ggml_backend_t be, int fd, ggml_type type, int64_t K, int64_t N,
                       off_t offset, const float * input, size_t arena_bytes, float * output,
                       void * arena, size_t * reads, int64_t * min_rows) {
    const size_t row_bytes=ggml_row_size(type,K); const int64_t rows=(int64_t)(arena_bytes/row_bytes);
    if(rows<1)return false; *min_rows=rows;
    std::vector<float> in((size_t)K); std::memcpy(in.data(),input,(size_t)K*4);
    for(int64_t row=0;row<N;row+=rows) {
        const int64_t nr=std::min<int64_t>(rows,N-row); const size_t bytes=(size_t)nr*row_bytes;
        if(!pread_all(fd,arena,bytes,offset+(off_t)row*(off_t)row_bytes))return false; *reads+=bytes;
        ggml_init_params ip={32*1024*1024,nullptr,true}; ggml_context*c=ggml_init(ip); if(!c)return false;
        ggml_tensor*w=ggml_new_tensor_2d(c,type,K,nr),*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,K,1),*y=ggml_mul_mat(c,w,x); ggml_cgraph*g=ggml_new_graph(c);ggml_build_forward_expand(g,y);
        ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,be); if(!buf){ggml_free(c);return false;}
        w->data=arena;ggml_backend_tensor_set(x,in.data(),0,in.size()*4);bool ok=ggml_backend_graph_compute(be,g)==GGML_STATUS_SUCCESS;if(ok)ggml_backend_tensor_get(y,output+row,0,(size_t)nr*4);
        ggml_backend_buffer_free(buf);ggml_free(c); if(!ok)return false;
    }
    return true;
}

int main(int argc,char**argv) {
    // shard gate_off up_off down_off hidden.f32 arena_bytes
    if(argc!=11)return 2;int fd=open(argv[1],O_RDONLY|O_CLOEXEC);if(fd<0)return 3;uint64_t off[3];off[0]=strtoull(argv[2],0,0);off[1]=strtoull(argv[4],0,0);off[2]=strtoull(argv[6],0,0);size_t arena=strtoull(argv[9],0,0);
    std::vector<float>x(2560);FILE*f=fopen(argv[8],"rb");if(!f)return 4;if(fread(x.data(),4,x.size(),f)!=x.size())return 5;fclose(f);
    ggml_backend_load_all();ggml_backend_t be=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,0);if(!be)return 6;
    void * arena_mem=nullptr;if(posix_memalign(&arena_mem,64,arena)!=0)return 7;
    std::vector<float> gate(640),up(640),act(640),out(2560);size_t reads=0;int64_t rg=0,ru=0,rd=0;
    bool ok=tiled_rows(be,fd,(ggml_type)23,2560,640,off[0],x.data(),arena,gate.data(),arena_mem,&reads,&rg)&&tiled_rows(be,fd,(ggml_type)23,2560,640,off[1],x.data(),arena,up.data(),arena_mem,&reads,&ru);
    if(ok){for(size_t i=0;i<act.size();++i){float s=gate[i]/(1.0f+std::exp(-gate[i]));act[i]=s*up[i];}}
    ok=ok&&tiled_rows(be,fd,(ggml_type)20,640,2560,off[2],act.data(),arena,out.data(),arena_mem,&reads,&rd);close(fd);free(arena_mem);
    if(ok){FILE*of=fopen(argv[10],"wb");if(of){fwrite(out.data(),4,out.size(),of);fclose(of);}}
    std::printf("FLASHNEXT_WINDOW_EXPERT136 status=%s arena=%zu gate_rows=%lld up_rows=%lld down_rows=%lld reads=%zu hash=%016llx\n",ok?"PASS":"FAIL",arena,(long long)rg,(long long)ru,(long long)rd,reads,(unsigned long long)(ok?hash_f32(out.data(),out.size()):0));
    ggml_backend_free(be);return ok?0:7;
}
