#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <vector>
#include <unistd.h>

static bool rd(int fd, void * p, size_t n, off_t o) {
    for (size_t d=0; d<n;) { ssize_t r=pread(fd,(uint8_t*)p+d,n-d,o+(off_t)d); if(r<=0)return false; d+=(size_t)r; }
    return true;
}
static uint64_t h(const float * p,size_t n) { uint64_t x=1469598103934665603ull; for(size_t i=0;i<n;++i){uint32_t u;memcpy(&u,p+i,4);x^=u;x*=1099511628211ull;}return x; }
static bool dump(ggml_backend_t be, ggml_tensor * t, const char * path, uint64_t * hash) {
    std::vector<float> v((size_t)ggml_nelements(t)); ggml_backend_tensor_get(t,v.data(),0,v.size()*4);
    FILE * f=fopen(path,"wb"); if(!f)return false; fwrite(v.data(),4,v.size(),f);fclose(f);*hash=h(v.data(),v.size());return true;
}
int main(int argc,char **argv) {
    // shard gate_off gate_bytes up_off up_bytes down_off down_bytes hidden output_prefix
    if(argc!=10)return 2; int fd=open(argv[1],O_RDONLY|O_CLOEXEC);if(fd<0)return 3;
    uint64_t off[3],sz[3];for(int i=0;i<3;++i){off[i]=strtoull(argv[2+i*2],0,0);sz[i]=strtoull(argv[3+i*2],0,0);}
    std::vector<uint8_t> wg0(sz[0]),wu0(sz[1]),wd0(sz[2]);if(!rd(fd,wg0.data(),sz[0],off[0])||!rd(fd,wu0.data(),sz[1],off[1])||!rd(fd,wd0.data(),sz[2],off[2]))return 4;close(fd);
    std::vector<float>x(2560);FILE * xf=fopen(argv[8],"rb");if(!xf||fread(x.data(),4,x.size(),xf)!=x.size())return 5;fclose(xf);
    ggml_backend_load_all();ggml_backend_t be=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,0);if(!be)return 6;
    ggml_init_params ip={128*1024*1024,0,true};ggml_context * c=ggml_init(ip);if(!c)return 7;
    ggml_tensor * wg=ggml_new_tensor_2d(c,(ggml_type)23,2560,640);ggml_tensor * wu=ggml_new_tensor_2d(c,(ggml_type)23,2560,640);ggml_tensor * wd=ggml_new_tensor_2d(c,(ggml_type)20,640,2560);ggml_tensor * in=ggml_new_tensor_2d(c,GGML_TYPE_F32,2560,1);
    ggml_tensor * gate=ggml_mul_mat(c,wg,in);ggml_tensor * up=ggml_mul_mat(c,wu,in);ggml_tensor * silu=ggml_silu(c,gate);ggml_tensor * act=ggml_mul(c,silu,up);ggml_tensor * down=ggml_mul_mat(c,wd,act);
    ggml_cgraph * g=ggml_new_graph(c);ggml_build_forward_expand(g,down);ggml_backend_buffer_t b=ggml_backend_alloc_ctx_tensors(c,be);if(!b)return 8;wg->data=wg0.data();wu->data=wu0.data();wd->data=wd0.data();ggml_backend_tensor_set(in,x.data(),0,x.size()*4);if(ggml_backend_graph_compute(be,g)!=GGML_STATUS_SUCCESS)return 9;
    uint64_t hh[4];char p[512];for(int i=0;i<4;++i){snprintf(p,sizeof(p),"%s.%s.f32",argv[9],i==0?"gate":i==1?"up":i==2?"silu":"down");ggml_tensor * t=i==0?gate:i==1?up:i==2?silu:down;if(!dump(be,t,p,&hh[i]))return 10;}
    printf("FLASHNEXT_REFERENCE_EXPERT136 gate=%016llx up=%016llx silu=%016llx down=%016llx\n",(unsigned long long)hh[0],(unsigned long long)hh[1],(unsigned long long)hh[2],(unsigned long long)hh[3]);
    ggml_backend_buffer_free(b);ggml_free(c);ggml_backend_free(be);return 0;
}
