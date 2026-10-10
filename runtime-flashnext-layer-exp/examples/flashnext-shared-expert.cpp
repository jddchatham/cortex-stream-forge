#include "ggml-backend.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <vector>
#include <unistd.h>
static bool rd(int f,void*p,size_t n,off_t o){for(size_t d=0;d<n;){ssize_t r=pread(f,(uint8_t*)p+d,n-d,o+d);if(r<=0)return false;d+=r;}return true;}
static uint64_t hh(const float*p,size_t n){uint64_t h=1469598103934665603ull;for(size_t i=0;i<n;++i){uint32_t u;memcpy(&u,p+i,4);h^=u;h*=1099511628211ull;}return h;}
int main(int ac,char**av){
 // shard gate_off up_off down_off scalar_gate_off hidden output
 if(ac!=8)return 2;int f=open(av[1],O_RDONLY|O_CLOEXEC);if(f<0)return 3;uint64_t o[4];for(int i=0;i<4;++i)o[i]=strtoull(av[2+i],0,0);
 std::vector<uint8_t> gate(870400),up(870400),down(921600);std::vector<float> sg(2560),x(2560);if(!rd(f,gate.data(),gate.size(),o[0])||!rd(f,up.data(),up.size(),o[1])||!rd(f,down.data(),down.size(),o[2])||!rd(f,sg.data(),sg.size()*4,o[3]))return 4;close(f);FILE*h=fopen(av[6],"rb");if(!h)return 5;if(fread(x.data(),4,x.size(),h)!=x.size())return 6;fclose(h);
 ggml_backend_load_all();ggml_backend_t be=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,0);ggml_init_params ip={64*1024*1024,0,true};ggml_context*c=ggml_init(ip);
 ggml_tensor*wg=ggml_new_tensor_2d(c,(ggml_type)23,2560,640),*wu=ggml_new_tensor_2d(c,(ggml_type)23,2560,640),*wd=ggml_new_tensor_2d(c,(ggml_type)20,640,2560),*wsg=ggml_new_tensor_2d(c,GGML_TYPE_F32,2560,1),*in=ggml_new_tensor_2d(c,GGML_TYPE_F32,2560,1);
 ggml_tensor*g=ggml_mul_mat(c,wg,in),*u=ggml_mul_mat(c,wu,in),*a=ggml_mul(c,ggml_silu(c,g),u),*y=ggml_mul_mat(c,wd,a),*scalar=ggml_sigmoid(c,ggml_mul_mat(c,wsg,in)),*out=ggml_mul(c,y,scalar);ggml_cgraph*gr=ggml_new_graph(c);ggml_build_forward_expand(gr,out);ggml_backend_buffer_t b=ggml_backend_alloc_ctx_tensors(c,be);wg->data=gate.data();wu->data=up.data();wd->data=down.data();wsg->data=sg.data();ggml_backend_tensor_set(in,x.data(),0,x.size()*4);if(ggml_backend_graph_compute(be,gr)!=GGML_STATUS_SUCCESS)return 7;std::vector<float>v(2560);ggml_backend_tensor_get(out,v.data(),0,v.size()*4);FILE*ofo=fopen(av[7],"wb");if(ofo){fwrite(v.data(),4,v.size(),ofo);fclose(ofo);}printf("FLASHNEXT_SHARED_EXPERT hash=%016llx\n",(unsigned long long)hh(v.data(),v.size()));ggml_backend_buffer_free(b);ggml_free(c);ggml_backend_free(be);return 0;}
