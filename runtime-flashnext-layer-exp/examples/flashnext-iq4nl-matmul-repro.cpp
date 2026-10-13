#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    constexpr int64_t K=320, N=10240;
    if (argc != 3) return 2;
    const size_t wb=ggml_row_size(GGML_TYPE_IQ4_NL,K);
    std::vector<uint8_t> w((size_t)N*wb); std::vector<float> x(K);
    std::ifstream wf(argv[1],std::ios::binary), xf(argv[2],std::ios::binary);
    wf.read((char *)w.data(),w.size()); xf.read((char *)x.data(),x.size()*sizeof(float));
    if (!wf || !xf) return 3;
    ggml_init_params p{.mem_size=32u*1024u*1024u,.mem_buffer=nullptr,.no_alloc=true}; ggml_context * ctx=ggml_init(p); if(!ctx)return 4;
    ggml_tensor * tw=ggml_new_tensor_2d(ctx,GGML_TYPE_IQ4_NL,K,N), * tx=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,1), * ty=ggml_mul_mat(ctx,tw,tx);
    ggml_cgraph * g=ggml_new_graph(ctx);ggml_build_forward_expand(g,ty);ggml_backend_t b=ggml_backend_cpu_init();ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(ctx,b);if(!buf)return 5;
    ggml_backend_tensor_set(tw,w.data(),0,w.size());ggml_backend_tensor_set(tx,x.data(),0,x.size()*sizeof(float));if(ggml_backend_graph_compute(b,g)!=GGML_STATUS_SUCCESS)return 6;ggml_backend_synchronize(b);
    std::vector<float> y(N);ggml_backend_tensor_get(ty,y.data(),0,y.size()*sizeof(float));
    std::vector<float> dw(K),q8x(K),ys(N);std::vector<uint8_t> q8(ggml_row_size(GGML_TYPE_Q8_0,K));quantize_row_q8_0_ref(x.data(),(block_q8_0 *)q8.data(),K);dequantize_row_q8_0((const block_q8_0 *)q8.data(),q8x.data(),K);
    double ma=0,qa=0,ss=0,qss=0;size_t over=0,qover=0;for(int64_t i=0;i<N;++i){dequantize_row_iq4_nl((const block_iq4_nl *)(w.data()+i*wb),dw.data(),K);double a=0,c=0;for(int j=0;j<K;++j){a+=(double)dw[j]*x[j];c+=(double)dw[j]*q8x[j];}ys[i]=(float)a;double e=std::fabs(y[i]-a),eq=std::fabs(y[i]-c);ma=std::max(ma,e);qa=std::max(qa,eq);ss+=e*e;qss+=eq*eq;if(e>1e-4)++over;if(eq>1e-4)++qover;}
    for(int i:{0,1,2,5119,10239})std::printf("IQ4NL channel=%d ggml=%g scalar_f32=%g scalar_q8=%g\n",i,y[i],ys[i],y[i]);
    std::printf("IQ4NL result rows=%lld row_bytes=%zu max_abs_f32=%g rms_f32=%g over_f32=%zu max_abs_q8=%g rms_q8=%g over_q8=%zu\n",(long long)N,wb,ma,std::sqrt(ss/N),over,qa,std::sqrt(qss/N),qover);
    ggml_backend_buffer_free(buf);ggml_backend_free(b);ggml_free(ctx);return 0;
}
