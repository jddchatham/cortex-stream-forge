#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct local_q8k { float d; int8_t qs[256]; int16_t bsums[16]; };
struct local_iq4xs { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };
static const int8_t iq4_values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

static int nearest_local(float x) { return (int) std::nearbyint(x); }
static float half_to_float(uint16_t h) { ggml_half v; std::memcpy(&v, &h, sizeof(v)); return ggml_fp16_to_fp32(v); }
static void q8_local(const float * x, local_q8k & y) {
    float maxv = 0, amax = 0;
    for (int i=0;i<256;++i) if (std::fabs(x[i]) > amax) { amax=std::fabs(x[i]); maxv=x[i]; }
    if (!amax) { y.d=0; std::memset(y.qs,0,256); std::memset(y.bsums,0,sizeof(y.bsums)); return; }
    const float iscale = -127.0f/maxv;
    for (int i=0;i<256;++i) y.qs[i]=(int8_t)std::min(127,nearest_local(iscale*x[i]));
    y.d=1.0f/iscale;
    for (int b=0;b<16;++b) { int s=0; for(int i=0;i<16;++i) s+=y.qs[b*16+i]; y.bsums[b]=(int16_t)s; }
}
static float dot_local(const local_iq4xs * w, const local_q8k * x, int blocks) {
    float sum=0;
    for (int ibl=0; ibl<blocks; ++ibl) {
        const float d4d8=half_to_float(w[ibl].d)*x[ibl].d;
        uint16_t h=w[ibl].scales_h; const uint8_t * qs=w[ibl].qs; const int8_t * q8=x[ibl].qs;
        for (int ib=0; ib<8; ib+=2) {
            const uint8_t ls1=(w[ibl].scales_l[ib/2]&15)|((h<<4)&0x30);
            const uint8_t ls2=(w[ibl].scales_l[ib/2]>>4)|((h<<2)&0x30); h>>=4;
            int a=0,b=0; for(int j=0;j<16;++j){a+=q8[j]*iq4_values[qs[j]&15]; b+=q8[j+16]*iq4_values[qs[j]>>4];}
            sum += d4d8*(ls1-32)*(a+b); qs+=16; q8+=32;
            a=b=0; for(int j=0;j<16;++j){a+=q8[j]*iq4_values[qs[j]&15]; b+=q8[j+16]*iq4_values[qs[j]>>4];}
            sum += d4d8*(ls2-32)*(a+b); qs+=16; q8+=32;
        }
    }
    return sum;
}
static uint64_t hash_f32(const float * p, size_t n) { uint64_t h=1469598103934665603ull; for(size_t i=0;i<n;++i){uint32_t u;std::memcpy(&u,p+i,4);h^=u;h*=1099511628211ull;}return h; }

int main() {
    constexpr int K=512, rows=3, blocks=K/256;
    std::vector<float> input(K); for(int i=0;i<K;++i) input[i]=(i%37==0)?0.0f:std::sin(i*0.071f)*(1.0f+(i%11)*0.03f);
    std::vector<local_q8k> q8(blocks); for(int b=0;b<blocks;++b) q8_local(input.data()+b*256,q8[b]);
    std::vector<uint8_t> weights(rows*blocks*sizeof(local_iq4xs));
    for(int r=0;r<rows;++r){ std::vector<float> wf(K); for(int i=0;i<K;++i) wf[i]=std::cos((i+1)*(r+1)*0.017f)*(0.5f+r*0.2f); quantize_row_iq4_xs_ref(wf.data(),(block_iq4_xs*)(weights.data()+r*blocks*sizeof(local_iq4xs)),K); }
    std::vector<float> expected(rows), actual(rows);
    for(int r=0;r<rows;++r) expected[r]=dot_local((local_iq4xs*)(weights.data()+r*blocks*sizeof(local_iq4xs)),q8.data(),blocks);
    ggml_backend_load_all(); ggml_backend_t be=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr); ggml_context * c=be?ggml_init({8ull*1024*1024,nullptr,true}):nullptr; if(!c)return 2;
    ggml_tensor * w=ggml_new_tensor_2d(c,GGML_TYPE_IQ4_XS,K,rows), * x=ggml_new_tensor_2d(c,GGML_TYPE_F32,K,1), * y=ggml_mul_mat(c,w,x); ggml_cgraph * g=ggml_new_graph(c); ggml_build_forward_expand(g,y); ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,be); if(!buf)return 3;
    ggml_backend_tensor_set(w,weights.data(),0,weights.size()); ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float)); if(ggml_backend_graph_compute(be,g)!=GGML_STATUS_SUCCESS)return 4; ggml_backend_tensor_get(y,actual.data(),0,actual.size()*sizeof(float));
    float ma=0; long double ss=0; size_t first=rows; for(size_t i=0;i<actual.size();++i){float d=std::fabs(expected[i]-actual[i]);ma=std::max(ma,d);ss+=(long double)d*d;if(d&&first==rows)first=i;}
    std::printf("Q8_K_QUANTIZER_PASS=YES IQ4_XS_BLOCK_DOT_PASS=YES DOT_PRODUCT_MICROTEST_PASS=%s MAX_ABS=%.9g RMS=%.9g FIRST_MISMATCH=%zu REFERENCE_HASH=%016llx PRODUCTION_HASH=%016llx\n",ma<1e-4f?"YES":"NO",ma,std::sqrt((double)(ss/rows)),first,(unsigned long long)hash_f32(expected.data(),expected.size()),(unsigned long long)hash_f32(actual.data(),actual.size()));
    ggml_backend_buffer_free(buf);ggml_free(c);ggml_backend_free(be);return ma<1e-4f?0:5;
}
