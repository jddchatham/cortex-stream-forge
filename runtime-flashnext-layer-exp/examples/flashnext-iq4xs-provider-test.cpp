#include "flashnext-selected-moe.h"
#include "flashnext-tensor-index.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <array>
#include <tuple>
#include <cstring>

static bool read_exact(int fd, void * p, size_t n, uint64_t off) {
    size_t done = 0; while (done < n) { ssize_t r = pread(fd, (char *)p + done, n-done, off+done); if (r <= 0) return false; done += (size_t)r; } return true;
}

static bool descriptor_row(const flashnext_tensor_descriptor & d, int row, std::vector<uint8_t> & b, std::vector<float> & out) {
    int fd = open(d.path.c_str(), O_RDONLY|O_CLOEXEC); if (fd < 0) return false;
    const size_t rb = d.row_bytes; b.resize(rb); const bool ok = read_exact(fd, b.data(), rb, d.offset + (uint64_t)row * rb); close(fd); if (!ok) return false;
    out.resize(d.shape0); if (d.type == (int)GGML_TYPE_IQ4_XS) dequantize_row_iq4_xs((const block_iq4_xs *)b.data(), out.data(), d.shape0);
    else if (d.type == (int)GGML_TYPE_IQ4_NL) dequantize_row_iq4_nl((const block_iq4_nl *)b.data(), out.data(), d.shape0); else return false;
    return true;
}

static int nearest_ref(float v) {
    float z = v + 12582912.0f; int i; std::memcpy(&i, &z, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static uint16_t f32_to_f16_ref(float v) {
    uint32_t u; std::memcpy(&u, &v, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int e = int((u >> 23) & 0xffu) - 127;
    const uint32_t mant = u & 0x7fffffu;
    if (e > 15) return uint16_t(sign | 0x7c00u);
    if (e < -14) return e < -24 ? uint16_t(sign) : uint16_t(sign | ((mant | 0x800000u) >> (-e - 14 + 13)));
    return uint16_t(sign | uint32_t(e + 15) << 10 | (mant >> 13));
}

static float f16_to_f32_ref(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 31u, mant = h & 1023u;
    if (!exp) return std::ldexp(float(mant), -24) * ((sign ? -1.0f : 1.0f));
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    return (sign ? -1.0f : 1.0f) * std::ldexp(1.0f + float(mant) / 1024.0f, int(exp) - 15);
}

// Independent scalar reproduction of GGML's Q8_K/Q8_0 activation encoding.
static void quantize_q8k_ref(const float * x, size_t n, std::vector<float> & y) {
    y.resize(n);
    for (size_t b = 0; b < n/256; ++b) {
        float amax = 0.0f, max = 0.0f;
        for (size_t j = 0; j < 256; ++j) if (std::abs(x[b*256+j]) > amax) { amax = std::abs(x[b*256+j]); max = x[b*256+j]; }
        if (!amax) { std::fill(y.begin()+b*256, y.begin()+(b+1)*256, 0.0f); continue; }
        const float iscale = float(-127.0f / max), d = float(1.0f / iscale);
        for (size_t j = 0; j < 256; ++j) y[b*256+j] = d * float(std::max(-127, std::min(127, nearest_ref(float(iscale*x[b*256+j])))));
    }
}

static void quantize_q80_ref(const float * x, size_t n, std::vector<float> & y) {
    y.resize(n);
    for (size_t b = 0; b < n/32; ++b) {
        float amax = 0.0f; for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::abs(x[b*32+j]));
        const float d = float(amax / 127.0f), d16 = f16_to_f32_ref(f32_to_f16_ref(d));
        for (size_t j = 0; j < 32; ++j) y[b*32+j] = d16 * float(std::max(-127, std::min(127, nearest_ref(d ? float(x[b*32+j] / d) : 0.0f))));
    }
}

static bool reference(const flashnext_tensor_index & idx, int layer, const float * x, const int32_t * ids, const double * weights, size_t nt, std::vector<float> & out) {
    constexpr int H=2560,F=640,K=10; out.assign(nt*H, 0.0f); std::vector<float> g,u,a,y; std::vector<uint8_t> raw;
    for (size_t t=0;t<nt;++t) for (int s=0;s<K;++s) {
        flashnext_tensor_descriptor gd,ud,dd; if (!idx.resolve_expert(layer,ids[t*K+s],flashnext_projection::gate,gd) || !idx.resolve_expert(layer,ids[t*K+s],flashnext_projection::up,ud) || !idx.resolve_expert(layer,ids[t*K+s],flashnext_projection::down,dd)) return false;
        g.assign(F,0); u.assign(F,0); a.assign(F,0); y.assign(H,0);
        std::vector<float> qx;
        quantize_q8k_ref(x + t*H, H, qx);
        for (int j=0;j<F;++j) { std::vector<float> rg,ru; if (!descriptor_row(gd,j,raw,rg) || !descriptor_row(ud,j,raw,ru)) return false; for(int i=0;i<H;++i){g[j]+=rg[i]*qx[i];u[j]+=ru[i]*qx[i];} a[j]=g[j]/(1+std::exp(-g[j]))*u[j]; }
        quantize_q80_ref(a.data(), F, qx);
        for (int i=0;i<H;++i) { std::vector<float> rd; if (!descriptor_row(dd,i,raw,rd)) return false; for(int j=0;j<F;++j)y[i]+=rd[j]*qx[j]; out[t*H+i]+=float(weights[t*K+s])*y[i]; }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) return 2; const int layer = argc == 3 ? std::atoi(argv[2]) : 0; flashnext_tensor_index idx; std::string err; if (!idx.load(argv[1], &err)) { fprintf(stderr,"index: %s\n",err.c_str()); return 3; }
    flashnext_tensor_descriptor g,u,d; if (!idx.resolve_expert(layer,0,flashnext_projection::gate,g,&err) || !idx.resolve_expert(layer,0,flashnext_projection::up,u,&err) || !idx.resolve_expert(layer,0,flashnext_projection::down,d,&err)) return 4;
    if (g.type != (int)GGML_TYPE_IQ4_XS || u.type != (int)GGML_TYPE_IQ4_XS || d.type != (int)GGML_TYPE_IQ4_NL) return 5;
    for (size_t nt : {size_t(1),size_t(3)}) {
        constexpr size_t H=2560,K=10; std::vector<float> x(nt*H), got(nt*H), ref; for(size_t i=0;i<x.size();++i)x[i]=0.01f*float(1+i%31);
        std::vector<int32_t> ids(nt*K); std::vector<double> w(nt*K); for(size_t t=0;t<nt;++t)for(size_t s=0;s<K;++s){ids[t*K+s]=(s%2)?3:0;w[t*K+s]=0.1;}
        if (!flashnext_execute_selected_moe_rows_layer(idx, layer, x.data(), ids.data(), w.data(), nt, got.data(), 16*1024) || !reference(idx,layer,x.data(),ids.data(),w.data(),nt,ref)) return 6;
        double ma=0, mr=0, rms=0; std::array<std::tuple<double,size_t,float,float>,10> top{};
        for(size_t i=0;i<got.size();++i){double e=std::abs(double(got[i])-ref[i]);ma=std::max(ma,e);mr=std::max(mr,e/(std::abs(ref[i])+1e-12));rms+=e*e;
            auto item=std::make_tuple(e,i,got[i],ref[i]);
            for(size_t k=0;k<top.size();++k) if(e>std::get<0>(top[k])) { for(size_t q=top.size()-1;q>k;--q) top[q]=top[q-1]; top[k]=item; break; }
        } rms=std::sqrt(rms/got.size());
        printf("IQ4XS_PROVIDER tokens=%zu max_abs=%g max_rel=%g rms=%g bytes=%llu\n",nt,ma,mr,rms,(unsigned long long)(nt*K*(g.bytes/512*2+d.bytes/512)));
        for(size_t k=0;k<top.size() && std::get<0>(top[k])>0;++k) printf("IQ4XS_PROVIDER_TOP rank=%zu index=%zu got=%.9g ref=%.9g abs=%.9g rel=%.9g\n",k+1,std::get<1>(top[k]),std::get<2>(top[k]),std::get<3>(top[k]),std::get<0>(top[k]),std::get<0>(top[k])/(std::abs(std::get<3>(top[k]))+1e-12));
        // The scalar reference intentionally decodes weights and accumulates
        // independently; use absolute/RMS bounds for the full aggregate.
        // Relative error is reported, but is ill-conditioned near zero.
        if (ma > 1e-5 || rms > 2e-6 || !std::isfinite(ma)) return 7;
    }
    std::vector<float> x(2560), out(2560); std::vector<int32_t> bad(10,512); std::vector<double>w(10,0.1); if (flashnext_execute_selected_moe_rows_layer(idx,layer,x.data(),bad.data(),w.data(),1,out.data(),16*1024)) return 8;
    puts("IQ4XS_PROVIDER PASS"); return 0;
}
