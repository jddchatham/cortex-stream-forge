#include "flashnext-tensor-index.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

static bool oracle_trace_target() {
    static const bool enabled = std::getenv("CORTEX_ORACLE_TRACE_TARGET") != nullptr;
    return enabled;
}
static uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static float fp16(uint16_t h) {
    const bool s=(h>>15)&1; const uint32_t e=(h>>10)&31; const uint32_t m=h&1023;
    float f;
    if (!e) f = std::ldexp(float(m), -24);
    else if (e == 31) f = m ? NAN : INFINITY;
    else f = std::ldexp(1.0f + float(m)/1024.0f, int(e)-15);
    return s ? -f : f;
}
static const int8_t lut[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

static void decode_nl(const uint8_t * p, float * y, size_t n) {
    for (size_t b=0; b<n/32; ++b) {
        const uint8_t * q=p+b*18; const float d=fp16(q[0]|uint16_t(q[1])<<8);
        for (int j=0;j<16;++j) { y[b*32+j]=d*lut[q[2+j]&15]; y[b*32+16+j]=d*lut[q[2+j]>>4]; }
    }
}
static void decode_xs(const uint8_t * p, float * y, size_t n) {
    for (size_t b=0; b<n/256; ++b) {
        const uint8_t * q=p+b*136; const float d=fp16(q[0]|uint16_t(q[1])<<8);
        const uint16_t sh=q[2]|uint16_t(q[3])<<8;
        for (int ib=0;ib<8;++ib) {
            const int ls=((q[4+ib/2]>>(4*(ib%2)))&15)|(((sh>>(2*ib))&3)<<4);
            const float dl=d*(ls-32); const uint8_t * z=q+8+ib*16;
            for (int j=0;j<16;++j) { y[b*256+ib*32+j]=dl*lut[z[j]&15]; y[b*256+ib*32+16+j]=dl*lut[z[j]>>4]; }
        }
    }
}
static int nearest(float v) { float z=v+12582912.0f; int i; std::memcpy(&i,&z,4); return (i&0x007fffff)-0x00400000; }
static uint16_t f32_to_f16(float v) {
    uint32_t u; std::memcpy(&u, &v, sizeof(u));
    const uint32_t s = (u >> 16) & 0x8000u;
    const int e = int((u >> 23) & 0xffu) - 127;
    const uint32_t m = u & 0x7fffffu;
    if (e > 15) return uint16_t(s | 0x7c00u);
    if (e < -14) {
        if (e < -24) return uint16_t(s);
        return uint16_t(s | ((m | 0x800000u) >> (-e - 14 + 13)));
    }
    return uint16_t(s | uint32_t(e + 15) << 10 | (m >> 13));
}
static uint16_t f32_to_f16_rne(float v) {
    uint32_t u; std::memcpy(&u, &v, sizeof(u));
    uint32_t s=(u>>16)&0x8000u; int e=int((u>>23)&0xffu)-127; uint32_t m=u&0x7fffffu;
    if(e>15) return uint16_t(s|0x7c00u);
    if(e < -14) return f32_to_f16(v);
    uint32_t h=m+0x1000u;
    if(h&0x800000u){ h=0; ++e; }
    if(e>15) return uint16_t(s|0x7c00u);
    return uint16_t(s|uint32_t(e+15)<<10|(h>>13));
}
// Independent scalar reproduction of quantize_row_q8_0_ref: one FP16 scale
// and 32 signed bytes per block.  This intentionally does not call GGML.
static void q8_0_roundtrip(const float * x, size_t n, std::vector<float> & y) {
    y.resize(n);
    for (size_t b = 0; b < n/32; ++b) {
        float amax = 0.0f;
        for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::abs(x[b*32+j]));
        const float d = amax / 127.0f;
        if (d == 0.0f) {
            std::fill(y.begin() + b*32, y.begin() + (b+1)*32, 0.0f);
            continue;
        }
        const float id = 1.0f/d;
        // q8_0 stores d as IEEE FP16; the kernel multiplies using that
        // rounded scale, not the original FP32 d.
        const float d16 = fp16(f32_to_f16_rne(d));
        for (size_t j = 0; j < 32; ++j) {
            const int q = std::max(-127, std::min(127, nearest(x[b*32+j]*id)));
            y[b*32+j] = d16 * float(q);
        }
    }
}
static void q8_0_encode(const float * x, size_t n, std::vector<uint8_t> & out) {
    out.resize((n/32)*34);
    for (size_t b=0; b<n/32; ++b) {
        float amax=0; for(size_t j=0;j<32;++j) amax=std::max(amax,std::abs(x[b*32+j]));
        float d=amax/127.0f; uint16_t dh=f32_to_f16_rne(d);
        std::memcpy(out.data()+b*34,&dh,2);
        const float id=d==0 ? 0.0f : 1.0f/d;
        for(size_t j=0;j<32;++j) { int q=d==0 ? 0 : std::max(-127,std::min(127,nearest(x[b*32+j]*id))); out[b*34+2+j]=uint8_t(int8_t(q)); }
    }
}
static void q8k_roundtrip(const float * x, size_t n, std::vector<float> & y) {
    y.resize(n);
    for(size_t b=0;b<n/256;++b){float mx=0;for(size_t j=0;j<256;++j)mx=std::max(mx,std::abs(x[b*256+j]));if(mx==0){std::fill(y.begin()+b*256,y.begin()+(b+1)*256,0);continue;}const float is=-127.0f/mx,d=1.0f/is;for(size_t j=0;j<256;++j)y[b*256+j]=d*std::max(-127,std::min(127,nearest(is*x[b*256+j])));}
}
static bool read_all(int fd, void * p, size_t n, uint64_t off) {
    size_t d=0; while(d<n) { ssize_t r=pread(fd,(uint8_t*)p+d,n-d,off+d); if(r<=0)return false; d+=size_t(r); } return true;
}
static bool read_row(const flashnext_tensor_descriptor & d, int row, size_t width, std::vector<uint8_t> & raw) {
    const size_t rb = d.type == 20 ? (width / 32) * 18 : (width / 256) * 136;
    raw.resize(rb); int fd=open(d.path.c_str(),O_RDONLY|O_CLOEXEC); if(fd<0)return false;
    bool ok=read_all(fd,raw.data(),rb,d.offset+uint64_t(row)*rb); close(fd); return ok && rb*8 >= width;
}
static bool load_rows(const flashnext_tensor_index & idx, int layer, flashnext_projection proj, const int32_t * ids,
                      size_t nt, size_t rows, size_t width, const float * x, std::vector<float> & out) {
    const size_t K=10; out.assign(nt*K*rows,0); std::vector<uint8_t> raw; std::vector<float> w(width);
    for(size_t t=0;t<nt;++t) for(size_t s=0;s<K;++s) {
        flashnext_tensor_descriptor d; if(!idx.resolve_expert(layer,ids[t*K+s],proj,d))return false;
        for(size_t r=0;r<rows;++r) {
            if(!read_row(d,int(r),width,raw))return false;
            if(proj==flashnext_projection::down) decode_nl(raw.data(),w.data(),width); else decode_xs(raw.data(),w.data(),width);
            const size_t xoff = proj == flashnext_projection::down ? (t*K+s)*width : t*width;
            std::vector<float> qx;
            if (proj == flashnext_projection::down) q8_0_roundtrip(x+xoff, width, qx);
            else q8k_roundtrip(x+xoff, width, qx);
            float z=0; for(size_t i=0;i<width;++i) z += w[i]*qx[i];
            if (oracle_trace_target() && proj == flashnext_projection::down && t == 1 && s == 9 && r == 0) {
                std::fprintf(stderr, "ORACLE_TARGET t=%zu slot=%zu expert=%d row=%zu desc_off=%llu row_bytes=%zu xoff=%zu x0=%g x1=%g xlast=%g q0=%g q1=%g qlast=%g raw_fnv=%016llx z=%.17g\n",
                    t, s, ids[t*K+s], r, (unsigned long long)d.offset, raw.size(), xoff,
                    x[xoff], x[xoff+1], x[xoff+width-1], qx[0], qx[1], qx[width-1],
                    (unsigned long long)fnv1a(raw.data(), raw.size()), double(z));
                std::vector<uint8_t> qb; q8_0_encode(x+xoff,width,qb);
                size_t mism=0, first=SIZE_MAX; std::vector<uint8_t> cpu;
                const char * cp="/tmp/cortex-iter162-p2-l4-t1-e257-q.bin"; FILE * f=std::fopen(cp,"rb");
                if(f){cpu.resize(qb.size()); size_t got=std::fread(cpu.data(),1,cpu.size(),f); std::fclose(f); if(got==cpu.size()) for(size_t i=0;i<qb.size();++i) if(qb[i]!=cpu[i]){++mism;if(first==SIZE_MAX)first=i;}}
                unsigned oh=qb.size()>=2?unsigned(qb[0])|(unsigned(qb[1])<<8):0, ch=cpu.size()>=2?unsigned(cpu[0])|(unsigned(cpu[1])<<8):0;
                std::fprintf(stderr,"ORACLE_Q8 target_bytes=%zu cpu_bytes=%zu mismatches=%zu first=%zu oracle_fnv=%016llx cpu_fnv=%016llx oracle_scale=0x%04x cpu_scale=0x%04x\n",qb.size(),cpu.size(),mism,first,(unsigned long long)fnv1a(qb.data(),qb.size()),(unsigned long long)(cpu.empty()?0:fnv1a(cpu.data(),cpu.size())),oh,ch);
            }
            out[(t*K+s)*rows+r]=z;
        }
    }
    return true;
}
template<class T> static bool read_vec(const char * p,std::vector<T>&v){std::ifstream f(p,std::ios::binary);if(!f)return false;f.seekg(0,std::ios::end);auto n=f.tellg();f.seekg(0);if(n<0||size_t(n)%sizeof(T))return false;v.resize(size_t(n)/sizeof(T));f.read((char*)v.data(),n);return f.good()||f.eof();}
static void compare(const char * name,const std::vector<float>&a,const std::vector<float>&b){double ma=0,ss=0;size_t mm=0,fi=0,first=SIZE_MAX;for(size_t i=0;i<a.size();++i){double e=std::abs(double(a[i])-b[i]);ma=std::max(ma,e);ss+=e*e;if(e>1e-4){mm++;if(first==SIZE_MAX)first=i;}if(!std::isfinite(a[i])||!std::isfinite(b[i]))fi++;}std::printf("CORTEX_L2_REF %s n=%zu max_abs=%g rms=%g mismatches=%zu first=%zu nonfinite=%zu\n",name,a.size(),ma,std::sqrt(ss/a.size()),mm,first,fi);}

int main(int argc,char **argv){
    if(argc!=7 && argc!=8)return 2; const int layer = argc == 8 ? std::atoi(argv[7]) : 2;
    flashnext_tensor_index idx; std::string err; if(!idx.load(argv[1],&err))return 3;
    std::vector<float>x,refup,refdown,out,gateprod,weights;std::vector<int32_t>ids; const char * gate_path = layer == 4 ? "/tmp/cortex-iter156-l4-gate-output.bin" : "/tmp/cortex-iter120-l2-gate-output.bin"; if(!read_vec(argv[2],x)||!read_vec(argv[3],ids)||!read_vec(argv[4],refup)||!read_vec(argv[5],refdown)||!read_vec(argv[6],out)||!read_vec(gate_path,gateprod))return 4;
    const size_t K=10,H=2560,F=640; const size_t nt=x.size()/H;
    if (nt == 0 || x.size()!=nt*H || ids.size()!=nt*K || refup.size()!=nt*K*F || refdown.size()!=nt*K*H || out.size()!=nt*H) return 5;
    std::vector<float> gate,up,act,down; if(!load_rows(idx,layer,flashnext_projection::gate,ids.data(),nt,K?F:F,H,x.data(),gate))return 6;
    if(!load_rows(idx,layer,flashnext_projection::up,ids.data(),nt,F,H,x.data(),up))return 7;
    act.resize(nt*K*F);for(size_t i=0;i<act.size();++i)act[i]=up[i]/(1+std::exp(-up[i])); // overwritten below after gate
    for(size_t i=0;i<act.size();++i)act[i]=gate[i]/(1+std::exp(-gate[i]))*up[i];
    if(!load_rows(idx,layer,flashnext_projection::down,ids.data(),nt,H,F,act.data(),down))return 8;
    auto dump = [](const char * path, const std::vector<float> & v) { FILE * f = std::fopen(path, "wb"); if (f) { std::fwrite(v.data(), sizeof(float), v.size(), f); std::fclose(f); } };
    dump("/tmp/cortex-iter160-oracle-gate.bin", gate);
    dump("/tmp/cortex-iter160-oracle-up.bin", up);
    dump("/tmp/cortex-iter160-oracle-down.bin", down);
    compare("gate",gate,gateprod);
    compare("up",up,refup);
    compare("down",down,refdown);
    std::vector<float> routed(nt*H); std::vector<float>w(nt*K); const char * weight_path = layer == 4 ? "/tmp/cortex-iter156-l4-weights.bin" : "/tmp/cortex-iter127-l2-weights.bin"; if(!read_vec(weight_path,w))return 9;
    for(size_t t=0;t<nt;++t)for(size_t f=0;f<H;++f){float z=0;for(size_t s=0;s<K;++s)z+=down[(t*K+s)*H+f]*w[t*K+s];routed[t*H+f]=z;}
    compare("aggregate",routed,out); return 0;
}
