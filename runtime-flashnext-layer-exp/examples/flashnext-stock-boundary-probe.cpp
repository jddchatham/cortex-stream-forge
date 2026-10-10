#include "../src/flashnext-layer-transaction.h"
#include "../src/flashnext-segment-b.h"
#include "../src/flashnext-stock-layer0-reference.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

static bool read_f32(const char * path, float * dst, size_t n) {
    std::ifstream f(path, std::ios::binary);
    return f.good() && bool(f.read(reinterpret_cast<char *>(dst), n*sizeof(float)));
}
static uint64_t hash_f32(const float * p, size_t n) {
    uint64_t h=1469598103934665603ull;
    for (size_t i=0;i<n;++i) { uint32_t u; std::memcpy(&u,p+i,4); h^=u; h*=1099511628211ull; }
    return h;
}
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    flashnext_layer_transaction tx;
    std::vector<float> prefix(12804), moe(2560);
    if (!read_f32(argv[1], prefix.data(), prefix.size()) || !read_f32(argv[2], moe.data(), moe.size())) return 3;
    std::memcpy(tx.pre_moe_hidden.data(), prefix.data(), 2560*sizeof(float));
    std::memcpy(tx.residual_hc.data(), prefix.data()+2560, 10240*sizeof(float));
    std::memcpy(tx.inject.data(), prefix.data()+12800, 4*sizeof(float));
    std::memcpy(tx.complete_moe.data(), moe.data(), 2560*sizeof(float));
    if (!flashnext_execute_segment_b(tx)) return 4;
    std::array<float,10240> stock{};
    if (!flashnext_stock_layer0_reference(tx, stock)) return 5;
    float max_abs=0, max_rel=0; size_t first=10240;
    for (size_t i=0;i<stock.size();++i) {
        const float d=std::fabs(stock[i]-tx.layer_output[i]);
        max_abs=std::max(max_abs,d); max_rel=std::max(max_rel,d/std::max(std::fabs(stock[i]),1e-30f));
        if (first==10240 && d!=0) first=i;
    }
    std::printf("STOCK_BOUNDARY=post-layer0-build_hc_combine shape=[2560,4,1]\n");
    std::printf("stock=%016llx cortex=%016llx max_abs=%.9g max_rel=%.9g first_diff=%s\n",
        (unsigned long long)hash_f32(stock.data(),stock.size()),
        (unsigned long long)hash_f32(tx.layer_output.data(),tx.layer_output.size()), max_abs,max_rel,
        first==10240?"NONE":"PRESENT");
    if (max_abs == 0) { std::puts("FLASHNEXT COMPLETE LAYER 0: PASS"); return 0; }
    return 6;
}
