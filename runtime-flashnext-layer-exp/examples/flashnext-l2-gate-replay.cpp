#include "flashnext-selected-moe.h"
#include "flashnext-tensor-index.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

template<typename T> static bool read_file(const char * path, std::vector<T> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end); const auto n = f.tellg(); f.seekg(0);
    if (n < 0 || (size_t)n % sizeof(T) != 0) return false;
    out.resize((size_t)n / sizeof(T)); f.read(reinterpret_cast<char *>(out.data()), n);
    return f.good() || f.eof();
}

int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    flashnext_tensor_index index; std::string error;
    if (!index.load(argv[1], &error)) { std::fprintf(stderr, "index=%s\n", error.c_str()); return 3; }
    std::vector<float> x, ref; std::vector<int32_t> ids; std::vector<float> got;
    if (!read_file(argv[2], x) || !read_file(argv[3], ids) || !read_file(argv[4], ref)) return 4;
    if (x.size() != 2560 * 9 || ids.size() != 10 * 9 || ref.size() != 640 * 10 * 9) return 5;
    got.resize(ref.size());
    if (!flashnext_execute_selected_gate_rows_layer(index, 2, x.data(), ids.data(), 9, got.data(), 16*1024)) return 6;
    double max_abs = 0, rms = 0, max_rel = 0; size_t mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = std::abs((double)got[i] - ref[i]);
        max_abs = std::max(max_abs, e); rms += e*e;
        max_rel = std::max(max_rel, e / std::max(std::abs((double)ref[i]), 1e-12));
        if (e > 1e-4) ++mismatches;
    }
    rms = std::sqrt(rms / got.size());
    std::printf("CORTEX_L2_GATE_REPLAY layer=2 tokens=9 activation_floats=%zu ids=%zu output_floats=%zu max_abs=%g rms=%g max_rel=%g mismatches=%zu status=%s\n",
        x.size(), ids.size(), got.size(), max_abs, rms, max_rel, mismatches, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0 ? 0 : 7;
}
