#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>

static bool read_f32(const char * path, std::vector<float> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const auto bytes = in.tellg();
    if (bytes < 0 || bytes % (std::streamoff) sizeof(float) != 0) return false;
    out.resize((size_t) bytes / sizeof(float));
    in.seekg(0);
    return (bool) in.read(reinterpret_cast<char *>(out.data()), bytes);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s conventional.f32 staged.f32\n", argv[0]);
        return 2;
    }
    std::vector<float> reference, candidate;
    if (!read_f32(argv[1], reference) || !read_f32(argv[2], candidate) || reference.size() != candidate.size()) {
        std::fprintf(stderr, "oracle input error or shape mismatch\n");
        return 3;
    }
    double sum_sq = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float a = reference[i], b = candidate[i];
        if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; continue; }
        const double abs_err = std::abs((double) a - b);
        const double denom = std::max(std::abs((double) a), std::numeric_limits<double>::min());
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, abs_err / denom);
        sum_sq += abs_err * abs_err;
    }
    const double rms = reference.empty() ? 0.0 : std::sqrt(sum_sq / reference.size());
    std::printf("count=%zu shape=[%zu] stride_bytes=%zu max_abs=%.9g rms=%.9g max_rel=%.9g nonfinite=%zu\n",
            reference.size(), reference.size(), sizeof(float), max_abs, rms, max_rel, nonfinite);
    return nonfinite == 0 ? 0 : 4;
}
