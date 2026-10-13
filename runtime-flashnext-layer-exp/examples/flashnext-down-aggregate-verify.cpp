#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>

#include "ggml.h"
#include "ggml-cpu.h"
#include "quants.h"

extern "C" void ggml_cpu_init(void);

int main() {
    ggml_cpu_init();

    // 1. Verify Layer 4 Down-Projection equivalence with hardware AVX2 dot kernel
    std::vector<float> down_inp(57600);
    FILE * f = fopen("/tmp/cortex-iter168-l4-down-input.bin", "rb");
    if (!f) { fprintf(stderr, "Missing /tmp/cortex-iter168-l4-down-input.bin\n"); return 1; }
    if (fread(down_inp.data(), sizeof(float), 57600, f) != 57600) { fclose(f); return 1; }
    fclose(f);

    std::vector<float> rdown(230400);
    f = fopen("/tmp/cortex-iter156-l4-down-output.bin", "rb");
    if (!f) { fprintf(stderr, "Missing /tmp/cortex-iter156-l4-down-output.bin\n"); return 1; }
    if (fread(rdown.data(), sizeof(float), 230400, f) != 230400) { fclose(f); return 1; }
    fclose(f);

    std::vector<int32_t> ids(90);
    f = fopen("/tmp/cortex-iter156-l4-gate-ids.bin", "rb");
    if (!f) { fprintf(stderr, "Missing /tmp/cortex-iter156-l4-gate-ids.bin\n"); return 1; }
    if (fread(ids.data(), sizeof(int32_t), 90, f) != 90) { fclose(f); return 1; }
    fclose(f);

    const char * shard = "/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf";
    uint64_t base_offset = 35366368544ULL;
    uint64_t stride = 921600ULL;
    FILE * fs = fopen(shard, "rb");
    if (!fs) { fprintf(stderr, "Missing model shard: %s\n", shard); return 1; }

    size_t bit_exact = 0;
    size_t float_match = 0;
    size_t mismatches = 0;
    double max_diff = 0;
    double sum_diff2 = 0;

    std::vector<uint8_t> expert_weights(stride);
    std::vector<uint8_t> q8(680);

    for (int t = 0; t < 9; ++t) {
        for (int s = 0; s < 10; ++s) {
            int expert = ids[t * 10 + s];
            const size_t xoff = (t * 10 + s) * 640;
            const float * x = down_inp.data() + xoff;
            quantize_row_q8_0(x, q8.data(), 640);

            uint64_t off = base_offset + uint64_t(expert) * stride;
            fseeko(fs, off, SEEK_SET);
            if (fread(expert_weights.data(), 1, stride, fs) != stride) {
                fclose(fs); return 1;
            }

            for (int r = 0; r < 2560; ++r) {
                float val = 0.0f;
                ggml_vec_dot_iq4_nl_q8_0(640, &val, 0, expert_weights.data() + r * 360, 0, q8.data(), 0, 1);

                size_t idx = (t * 10 + s) * 2560 + r;
                float r_val = rdown[idx];
                float diff = std::abs(val - r_val);

                if (diff == 0.0f) {
                    bit_exact++;
                } else if (diff < 1e-6f) {
                    float_match++;
                } else {
                    mismatches++;
                }
                if (diff > max_diff) max_diff = diff;
                sum_diff2 += diff * diff;
            }
        }
    }
    fclose(fs);

    printf("=== DOWN PROJECTION EQUIVALENCE (230,400 elements) ===\n");
    printf("Total elements:    230400\n");
    printf("Bit-exact matches: %zu (%.2f%%)\n", bit_exact, (bit_exact * 100.0) / 230400.0);
    printf("Mismatches (>1e-6):%zu\n", mismatches);
    printf("Max absolute diff: %.9e\n", max_diff);
    printf("RMS difference:    %.9e\n", std::sqrt(sum_diff2 / 230400.0));

    // 2. Verify Layer 4 MoE Aggregate Routing against runtime out.bin
    std::vector<float> rout(23040);
    f = fopen("/tmp/cortex-iter156-l4-out.bin", "rb");
    if (!f) { fprintf(stderr, "Missing /tmp/cortex-iter156-l4-out.bin\n"); return 1; }
    if (fread(rout.data(), sizeof(float), 23040, f) != 23040) { fclose(f); return 1; }
    fclose(f);

    std::vector<float> w(90);
    f = fopen("/tmp/cortex-iter156-l4-weights.bin", "rb");
    if (!f) { fprintf(stderr, "Missing /tmp/cortex-iter156-l4-weights.bin\n"); return 1; }
    if (fread(w.data(), sizeof(float), 90, f) != 90) { fclose(f); return 1; }
    fclose(f);

    size_t agg_bit_exact = 0;
    double agg_max_diff = 0;
    double agg_sum_diff2 = 0;

    for (int t = 0; t < 9; ++t) {
        for (int f_dim = 0; f_dim < 2560; ++f_dim) {
            float z = 0.0f;
            for (int s = 0; s < 10; ++s) {
                z += rdown[(t * 10 + s) * 2560 + f_dim] * w[t * 10 + s];
            }
            float expected = rout[t * 2560 + f_dim];
            float diff = std::abs(z - expected);
            if (diff == 0.0f) agg_bit_exact++;
            if (diff > agg_max_diff) agg_max_diff = diff;
            agg_sum_diff2 += diff * diff;
        }
    }

    printf("=== MOE AGGREGATE EQUIVALENCE (23,040 elements) ===\n");
    printf("Total elements:    23040\n");
    printf("Bit-exact matches: %zu (%.2f%%)\n", agg_bit_exact, (agg_bit_exact * 100.0) / 23040.0);
    printf("Max absolute diff: %.9e\n", agg_max_diff);
    printf("RMS difference:    %.9e\n", std::sqrt(agg_sum_diff2 / 23040.0));

    if (mismatches == 0 && agg_max_diff == 0.0) {
        printf(">>> PHASE 3 NUMERICAL PROOF: 100%% VERIFIED PASS <<<\n");
        return 0;
    }
    return 1;
}
