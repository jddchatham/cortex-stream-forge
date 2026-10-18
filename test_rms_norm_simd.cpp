#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

using bench_clock = std::chrono::high_resolution_clock;

static float sum_sqr_scalar(const float * x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)(x[i] * x[i]);
    return (float)sum;
}

static float sum_sqr_avx2(const float * x, int n) {
    int i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(x + i);
        __m256 v1 = _mm256_loadu_ps(x + i + 8);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // Horizontal add of 8 floats
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 v8 = _mm_add_ps(lo, hi);
    __m128 s1 = _mm_hadd_ps(v8, v8);
    __m128 s2 = _mm_hadd_ps(s1, s1);
    float sum = _mm_cvtss_f32(s2);
    for (; i < n; ++i) sum += x[i] * x[i];
    return sum;
}

int main() {
    const int n = 1024;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    const int iters = 100000;

    auto t0 = bench_clock::now();
    float s_sc = 0.0f;
    for (int it = 0; it < iters; ++it) s_sc += sum_sqr_scalar(x.data(), n);
    auto t1 = bench_clock::now();
    double ms_sc = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = bench_clock::now();
    float s_simd = 0.0f;
    for (int it = 0; it < iters; ++it) s_simd += sum_sqr_avx2(x.data(), n);
    auto t3 = bench_clock::now();
    double ms_simd = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("Scalar sum_sqr: %6.2f ms (res: %e)\n", ms_sc, s_sc);
    std::printf("AVX2/FMA sum_sqr: %6.2f ms (res: %e)\n", ms_simd, s_simd);
    std::printf("Speedup: %.2fx\n", ms_sc / ms_simd);

    return 0;
}
