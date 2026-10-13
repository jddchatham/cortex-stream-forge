#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

static void stats(const char * tag, const std::vector<float> & v) {
    size_t nz = 0, nan = 0, inf = 0; float mn = INFINITY, mx = -INFINITY; double ss = 0;
    for (float x : v) { if (x != 0) ++nz; if (std::isnan(x)) ++nan; else if (std::isinf(x)) ++inf; else { mn=std::min(mn,x); mx=std::max(mx,x); ss+=(double)x*x; } }
    std::printf("IQ4XS_REPRO tag=%s n=%zu nonzero=%zu min=%g max=%g rms=%g nan=%zu inf=%zu first=%g,%g,%g,%g\n",
        tag, v.size(), nz, mn, mx, std::sqrt(ss/std::max<size_t>(1,v.size())), nan, inf,
        v.size()>0?v[0]:0, v.size()>1?v[1]:0, v.size()>2?v[2]:0, v.size()>3?v[3]:0);
}

static bool run(bool quantized, ggml_backend_t backend, ggml_context * ctx,
        const std::vector<uint8_t> & qbytes, const std::vector<float> & fweights,
        const std::vector<float> & input, std::vector<float> & output) {
    constexpr int64_t K=10240, N=320;
    ggml_tensor * w = quantized ? ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, K, N)
                                : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_name(w, quantized ? "iq4xs_weight" : "f32_weight");
    ggml_set_name(x, "input"); ggml_set_name(y, "output");
    ggml_set_input(w); ggml_set_input(x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return false;
    const void * wsrc = quantized ? static_cast<const void *>(qbytes.data()) : static_cast<const void *>(fweights.data());
    ggml_backend_tensor_set(w, wsrc, 0, quantized ? qbytes.size() : fweights.size()*sizeof(float));
    ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(buffer); return false; }
    ggml_backend_synchronize(backend);
    output.resize(N);
    ggml_backend_tensor_get(y, output.data(), 0, output.size()*sizeof(float));
    ggml_backend_buffer_free(buffer);
    return true;
}

static bool run_sched(bool quantized, ggml_backend_t backend,
        const std::vector<uint8_t> & qbytes, const std::vector<float> & fweights,
        const std::vector<float> & input, std::vector<float> & output) {
    constexpr int64_t K=10240, N=320;
    ggml_init_params p{ .mem_size = 512u*1024u*1024u, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;
    ggml_tensor * w = quantized ? ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_XS, K, N)
                                : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_name(w, quantized ? "sched_iq4xs_weight" : "sched_f32_weight");
    ggml_set_name(x, "sched_input"); ggml_set_name(y, "sched_output");
    ggml_set_input(w); ggml_set_input(x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, 1024, false, false);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, graph)) {
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    const void * wsrc = quantized ? static_cast<const void *>(qbytes.data()) : static_cast<const void *>(fweights.data());
    ggml_backend_tensor_set(w, wsrc, 0, quantized ? qbytes.size() : fweights.size()*sizeof(float));
    ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
    const ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    ggml_backend_sched_synchronize(sched);
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        output.resize(N);
        ggml_backend_tensor_get(y, output.data(), 0, output.size()*sizeof(float));
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    constexpr int64_t K=10240, N=320;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_IQ4_XS, K);
    std::vector<float> row(K), input(K), reference(N), decoded(K);
    for (int64_t i=0;i<K;++i) { row[i]=0.001f*(1+(i%29)); input[i]=0.0007f*(1+(i%17)); }
    std::vector<uint8_t> qbytes(row_bytes*N);
    if (argc == 3) {
        std::ifstream wf(argv[1], std::ios::binary), xf(argv[2], std::ios::binary);
        wf.seekg(0, std::ios::end); xf.seekg(0, std::ios::end);
        const auto ws = wf.tellg(); const auto xs = xf.tellg();
        if (ws != (std::streamoff) qbytes.size() || xs != (std::streamoff) (K*sizeof(float))) return 6;
        wf.seekg(0); xf.seekg(0);
        std::vector<float> exact_input(K);
        wf.read(reinterpret_cast<char *>(qbytes.data()), qbytes.size());
        xf.read(reinterpret_cast<char *>(exact_input.data()), K*sizeof(float));
        if (!wf || !xf) return 7;
        input = std::move(exact_input);
        std::printf("IQ4XS_REPRO operands=exact weight_bytes=%zu input_bytes=%zu\n", qbytes.size(), input.size()*sizeof(float));
    } else {
        for (int64_t j=0;j<N;++j) quantize_row_iq4_xs_ref(row.data(), reinterpret_cast<block_iq4_xs *>(qbytes.data()+j*row_bytes), K);
        std::printf("IQ4XS_REPRO operands=synthetic weight_bytes=%zu input_bytes=%zu\n", qbytes.size(), input.size()*sizeof(float));
    }
    for (int64_t j=0;j<N;++j) {
        dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(qbytes.data()+j*row_bytes), decoded.data(), K);
        double sum=0; for (int64_t i=0;i<K;++i) sum+=(double)decoded[i]*input[i]; reference[j]=(float)sum;
    }
    std::vector<float> deq_weights(K*N);
    for (int64_t j=0;j<N;++j) dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(qbytes.data()+j*row_bytes), deq_weights.data()+j*K, K);
    stats("reference", reference);
    std::vector<uint8_t> q8_bytes(ggml_row_size(GGML_TYPE_Q8_K, K));
    std::vector<float> q8_input(K), q8_reference(N);
    quantize_row_q8_K_ref(input.data(), reinterpret_cast<block_q8_K *>(q8_bytes.data()), K);
    dequantize_row_q8_K(reinterpret_cast<const block_q8_K *>(q8_bytes.data()), q8_input.data(), K);
    for (int64_t j=0;j<N;++j) { double sum=0; for (int64_t i=0;i<K;++i) sum+=(double)deq_weights[j*K+i]*q8_input[i]; q8_reference[j]=(float)sum; }
    std::printf("IQ4XS_REPRO row0_dequant_first32="); for (int i=0;i<32;++i) std::printf("%s%.9g", i ? "," : "", deq_weights[i]); std::printf("\n");

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 2;
    std::printf("IQ4XS_REPRO backend=cpu row_bytes=%zu threads=default\n", row_bytes);
    ggml_init_params p{ .mem_size = 512u*1024u*1024u, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context * cq = ggml_init(p);
    ggml_context * cf = ggml_init(p);
    if (!cq || !cf) return 3;
    std::vector<float> iq, f32, iq_sched, f32_sched;
    if (!run(true, backend, cq, qbytes, deq_weights, input, iq) || !run(false, backend, cf, qbytes, deq_weights, input, f32)) return 4;
    if (!run_sched(true, backend, qbytes, deq_weights, input, iq_sched) || !run_sched(false, backend, qbytes, deq_weights, input, f32_sched)) return 5;
    stats("iq4xs", iq); stats("f32", f32); stats("sched_iq4xs", iq_sched); stats("sched_f32", f32_sched); stats("q8_reference", q8_reference);
    double max_iq=0, max_f=0, max_si=0, max_sf=0, max_q8=0, sum_q8=0; size_t over_q8=0;
    for (size_t i=0;i<reference.size();++i) {
        max_iq=std::max(max_iq,std::abs((double)iq[i]-reference[i]));
        max_f=std::max(max_f,std::abs((double)f32[i]-reference[i]));
        max_si=std::max(max_si,std::abs((double)iq_sched[i]-reference[i]));
        max_sf=std::max(max_sf,std::abs((double)f32_sched[i]-reference[i]));
        const double eq=std::abs((double)iq[i]-q8_reference[i]); max_q8=std::max(max_q8,eq); sum_q8+=eq*eq; if(eq>1e-4)++over_q8;
    }
    std::printf("IQ4XS_REPRO errors max_abs_iq4xs=%g max_abs_f32=%g max_abs_sched_iq4xs=%g max_abs_sched_f32=%g max_abs_q8_reference=%g rms_q8_reference=%g over_q8_reference=%zu\n", max_iq, max_f, max_si, max_sf, max_q8, std::sqrt(sum_q8/N), over_q8);
    for (int i : {0,1,2,159,319}) std::printf("IQ4XS_REPRO channel=%d production=%g scalar_f32=%g scalar_q8=%g\n", i, iq[i], reference[i], q8_reference[i]);
    ggml_free(cq); ggml_free(cf); ggml_backend_free(backend);
    return 0;
}
