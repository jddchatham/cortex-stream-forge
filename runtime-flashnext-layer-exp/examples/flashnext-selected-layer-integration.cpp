#include "llama.h"
#include "llama-context.h"
#include "flashnext-selected-moe.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <array>
#include <limits>

struct owned_post_moe_payload {
    std::vector<float> data;
    size_t pre_off = 0, selected_off = 0, residual_off = 0, inject_off = 0;
    size_t pre_elems = 0, selected_elems = 0, residual_elems = 0, inject_elems = 0;
};

static bool pack_post_moe(const std::vector<float> & pre, const std::vector<float> & selected,
        const std::vector<float> & residual, const std::vector<float> & inject,
        size_t n_embd, size_t hc, size_t n_tokens, owned_post_moe_payload & out) {
    if (!n_embd || !hc || !n_tokens || n_embd > SIZE_MAX / n_tokens) return false;
    const size_t mat = n_embd * n_tokens;
    if (mat > SIZE_MAX / hc) return false;
    const size_t res = mat * hc, inj = hc * n_tokens;
    if (pre.size() != mat || selected.size() != mat || residual.size() != res || inject.size() != inj) return false;
    for (const auto * v : { &pre, &selected, &residual, &inject }) for (float x : *v) if (!std::isfinite(x)) return false;
    if (mat > SIZE_MAX - mat || mat + mat > SIZE_MAX - res || mat + mat + res > SIZE_MAX - inj) return false;
    out.pre_off = 0; out.selected_off = mat; out.residual_off = mat + mat; out.inject_off = out.residual_off + res;
    out.pre_elems = mat; out.selected_elems = mat; out.residual_elems = res; out.inject_elems = inj;
    out.data.resize(out.inject_off + inj);
    std::memcpy(out.data.data() + out.pre_off, pre.data(), mat * sizeof(float));
    std::memcpy(out.data.data() + out.selected_off, selected.data(), mat * sizeof(float));
    std::memcpy(out.data.data() + out.residual_off, residual.data(), res * sizeof(float));
    std::memcpy(out.data.data() + out.inject_off, inject.data(), inj * sizeof(float));
    return true;
}

static bool pack_smoke() {
    for (size_t nt : { size_t(1), size_t(3) }) {
        const size_t n = 8, hc = 4;
        std::vector<float> a(n*nt, 1), b(n*nt, 2), r(n*hc*nt, 3), i(hc*nt, 4);
        owned_post_moe_payload p;
        if (!pack_post_moe(a,b,r,i,n,hc,nt,p) || p.data.size() != 2*n*nt + n*hc*nt + hc*nt ||
            std::memcmp(p.data.data()+p.selected_off, b.data(), b.size()*sizeof(float)) != 0 ||
            p.inject_off + p.inject_elems != p.data.size()) return false;

        auto bad = r;
        bad.pop_back();
        if (pack_post_moe(a, b, bad, i, n, hc, nt, p)) return false;
        auto nan = a;
        nan[0] = NAN;
        if (pack_post_moe(nan, b, r, i, n, hc, nt, p)) return false;
    }
    return true;
}

static void stage(const char * s) { std::fprintf(stderr, "CORTEX_SELECTED_LAYER stage=%s\n", s); std::fflush(stderr); }

struct state_snap { std::vector<uint8_t> b; size_t bytes = 0; };
static std::array<state_snap, 3> snap_states(llama_context * c, int layer, const char * tag) {
    std::array<state_snap,3> out{};
    auto * in = reinterpret_cast<llama_context *>(c);
    for (int k=0;k<3;++k) {
        size_t n = in->debug_copy_recurrent_state(layer, k, nullptr, 0);
        if (!n) { fprintf(stderr,"CORTEX_STATE tag=%s layer=%d kind=%d applicable=0\n",tag,layer,k); continue; }
        out[k].b.resize(n); out[k].bytes = in->debug_copy_recurrent_state(layer,k,out[k].b.data(),n);
        size_t bad=0; for (size_t i=0;i<out[k].b.size();i+=sizeof(float)) { float v; memcpy(&v,out[k].b.data()+i,sizeof(v)); if (!std::isfinite(v)) ++bad; }
        fprintf(stderr,"CORTEX_STATE tag=%s layer=%d kind=%d applicable=1 bytes=%zu finite_bad=%zu\n",tag,layer,k,n,bad);
    }
    return out;
}
static void compare_states(const std::array<state_snap,3> & a, const std::array<state_snap,3> & b, const char * tag) {
    for (int k=0;k<3;++k) if (a[k].bytes && b[k].bytes) { size_t n=std::min(a[k].b.size(),b[k].b.size())/sizeof(float), changed=0; double ma=0,rms=0; for(size_t i=0;i<n;++i){float x,y;memcpy(&x,a[k].b.data()+4*i,4);memcpy(&y,b[k].b.data()+4*i,4);double d=std::abs(double(x)-y);if(d>0)++changed;ma=std::max(ma,d);rms+=d*d;} fprintf(stderr,"CORTEX_STATE_COMPARE tag=%s kind=%d bytes_a=%zu bytes_b=%zu changed=%zu max_abs=%g rms=%g\n",tag,k,a[k].bytes,b[k].bytes,changed,ma,std::sqrt(rms/std::max<size_t>(1,n))); }
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    if (!pack_smoke()) { std::fprintf(stderr, "CORTEX_POST_MOE_PACK result=FAIL\n"); return 13; }
    std::fprintf(stderr, "CORTEX_POST_MOE_PACK result=PASS cases=1,3\n");
    if (std::getenv("CORTEX_FLASHNEXT_PACK_ONLY")) return 0;
    std::fprintf(stderr, "CORTEX_SELECTED_LAYER pid=%d checkpoint=ENTRY\n", (int) getpid());
    std::fflush(stderr);
    setenv("CORTEX_FLASHNEXT_EXTERNAL_EMBEDDING", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_START", "0", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_END", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
    setenv("CORTEX_FLASHNEXT_SYNTHETIC_SELECTED_PAYLOAD", "1", 1);
    setenv("CORTEX_FLASHNEXT_SYNTHETIC_NONZERO_RECURRENT", "1", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { stage("MODEL_INITIALIZED_FAIL"); return 3; }
    stage("MODEL_INITIALIZED");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8; cp.n_batch = 1;
    llama_context * public_ctx = llama_init_from_model(model, cp);
    if (!public_ctx) { stage("CONTEXT_INITIALIZED_FAIL"); llama_model_free(model); return 4; }
    stage("CONTEXT_INITIALIZED");
    int callback_count = 0;
    bool callback_post_ok = false;
    cortex_post_moe_result callback_post_output;
    public_ctx->set_cortex_decode_callback([&](const llama_ubatch & ub, llama_memory_context_i * mem, llm_graph_result * result) {
        ++callback_count;
        std::fprintf(stderr, "CORTEX_TEST_CALLBACK count=%d position=%d memory=%s result=%s\n",
                callback_count, ub.n_tokens ? ub.pos[0] : -1,
                mem ? "VALID" : "NULL", result ? "AVAILABLE" : "NULL");
        std::fflush(stderr);
        if (std::getenv("CORTEX_TEST_CALLBACK_FAIL") != nullptr) return false;
        if (!mem || !result) return false;
        auto * c = reinterpret_cast<llama_context *>(public_ctx);
        cortex_pre_moe_result boundary;
        if (!c->capture_cortex_pre_moe(boundary) || !boundary.valid) return false;
        std::vector<int32_t> ids;
        std::vector<double> weights;
        if (!flashnext_route_softmax_topk(boundary.router_logits.values.data(),
                (size_t) boundary.router_logits.ne[0], boundary.n_tokens, 10, 1.0f, ids, weights)) return false;
        for (size_t t = 0; t < boundary.n_tokens; ++t) for (size_t s = 0; s < 10; ++s) {
            ids[t*10+s] = (s & 1) ? 3 : 0;
            weights[t*10+s] = 0.1;
        }
        flashnext_tensor_index expert_index;
        std::string index_error;
        if (!expert_index.load(argv[2], &index_error)) return false;
        std::vector<float> selected(boundary.hidden.values.size());
        if (!flashnext_execute_selected_moe_rows(expert_index, argv[2], boundary.hidden.values.data(),
                ids.data(), weights.data(), boundary.n_tokens, selected.data())) return false;
        owned_post_moe_payload packed;
        if (!pack_post_moe(boundary.hidden.values, selected, boundary.residual_hc.values,
                boundary.inject.values, (size_t) boundary.hidden.ne[0], 4, boundary.n_tokens, packed)) return false;
        setenv("CORTEX_FLASHNEXT_POST_MOE", "1", 1);
        callback_post_ok = c->cortex_execute_post_moe_in_transaction(ub, mem,
                packed.data.data(), packed.data.size() * sizeof(float), callback_post_output);
        unsetenv("CORTEX_FLASHNEXT_POST_MOE");
        return callback_post_ok && callback_post_output.valid;
    });
    const auto state_before = snap_states(public_ctx, 0, "before_pre_moe");

    const int n_embd = llama_model_n_embd(model);
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    // Embedding mode leaves batch.token unset, but Qwen4Exp still needs the
    // original token ID for PLE metadata. Keep that ID separately owned.
    if (!batch.embd || !batch.pos || !batch.n_seq_id ||
            !batch.seq_id || !batch.seq_id[0] || !batch.logits) {
        std::fprintf(stderr, "CORTEX_SELECTED_LAYER stage=BATCH_INIT_FAIL reason=allocation\n");
        llama_batch_free(batch); llama_free(public_ctx); llama_model_free(model); return 5;
    }
    stage("BATCH_INITIALIZED");
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    for (int i = 0; i < n_embd; ++i) batch.embd[i] = 0.01f * (float) (1 + (i % 17));
    batch.n_tokens = 1;
    stage("SELECTED_LAYER_BOUND");
    const int rc = llama_decode(public_ctx, batch);
    if (rc != 0) {
        std::fprintf(stderr, "CORTEX_SELECTED_LAYER stage=PRE_MOE_EXECUTED_FAIL decode=%d\n", rc);
        llama_batch_free(batch); llama_free(public_ctx); llama_model_free(model); return 5;
    }
    stage("PRE_MOE_EXECUTED");
    if (callback_post_ok) {
        stage("POST_MOE_EXECUTED");
        llama_batch_free(batch);
        std::fprintf(stderr, "CORTEX_RESET_ENV present=%d\n", std::getenv("CORTEX_TEST_RESET_S") != nullptr ? 1 : 0);
        if (std::getenv("CORTEX_TEST_RESET_S") != nullptr) {
            auto * c = reinterpret_cast<llama_context *>(public_ctx);
            const auto before_reset = snap_states(public_ctx, 0, "before_s_reset");
            if (!c->debug_zero_recurrent_state(0, 1)) { stage("S_RESET_FAIL"); llama_free(public_ctx); llama_model_free(model); return 14; }
            auto reset = snap_states(public_ctx, 0, "after_s_reset");
            if (!reset[1].bytes) { stage("S_RESET_VERIFY_FAIL"); llama_free(public_ctx); llama_model_free(model); return 14; }
            for (uint8_t v : reset[1].b) if (v != 0) { stage("S_RESET_VERIFY_FAIL"); llama_free(public_ctx); llama_model_free(model); return 14; }
            compare_states(before_reset, reset, "s_reset_delta");
            std::fprintf(stderr, "CORTEX_S_RESET result=PASS layer=0 kind=1 bytes=%zu\n", reset[1].bytes);
        }
        llama_batch step1 = llama_batch_init(1, n_embd, 1);
        if (!step1.embd || !step1.pos || !step1.n_seq_id ||
                !step1.seq_id || !step1.seq_id[0] || !step1.logits) {
            stage("SECOND_PRE_MOE_FAIL allocation");
            llama_batch_free(step1); llama_free(public_ctx); llama_model_free(model); return 11;
        }
        step1.pos[0] = 1;
        step1.n_seq_id[0] = 1;
        step1.seq_id[0][0] = 0;
        step1.logits[0] = 1;
        for (int i = 0; i < n_embd; ++i) step1.embd[i] = 0.02f * (float) (1 + (i % 17));
        step1.n_tokens = 1;
        if (llama_decode(public_ctx, step1) != 0) {
            stage("SECOND_PRE_MOE_FAIL decode");
            llama_batch_free(step1); llama_free(public_ctx); llama_model_free(model); return 12;
        }
        stage("SECOND_PRE_MOE_EXECUTED");
        auto final_state = snap_states(public_ctx, 0, "after_token1");
        if (final_state[1].bytes >= 4) {
            float first = 0, last = 0;
            std::memcpy(&first, final_state[1].b.data(), sizeof(first));
            std::memcpy(&last, final_state[1].b.data() + final_state[1].b.size() - sizeof(last), sizeof(last));
            std::fprintf(stderr, "CORTEX_S_FINAL bytes=%zu first=%g last=%g\n", final_state[1].bytes, first, last);
        }
        stage("HC_OUTPUT_CAPTURED");
        llama_batch_free(step1); llama_free(public_ctx); llama_model_free(model);
        return 0;
    }
    const auto state_after_pre = snap_states(public_ctx, 0, "after_pre_moe");
    compare_states(state_before, state_after_pre, "pre_moe_delta");

    auto * ctx = reinterpret_cast<llama_context *>(public_ctx);
    cortex_pre_moe_result boundary;
    if (!ctx->capture_cortex_pre_moe(boundary) || !boundary.valid) {
        stage("FOUR_TENSORS_CAPTURED_FAIL");
        llama_batch_free(batch); llama_free(public_ctx); llama_model_free(model); return 6;
    }
    stage("FOUR_TENSORS_CAPTURED");
    std::vector<int32_t> ids;
    std::vector<double> weights;
    if (!flashnext_route_softmax_topk(boundary.router_logits.values.data(),
            (size_t) boundary.router_logits.ne[0], boundary.n_tokens, 10, 1.0f, ids, weights)) {
        stage("ROUTING_EXECUTED_FAIL");
        llama_batch_free(batch); llama_free(public_ctx); llama_model_free(model); return 7;
    }
    stage("ROUTING_EXECUTED");
    // The integration path is deliberately canonical-only.  The sparse
    // fixture index is supplied by argv[2]; no synthetic expert fallback is
    // permitted here.
    for (size_t t = 0; t < boundary.n_tokens; ++t) {
        for (size_t s = 0; s < 10; ++s) {
            ids[t*10+s] = (s & 1) ? 3 : 0;
            weights[t*10+s] = 0.1;
        }
    }
    flashnext_tensor_index expert_index;
    std::string index_error;
    if (!expert_index.load(argv[2], &index_error)) {
        std::fprintf(stderr, "CORTEX_SELECTED_LAYER stage=CANONICAL_SELECTED_MOE_EXECUTED_FAIL reason=index:%s\n", index_error.c_str());
        llama_batch_free(batch); llama_free(public_ctx); llama_model_free(model); return 8;
    }
    std::vector<float> selected((size_t) n_embd * boundary.n_tokens);
    if (!flashnext_execute_selected_moe_rows(expert_index, argv[2], boundary.hidden.values.data(), ids.data(), weights.data(), boundary.n_tokens, selected.data())) {
        std::fprintf(stderr, "CORTEX_SELECTED_LAYER stage=CANONICAL_SELECTED_MOE_EXECUTED_FAIL reason=provider\n");
        llama_batch_free(batch); llama_free(public_ctx); llama_model_free(model); return 8;
    }
    stage("CANONICAL_SELECTED_MOE_EXECUTED");
    llama_batch_free(batch);
    // Retain public_ctx: graph/scheduler reconstruction must not destroy the
    // llama_memory_hybrid owner of Qwen4Exp R/S/P state.
    setenv("CORTEX_FLASHNEXT_POST_MOE", "1", 1);
    llama_context * post_ctx = public_ctx;
    llama_batch post = llama_batch_init(1, n_embd, 1);
    free(post.embd);
    post.embd = nullptr;
    owned_post_moe_payload packed;
    if (!pack_post_moe(boundary.hidden.values, selected, boundary.residual_hc.values,
            boundary.inject.values, (size_t) n_embd, 4, boundary.n_tokens, packed)) {
        stage("POST_MOE_PACK_FAIL validation");
        llama_batch_free(post); llama_free(post_ctx); llama_model_free(model); return 13;
    }
    post.embd = packed.data.data();
    post.token = (llama_token *) malloc(sizeof(llama_token));
    if (!post.token) { stage("POST_MOE_EXECUTED_FAIL allocation"); post.embd = nullptr; llama_batch_free(post); llama_free(post_ctx); llama_model_free(model); return 9; }
    post.token[0] = 1; post.pos[0] = 1; post.n_seq_id[0] = 1; post.seq_id[0][0] = 0; post.logits[0] = 1; post.n_tokens = 1;
    if (llama_decode(post_ctx, post) != 0) { stage("POST_MOE_EXECUTED_FAIL decode"); post.embd = nullptr; llama_batch_free(post); llama_free(post_ctx); llama_model_free(model); return 10; }
    stage("POST_MOE_EXECUTED");
    const auto state_after_post = snap_states(post_ctx, 0, "after_post_moe");
    compare_states(state_after_pre, state_after_post, "post_moe_delta");

    // Re-enter the real pre-MoE graph on the same persistent context.  The
    // post-MoE transaction above is not a recurrent-body step; remove only
    // its graph-selection flag and retain the memory owner and state buffers.
    unsetenv("CORTEX_FLASHNEXT_POST_MOE");
    llama_batch step1 = llama_batch_init(1, n_embd, 1);
    if (!step1.embd || !step1.pos || !step1.n_seq_id ||
            !step1.seq_id || !step1.seq_id[0] || !step1.logits) {
        stage("SECOND_PRE_MOE_FAIL allocation");
        llama_batch_free(step1); post.embd = nullptr; llama_batch_free(post); llama_free(post_ctx); llama_model_free(model); return 11;
    }
    step1.pos[0] = 1;
    step1.n_seq_id[0] = 1;
    step1.seq_id[0][0] = 0;
    step1.logits[0] = 1;
    for (int i = 0; i < n_embd; ++i) step1.embd[i] = 0.02f * (float) (1 + (i % 17));
    step1.n_tokens = 1;
    const auto state_before_step1 = snap_states(post_ctx, 0, "before_pre_moe_token1");
    if (llama_decode(post_ctx, step1) != 0) {
        stage("SECOND_PRE_MOE_FAIL decode");
        llama_batch_free(step1); post.embd = nullptr; llama_batch_free(post); llama_free(post_ctx); llama_model_free(model); return 12;
    }
    stage("SECOND_PRE_MOE_EXECUTED");
    const auto state_after_step1 = snap_states(post_ctx, 0, "after_pre_moe_token1");
    compare_states(state_before_step1, state_after_step1, "second_pre_moe_delta");
    llama_batch_free(step1);
    stage("HC_OUTPUT_CAPTURED");
    post.embd = nullptr;
    llama_batch_free(post); llama_free(post_ctx); llama_model_free(model);
    return 0;
}
