// Stock two-layer reference runner (diagnostic).
//
// Runs the unmodified stock FlashNext graph for a single token decode:
// normal model load (no low-residency probe, no pread arena, no external
// global tensors, no segment cuts), in-graph token embedding lookup, and the
// integrity-protected diagnostic captures. Used as the stock reference for
// streamed two-layer equivalence checks.

#include "llama.h"
#include "../src/llama-context.h"

#include <algorithm>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static bool write_stock_logits(const std::string & dir, size_t index, const float * logits, size_t n, const char * prefix = "logits-select-") {
    const std::string path = dir + "/" + prefix + std::to_string(index) + ".bin";
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) { std::perror(("open " + path).c_str()); return false; }
    const uint8_t * data = reinterpret_cast<const uint8_t *>(logits);
    size_t bytes = n * sizeof(float), done = 0;
    while (done < bytes) {
        const ssize_t w = ::write(fd, data + done, bytes - done);
        if (w <= 0) { ::close(fd); return false; }
        done += (size_t) w;
    }
    return ::close(fd) == 0;
}

static bool run_stock_prompt(llama_model * model, llama_context * ctx, const char * text) {
    const char * out_dir = std::getenv("STOCK_PROMPT_LOGITS_DIR");
    if (out_dir == nullptr || *out_dir == '\0') {
        std::fprintf(stderr, "STOCK_PROMPT_LOGITS_DIR must name a new directory\n"); return false;
    }
    if (::mkdir(out_dir, 0700) != 0) { std::perror("mkdir stock logits directory"); return false; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl == nullptr || std::strstr(tmpl, "enable_thinking") == nullptr ||
            std::strstr(tmpl, "<think>\\n\\n</think>\\n\\n") == nullptr) {
        std::fprintf(stderr, "stock model template lacks verified non-thinking support\n"); return false;
    }
    // Same literal rendering of the GGUF template's enable_thinking=false branch as QUORTEX.
    const std::string prompt = std::string("<|im_start|>user\n") + text +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::fprintf(stderr, "QIES072 stock nonthinking_template=GGUF_native enable_thinking=false\n");
    std::fprintf(stderr, "QIES072 stock formatted_prompt=%s\n", prompt.c_str());
    const int32_t n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, false, true);
    if (n_prompt <= 0 || n_prompt + 16 > (int32_t) llama_n_ctx(ctx)) {
        std::fprintf(stderr, "stock prompt is empty or exceeds context capacity\n"); return false;
    }
    std::vector<llama_token> tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), n_prompt, false, true) != n_prompt) return false;
    std::printf("STOCK_PROMPT_TOKENS=%d IDS=", n_prompt);
    for (int32_t i = 0; i < n_prompt; ++i) std::printf("%s%d", i ? "," : "", (int) tokens[i]);
    std::printf("\n");

    const auto decode_token = [&](llama_token token, llama_pos pos) -> const float * {
        ctx->debug_reset_flashnext_diag_capture_counters();
        if (std::getenv("CORTEX_FLASHNEXT_DIAG_DUMP_ROOT")) {
            const std::string dir = std::string(std::getenv("CORTEX_FLASHNEXT_DIAG_DUMP_ROOT")) + "/p" + std::to_string(pos);
            if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) return nullptr;
            setenv("CORTEX_FLASHNEXT_DIAG_DUMP", dir.c_str(), 1);
        }
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = token; batch.pos[0] = pos; batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0; batch.logits[0] = 1; batch.n_tokens = 1;
        const int rc = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        llama_batch_free(batch);
        return rc == 0 ? llama_get_logits_ith(ctx, 0) : nullptr;
    };
    const float * logits = nullptr;
    for (size_t i = 0; i < tokens.size(); ++i) {
        logits = decode_token(tokens[i], (llama_pos) i);
        if (!logits) { std::fprintf(stderr, "stock prompt decode failed at %zu\n", i); return false; }
        if (!write_stock_logits(out_dir, i, logits, (size_t) llama_vocab_n_tokens(vocab))) return false;
        std::printf("STOCK_PREFIX_COMPLETE=%zu\n", i); std::fflush(stdout);
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    uint32_t max_tokens = 1;
    if (const char * n = std::getenv("STOCK_PROMPT_MAX_TOKENS")) {
        max_tokens = (uint32_t) std::strtoul(n, nullptr, 10);
        if (max_tokens == 0 || max_tokens > 16) return false;
    }
    std::vector<llama_token> generated;
    for (uint32_t i = 0; i < max_tokens; ++i) {
        llama_token best = 0;
        for (int t = 1; t < n_vocab; ++t) if (logits[t] > logits[best]) best = t;
        if (llama_vocab_is_eog(vocab, best)) break;
        generated.push_back(best);
        if (i + 1 == max_tokens) break;
        logits = decode_token(best, (llama_pos) (tokens.size() + i));
        if (!logits || !write_stock_logits(out_dir, i + 1, logits, (size_t) n_vocab, "logits-generated-")) return false;
    }
    std::printf("STOCK_GENERATED_TOKENS=%zu IDS=", generated.size());
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", (int) generated[i]);
    std::printf("\n");
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }

    static const char * cortex_envs[] = {
        "CORTEX_LOW_RESIDENCY_LAYER",
        "CORTEX_EXTERNAL_GLOBAL_TENSORS",
        "CORTEX_FLASHNEXT_ROUTER_ONLY",
        "CORTEX_FLASHNEXT_SEGMENT_A",
        "CORTEX_FLASHNEXT_SEGMENT_A_ROUTER",
        "CORTEX_FLASHNEXT_PRE_MOE",
        "CORTEX_FLASHNEXT_LAYER_START",
        "CORTEX_FLASHNEXT_LAYER_END",
        "CORTEX_FLASHNEXT_LAYER_RANGE",
        "CORTEX_QWEN35_SEGMENT_END",
        "CORTEX_FLASHNEXT_FINAL_HEAD",
        "CORTEX_FLASHNEXT_SEGMENT_B",
        "CORTEX_PREAD_ARENA",
        "CORTEX_DIRECT_MMAP",
        "CORTEX_FLASHNEXT_CHAIN_1_2",
        "CORTEX_FLASHNEXT_CHAIN_1_2_3",
        "CORTEX_FLASHNEXT_CHAIN_0_48",
        "CORTEX_FLASHNEXT_CHAIN_0_N",
        "CORTEX_FLASHNEXT_INJECT_PRE_COMMIT_FAILURE",
    };
    for (const char * name : cortex_envs) {
        unsetenv(name);
    }
    if (std::getenv("STOCK_NO_DIAG") == nullptr) {
        setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "STOCK_REF model load failed\n");
        llama_backend_free();
        return 3;
    }

    llama_context_params cp = llama_context_default_params();
    uint32_t stock_n_ctx = 256;
    if (const char * n_env = std::getenv("STOCK_N_TOKENS")) {
        stock_n_ctx = std::max<uint32_t>(256, (uint32_t) std::strtoul(n_env, nullptr, 10) + 16);
    }
    cp.n_ctx  = stock_n_ctx;
    cp.n_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "STOCK_REF context init failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    if (const char * prompt = std::getenv("STOCK_PROMPT")) {
        const bool ok = run_stock_prompt(model, ctx, prompt);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return ok ? 0 : 5;
    }

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0]    = 248045;
    batch.pos[0]      = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]   = 1;
    batch.n_tokens    = 1;

    const int status = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    std::printf("STOCK_REF_DECODE status=%d\n", status);

    // stock logits reference for the 48-layer chain: full-graph output
    // projection of the final HC-mixed activation, token 248045 pos 0.
    // With TWO_TOKEN=1, additionally decodes the stock argmax token at
    // position 1 and exports the position-1 logits.
    const float * logits = llama_get_logits_ith(ctx, 0);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> logits0;
    if (logits && n_vocab > 0) {
        logits0.assign(logits, logits + n_vocab);
        const char * out = std::getenv("STOCK_LOGITS_OUT");
        FILE * f = std::fopen(out ? out : "stock_logits_p0.bin", "wb");
        if (f) { std::fwrite(logits0.data(), sizeof(float), logits0.size(), f); std::fclose(f); }
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits0[i] > logits0[best]) best = i;
        std::printf("STOCK_REF_LOGITS n=%d argmax=%d logit=%.9g\n", n_vocab, best, logits0[best]);
    }

    uint32_t n_tokens_stock = 1;
    if (std::getenv("STOCK_TWO_TOKEN") != nullptr) n_tokens_stock = 2;
    if (const char * n_env = std::getenv("STOCK_N_TOKENS")) {
        n_tokens_stock = (uint32_t) std::strtoul(n_env, nullptr, 10);
        if (n_tokens_stock < 1) n_tokens_stock = 1;
    }
    const uint32_t stop_at = [&]{
        if (const char * s = std::getenv("STOCK_STOP_AT")) return (uint32_t) std::strtoul(s, nullptr, 10);
        return n_tokens_stock - 1;
    }();
    if (n_tokens_stock > 1 && !logits0.empty()) {
        llama_token token_cur = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits0[i] > logits0[token_cur]) token_cur = i;
        for (uint32_t pos = 1; pos <= stop_at; ++pos) {
            llama_batch batch1 = llama_batch_init(1, 0, 1);
            batch1.token[0]    = token_cur;
            batch1.pos[0]      = (llama_pos) pos;
            batch1.n_seq_id[0] = 1;
            batch1.seq_id[0][0] = 0;
            batch1.logits[0]   = 1;
            batch1.n_tokens    = 1;
            const int status1 = llama_decode(ctx, batch1);
            llama_synchronize(ctx);
            std::printf("STOCK_REF_DECODE_POS%u status=%d token=%d\n", pos, status1, (int) token_cur);
            const float * logits1 = status1 == 0 ? llama_get_logits_ith(ctx, 0) : nullptr;
            if (!logits1 || n_vocab <= 0) { llama_batch_free(batch1); break; }
            const char * out = std::getenv("STOCK_LOGITS_OUT");
            const std::string pp = out ? std::string(out) + ".p" + std::to_string(pos)
                                       : "stock_logits_p" + std::to_string(pos) + ".bin";
            FILE * f = std::fopen(pp.c_str(), "wb");
            if (f) { std::fwrite(logits1, sizeof(float), (size_t) n_vocab, f); std::fclose(f); }
            llama_token best = 0;
            for (int i = 1; i < n_vocab; ++i) if (logits1[i] > logits1[best]) best = i;
            std::printf("STOCK_REF_LOGITS_POS%u n=%d argmax=%d logit=%.9g\n", pos, n_vocab, (int) best, logits1[best]);
            token_cur = best;
            llama_batch_free(batch1);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return status == 0 ? 0 : 5;
}
