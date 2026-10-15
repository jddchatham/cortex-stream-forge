#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "llama-context.h"
#include "flashnext-vocab-projector.h"
#include "../ggml/src/ggml-quants.h"
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <filesystem>

#include <csignal>

static std::string g_cortex_state_dir;

static void cortex_clean_state_dir() {
    if (!g_cortex_state_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(g_cortex_state_dir, ec);
    }
}

static void cortex_signal_handler(int sig) {
    cortex_clean_state_dir();
    _exit(128 + sig);
}

static std::string cortex_get_state_dir() {
    if (!g_cortex_state_dir.empty()) return g_cortex_state_dir;

    const char * env_dir = std::getenv("CORTEX_STATE_DIR");
    if (env_dir && *env_dir) {
        g_cortex_state_dir = env_dir;
        std::error_code ec;
        std::filesystem::create_directories(g_cortex_state_dir, ec);
        return g_cortex_state_dir;
    }

    // Secure isolated per-process state directory with 0700 permissions
    // Placed in ~/.cortex_states on disk (/home partition) so tmpfs is not exhausted by recurrent state checkpoints
    const char * home = std::getenv("HOME");
    std::string base = home ? std::string(home) + "/.cortex_states" : "/home/user/.cortex_states";
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    std::string path_template = base + "/session_XXXXXX";
    std::vector<char> tmpl_buf(path_template.begin(), path_template.end());
    tmpl_buf.push_back('\0');
    char * res = mkdtemp(tmpl_buf.data());
    if (res) {
        g_cortex_state_dir = res;
    } else {
        g_cortex_state_dir = base;
    }

    std::atexit(cortex_clean_state_dir);
    std::signal(SIGINT, cortex_signal_handler);
    std::signal(SIGTERM, cortex_signal_handler);

    return g_cortex_state_dir;
}

static bool cortex_chat_is_debug() {
    static const bool debug = (std::getenv("CORTEX_DEBUG") != nullptr && std::strcmp(std::getenv("CORTEX_DEBUG"), "0") != 0) ||
                              (std::getenv("CORTEX_DIAG") != nullptr && std::strcmp(std::getenv("CORTEX_DIAG"), "0") != 0) ||
                              (std::getenv("CORTEX_VERBOSE") != nullptr && std::strcmp(std::getenv("CORTEX_VERBOSE"), "0") != 0);
    return debug;
}

#define cortex_log_diag(...) do { if (cortex_chat_is_debug()) { std::fprintf(stderr, __VA_ARGS__); std::fflush(stderr); } } while(0)

static llama_token process_positions(llama_model * model, std::vector<std::vector<uint8_t>> & states,
                                     const std::vector<llama_token> & tokens, llama_pos pos,
                                     std::vector<float> & hidden, llama_sampler * sampler) {
    cortex_log_diag("CORTEX_CHAT checkpoint=PROCESS_ENTRY layer=-1 tokens=%zu result=BEGIN\n", tokens.size());
    const int n_layer = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_tokens = (int) tokens.size();
    const int hc = 4;
    std::vector<int32_t> token_ids(tokens.begin(), tokens.end());
    std::vector<float> external_embd;
    external_embd.resize((size_t) n_tokens * n_embd);
    cortex_log_diag("CORTEX_CHAT checkpoint=TOKEN_ROWS_BEFORE layer=-1 tokens=%zu\n", tokens.size());
    if (!llama_cortex_external_token_rows(model, token_ids.data(), token_ids.size(), external_embd.data())) {
        std::fprintf(stderr, "CORTEX_CHAT checkpoint=TOKEN_ROWS_AFTER layer=-1 result=FAIL reason=row_materialization\n");
        std::fflush(stderr);
        return -1;
    }
    cortex_log_diag("CORTEX_CHAT checkpoint=TOKEN_ROWS_AFTER layer=-1 result=PASS\n");
    for (int il = 0; il < n_layer; ++il) {
        char start[32], end[32];
        std::snprintf(start, sizeof(start), "%d", il);
        std::snprintf(end, sizeof(end), "%d", il + 1);
        setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
        setenv("CORTEX_FLASHNEXT_LAYER_START", start, 1);
        setenv("CORTEX_FLASHNEXT_LAYER_END", end, 1);
        cortex_log_diag("CORTEX_CHAT checkpoint=REBIND_BEFORE layer=%d\n", il);
        const bool rebound = llama_cortex_activate_layer(model, il);
        cortex_log_diag("CORTEX_CHAT checkpoint=REBIND_AFTER layer=%d result=%s\n", il, rebound ? "PASS" : "FAIL");
        if (!rebound) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=rebind\n", il);
            std::fflush(stderr);
            return -1;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 256; cp.n_batch = n_tokens;
        cortex_log_diag("CORTEX_CHAT checkpoint=CONTEXT_BEFORE layer=%d\n", il);
        llama_context * ctx = llama_init_from_model(model, cp);
        cortex_log_diag("CORTEX_CHAT checkpoint=CONTEXT_AFTER layer=%d result=%s\n", il, ctx ? "PASS" : "FAIL");
        if (!ctx) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=context_init\n", il);
            std::fflush(stderr);
            return -1;
        }
        const bool terminal_layer = il + 1 == n_layer;
        if (terminal_layer) llama_set_embeddings(ctx, true);
        else llama_set_embeddings_layer_inp(ctx, il + 1, true);
        {
            std::string state_file = cortex_get_state_dir() + "/layer_" + std::to_string(il) + ".bin";
            std::ifstream sf(state_file, std::ios::binary | std::ios::ate);
            if (sf.is_open()) {
                std::streamsize sf_size = sf.tellg();
                if (sf_size > 0) {
                    sf.seekg(0, std::ios::beg);
                    std::vector<uint8_t> sf_buf(sf_size);
                    if (sf.read(reinterpret_cast<char *>(sf_buf.data()), sf_size)) {
                        if (llama_state_set_data(ctx, sf_buf.data(), sf_buf.size()) == 0) {
                            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_restore\n", il);
                            std::fflush(stderr);
                            return -1;
                        }
                    }
                }
            }
        }
        if (il == 0 && pos > 0 && std::getenv("CORTEX_CHAT_RESET_S_BEFORE_TOKEN") != nullptr) {
            bool reset_ok = reinterpret_cast<llama_context *>(ctx)->debug_zero_recurrent_state(0, 1);
            std::fprintf(stderr, "CORTEX_CHAT RESET_S position=%lld layer=0 result=%s\n",
                         (long long) pos, reset_ok ? "PASS" : "FAIL");
            if (!reset_ok) { llama_free(ctx); return -1; }
        }
        llama_batch batch;
        if (il == 0) {
            batch = llama_batch_init(n_tokens, n_embd, 1);
            batch.token = (llama_token *) std::malloc(sizeof(llama_token) * n_tokens);
            std::copy(tokens.begin(), tokens.end(), batch.token);
            std::copy(external_embd.begin(), external_embd.end(), batch.embd);
        } else {
            batch = llama_batch_init(n_tokens, n_embd * hc, 1);
            batch.token = (llama_token *) std::malloc(sizeof(llama_token) * n_tokens);
            std::copy(tokens.begin(), tokens.end(), batch.token);
            std::copy(hidden.begin(), hidden.end(), batch.embd);
        }
        for (int i = 0; i < n_tokens; ++i) {
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (il == n_layer - 1 && i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;
        if (il == 47 && std::getenv("CORTEX_KV_READ_CAPTURE") != nullptr) {
            const std::string p = std::to_string(pos + n_tokens - 1);
            setenv("CORTEX_KV_READ_POSITION", p.c_str(), 1);
        }
        const int decode_rc = llama_decode(ctx, batch);
        if (decode_rc != 0) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=decode return_code=%d\n", il, decode_rc);
            std::fflush(stderr);
            return -1;
        }
        if (il == 47 && std::getenv("CORTEX_KV_CAPTURE") != nullptr) {
            const std::string prefix = std::getenv("CORTEX_KV_CAPTURE_PREFIX") ? std::getenv("CORTEX_KV_CAPTURE_PREFIX") : "/tmp/cortex-kv-l47";
            for (int kind = 0; kind < 2; ++kind) {
                const size_t need = reinterpret_cast<llama_context *>(ctx)->debug_copy_kv_cache(47, kind, nullptr, 0);
                std::vector<uint8_t> bytes(need);
                if (need == 0 || reinterpret_cast<llama_context *>(ctx)->debug_copy_kv_cache(47, kind, bytes.data(), bytes.size()) != need) {
                    std::fprintf(stderr, "CORTEX_KV_CAPTURE layer=47 position=%lld kind=%d result=FAIL\n", (long long) pos, kind);
                    llama_batch_free(batch); llama_free(ctx); return -1;
                }
                const std::string path = prefix + "-pos" + std::to_string(pos + n_tokens - 1) + (kind == 0 ? "-k.bin" : "-v.bin");
                std::ofstream f(path, std::ios::binary | std::ios::trunc); f.write((const char *) bytes.data(), (std::streamsize) bytes.size());
                uint64_t h = 14695981039346656037ull; for (uint8_t b : bytes) { h ^= b; h *= 1099511628211ull; }
                std::fprintf(stderr, "CORTEX_KV_CAPTURE layer=47 position=%lld kind=%d bytes=%zu fnv=%016llx path=%s\n", (long long) pos + n_tokens - 1, kind, bytes.size(), (unsigned long long) h, path.c_str());
            }
        }
        if (il + 1 < n_layer) {
            float * out = llama_get_embeddings_layer_inp(ctx, il + 1);
            if (!out) {
                std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=hidden_output\n", il);
                std::fflush(stderr);
                return -1;
            }
            hidden.assign(out, out + (size_t) n_tokens * n_embd * hc);
        } else {
            // Terminal range output is the HC-wide hidden state.  It is
            // captured through the regular embedding output path, matching
            // the validated focused runner, rather than layer-input storage.
            float * hc_ptr = llama_get_embeddings(ctx);
            if (!hc_ptr) {
                std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=terminal_hc_output\n", il);
                llama_batch_free(batch); llama_free(ctx); return -1;
            }
            const size_t hc_floats = (size_t) n_embd * hc;
            if (n_tokens <= 0) { llama_batch_free(batch); llama_free(ctx); return -1; }
            std::vector<float> terminal_hc(hc_ptr + ((size_t) n_tokens - 1) * hc_floats,
                                           hc_ptr + (size_t) n_tokens * hc_floats);
            // The terminal layer context owns the active scheduler and its
            // recurrent state transaction.  Release it after copying the HC
            // boundary and state, before constructing the separate final-head
            // graph context.
            llama_batch_free(batch);
            const size_t state_size = llama_state_get_size(ctx);
            {
                std::vector<uint8_t> tmp_state(state_size);
                if (llama_state_get_data(ctx, tmp_state.data(), state_size) != state_size) {
                    llama_free(ctx); return -1;
                }
                std::string state_file = cortex_get_state_dir() + "/layer_" + std::to_string(il) + ".bin";
                std::ofstream sf(state_file, std::ios::binary | std::ios::trunc);
                if (!sf.write(reinterpret_cast<const char *>(tmp_state.data()), (std::streamsize) state_size)) {
                    llama_free(ctx); return -1;
                }
            }
            cortex_log_diag("CORTEX_CHAT STATE layer=%d kind=serialized bytes=%zu saved_position=%lld restored_position=%lld seq=0\n",
                            il, state_size, (long long) (pos + n_tokens - 1), (long long) pos);
            llama_free(ctx);
            ctx = nullptr;
            llama_context_params hcp = llama_context_default_params(); hcp.n_ctx = 1; hcp.n_batch = 1;
            setenv("CORTEX_FLASHNEXT_FINAL_HEAD", "1", 1);
            setenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY", "1", 1);
            llama_context * head = llama_init_from_model(model, hcp);
            cortex_post_moe_result normalized;
            bool ok = head != nullptr && reinterpret_cast<llama_context *>(head)->cortex_execute_final_head_only(
                terminal_hc.data(), terminal_hc.size(), normalized);
            cortex_vocab_projection_result projection;
            if (ok) {
                cortex_output_weight_descriptor descriptor;
                ok = reinterpret_cast<llama_model_base *>(model)->get_cortex_output_weight_descriptor(descriptor) &&
                     cortex_stream_vocab_project(descriptor, normalized.hc_output, 1024, projection);
            }
            if (head) llama_free(head);
            unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD");
            unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY");
            if (!ok) { return -1; }
            cortex_log_diag("CORTEX_CHAT TOKEN position=%lld id=%d logit=%g bytes=%zu scratch=%zu\n",
                            (long long) (pos + n_tokens - 1), projection.token, projection.logit,
                            projection.bytes_read, projection.scratch_bytes);
            if (cortex_chat_is_debug()) for (size_t r = 0; r < std::min<size_t>(5, projection.top_count); ++r) {
                const llama_vocab * top_vocab = llama_model_get_vocab(model);
                char piece[128]; int pn = llama_token_to_piece(top_vocab, projection.top_tokens[r], piece, sizeof(piece), 0, false);
                std::fprintf(stderr, "CORTEX_CHAT TOP rank=%zu token=%d logit=%g eog=%d control=%d piece=%.*s\n",
                    r + 1, projection.top_tokens[r], projection.top_logits[r],
                    llama_vocab_is_eog(top_vocab, projection.top_tokens[r]),
                    llama_vocab_is_control(top_vocab, projection.top_tokens[r]), std::max(0, pn), pn > 0 ? piece : "");
            }
            if (cortex_chat_is_debug() && pos + n_tokens - 1 == 8) {
                cortex_output_weight_descriptor rd;
                if (reinterpret_cast<llama_model_base *>(model)->get_cortex_output_weight_descriptor(rd) && rd.type == GGML_TYPE_Q6_K) {
                    std::vector<uint8_t> row(rd.row_bytes); std::vector<float> weights((size_t) rd.ne[0]);
                    std::vector<block_q8_K> q8((size_t) rd.ne[0] / QK_K); std::vector<float> q8_f32((size_t) rd.ne[0]);
                    quantize_row_q8_K_ref(normalized.hc_output.data(), q8.data(), rd.ne[0]);
                    dequantize_row_q8_K(q8.data(), q8_f32.data(), rd.ne[0]);
                    for (size_t r = 0; r < projection.top_count; ++r) {
                        const int tok = projection.top_tokens[r];
                        const ssize_t got = pread(rd.file_descriptor, row.data(), row.size(),
                            (off_t) rd.absolute_offset + (off_t) tok * (off_t) rd.row_bytes);
                        float ref = 0.0f;
                        float ref_q8 = 0.0f;
                        if (got == (ssize_t) row.size()) {
                            dequantize_row_q6_K((const block_q6_K *) row.data(), weights.data(), rd.ne[0]);
                            for (size_t j = 0; j < weights.size(); ++j) ref += weights[j] * normalized.hc_output[j];
                            for (size_t j = 0; j < weights.size(); ++j) ref_q8 += weights[j] * q8_f32[j];
                        }
                        const float streamed = projection.top_logits[r];
                        const float abs_err = std::fabs(ref - streamed);
                        const float rel_err = abs_err / std::max(1.0e-12f, std::fabs(ref));
                        std::fprintf(stderr, "CORTEX_CHAT Q6K_REF token=%d ref=%g q8_ref=%g streamed=%g abs=%g rel=%g q8_abs=%g bytes=%zd\n",
                            tok, ref, ref_q8, streamed, abs_err, rel_err, std::fabs(ref_q8 - streamed), got);
                    }
                }
            }
            llama_token token = (llama_token) projection.token;
            if (sampler && projection.top_count > 0) {
                std::vector<llama_token_data> cur_candidates;
                cur_candidates.reserve(projection.top_count);
                for (size_t r = 0; r < projection.top_count; ++r) {
                    cur_candidates.push_back({ (llama_token) projection.top_tokens[r], projection.top_logits[r], 0.0f });
                }
                llama_token_data_array cur_p = {
                    /* .data     = */ cur_candidates.data(),
                    /* .size     = */ cur_candidates.size(),
                    /* .selected = */ -1,
                    /* .sorted   = */ true,
                };
                llama_sampler_apply(sampler, &cur_p);
                if (cur_p.selected >= 0 && cur_p.selected < (int64_t) cur_p.size) {
                    token = cur_p.data[cur_p.selected].id;
                }
                llama_sampler_accept(sampler, token);
            } else if (sampler) {
                llama_sampler_accept(sampler, token);
            }
            return token;
        }
        llama_batch_free(batch);
        const size_t state_size = llama_state_get_size(ctx);
        {
            std::vector<uint8_t> tmp_state(state_size);
            if (llama_state_get_data(ctx, tmp_state.data(), state_size) != state_size) {
                std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_serialize_intermediate expected=%zu\n", il, state_size);
                std::fflush(stderr);
                llama_free(ctx);
                return -1;
            }
            std::string state_file = cortex_get_state_dir() + "/layer_" + std::to_string(il) + ".bin";
            std::ofstream sf(state_file, std::ios::binary | std::ios::trunc);
            if (!sf.write(reinterpret_cast<const char *>(tmp_state.data()), (std::streamsize) state_size)) {
                std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_write_file expected=%zu\n", il, state_size);
                std::fflush(stderr);
                llama_free(ctx);
                return -1;
            }
        }
        cortex_log_diag("CORTEX_CHAT STATE layer=%d kind=serialized bytes=%zu saved_position=%lld restored_position=%lld seq=0\n",
                        il, state_size, (long long) (pos + n_tokens - 1), (long long) pos);
        llama_free(ctx);
        if (il + 1 < n_layer && !llama_cortex_activate_layer(model, il + 1)) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=activate_next_layer\n", il + 1);
            std::fflush(stderr);
            return -1;
        }
    }
    std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=-1 reason=range_exhausted\n");
    std::fflush(stderr);
    return -1;
}

int main(int argc, char ** argv) {
    if (argc < 2) return 2;
    std::string model_path;
    int max_generate = 8;
    bool debug_flag = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug" || arg == "--diag" || arg == "-v") {
            debug_flag = true;
            setenv("CORTEX_DEBUG", "1", 1);
        } else if (model_path.empty()) {
            model_path = arg;
        } else {
            max_generate = std::atoi(arg.c_str());
        }
    }
    if (model_path.empty()) return 2;

    if (!cortex_chat_is_debug()) {
        llama_log_set([](ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fputs(text, stderr);
                fflush(stderr);
            }
        }, nullptr);
    }

    // Establish the authoritative bounded selector before model loading.  The
    // loader uses this contract to exclude dense global tensors; the chat
    // graph updates the same range for each physical-layer transaction.
    setenv("CORTEX_FLASHNEXT_EXTERNAL_EMBEDDING", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_START", "0", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_END", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD", "1", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY", "1", 1);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 3;
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD");
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY");
    if (std::getenv("CORTEX_FLASHNEXT_LOADER_ONLY") != nullptr) {
        std::fprintf(stderr, "CORTEX_LOADER_ONLY=PASS model_loaded=1 context_created=0 decode=0 graph_compute=0 ssm_conv=0\n");
        llama_free_model(model);
        return 0;
    }
    cortex_clean_state_dir();
    const int n_embd = llama_model_n_embd(model);
    std::vector<std::vector<uint8_t>> states; // states are persisted to disk in cortex_get_state_dir()
    std::vector<float> hidden(n_embd);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    std::vector<llama_chat_message> messages;
    std::vector<std::string> message_storage;
    message_storage.reserve(128);
    std::vector<char> formatted(8192);
    std::string line; llama_pos pos = 0; int prev_len = 0;
    std::cout << "Qwenny\nType /exit to quit.\n\nYou: " << std::flush;
    while (std::getline(std::cin, line)) {
        if (line == "/exit") break;
        message_storage.push_back(line);
        messages.push_back({"user", message_storage.back().c_str()});
        const char * tmpl = llama_model_chat_template(model, nullptr);
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len >= (int) formatted.size()) {
            formatted.resize((size_t) new_len + 1);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) { std::fprintf(stderr, "Qwenny: failed to apply chat template\n"); break; }
        std::string formatted_prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
        int n = -llama_tokenize(vocab, formatted_prompt.c_str(), (int32_t) formatted_prompt.size(), nullptr, 0, pos == 0, true);
        std::vector<llama_token> prompt_tokens(n);
        if (llama_tokenize(vocab, formatted_prompt.c_str(), (int32_t) formatted_prompt.size(), prompt_tokens.data(), n, pos == 0, true) < 0) return 4;
        if (cortex_chat_is_debug() || std::getenv("CORTEX_LOGIT_DEBUG") != nullptr) {
            std::fprintf(stderr, "CORTEX_PROMPT_TOKENS n=%zu:", prompt_tokens.size());
            for (llama_token tok : prompt_tokens) std::fprintf(stderr, " %d", tok);
            std::fprintf(stderr, "\n");
        }
        llama_token next = -1;
        if (std::getenv("CORTEX_BOUNDED_PROMPT") != nullptr && prompt_tokens.size() > 1) {
            std::fprintf(stderr, "CORTEX_BOUNDED_PROMPT begin tokens=%zu position=%lld\n",
                         prompt_tokens.size(), (long long) pos);
            for (size_t pi = 0; pi < prompt_tokens.size(); ++pi) {
                next = process_positions(model, states, { prompt_tokens[pi] }, pos, hidden, sampler);
                if (next < 0) {
                    std::fprintf(stderr, "CORTEX_BOUNDED_PROMPT fail token_index=%zu position=%lld\n",
                                 pi, (long long) pos);
                    return 5;
                }
                std::fprintf(stderr, "CORTEX_BOUNDED_PROMPT token_index=%zu position=%lld token=%d\n",
                             pi, (long long) pos, prompt_tokens[pi]);
                ++pos;
            }
            std::fprintf(stderr, "CORTEX_BOUNDED_PROMPT end position=%lld\n", (long long) pos);
        } else {
            next = process_positions(model, states, prompt_tokens, pos, hidden, sampler);
        }
        if (next < 0) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=MAIN_EXIT result=FAIL exit_code=5 reason=process_positions\n");
            std::fflush(stderr);
            return 5;
        }
        if (std::getenv("CORTEX_BOUNDED_PROMPT") == nullptr || prompt_tokens.size() <= 1) {
            pos += (llama_pos) prompt_tokens.size();
        }
        if (prompt_tokens.size() > 1) {
            hidden.erase(hidden.begin(), hidden.end() - n_embd);
        }
        std::string response;
        for (int i = 0; i < max_generate; ++i) {
            char piece[256]; int len = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, false);
            cortex_log_diag("CORTEX_CHAT FEEDBACK position=%lld token=%d valid=%s piece=%.*s\n",
                            (long long) pos, next, llama_vocab_is_eog(vocab, next) || next >= 0 ? "YES" : "NO",
                            std::max(0, len), len > 0 ? piece : "");
            if (len > 0) { std::cout.write(piece, len) << std::flush; response.append(piece, len); }
            if (llama_vocab_is_eog(vocab, next)) break;
            llama_token feedback = next;
            next = process_positions(model, states, {feedback}, pos++, hidden, sampler);
            if (next < 0) return 6;
        }
        message_storage.push_back(response);
        messages.push_back({"assistant", message_storage.back().c_str()});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        std::cout << "\n\nYou: " << std::flush;
    }
    llama_sampler_free(sampler);
    llama_model_free(model); llama_backend_free();
    cortex_clean_state_dir();
    return 0;
}
