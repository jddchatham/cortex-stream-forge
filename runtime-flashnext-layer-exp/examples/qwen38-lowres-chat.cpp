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
#include <string>
#include <vector>

static llama_token process_positions(llama_model * model, std::vector<std::vector<uint8_t>> & states,
                                     const std::vector<llama_token> & tokens, llama_pos pos,
                                     std::vector<float> & hidden, llama_sampler * sampler) {
    std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_ENTRY layer=-1 tokens=%zu result=BEGIN\n", tokens.size());
    std::fflush(stderr);
    const int n_layer = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_tokens = (int) tokens.size();
    const int hc = 4;
    std::vector<int32_t> token_ids(tokens.begin(), tokens.end());
    std::vector<float> external_embd;
    external_embd.resize((size_t) n_tokens * n_embd);
    std::fprintf(stderr, "CORTEX_CHAT checkpoint=TOKEN_ROWS_BEFORE layer=-1 tokens=%zu\n", tokens.size());
    std::fflush(stderr);
    if (!llama_cortex_external_token_rows(model, token_ids.data(), token_ids.size(), external_embd.data())) {
        std::fprintf(stderr, "CORTEX_CHAT checkpoint=TOKEN_ROWS_AFTER layer=-1 result=FAIL reason=row_materialization\n");
        std::fflush(stderr);
        return -1;
    }
    std::fprintf(stderr, "CORTEX_CHAT checkpoint=TOKEN_ROWS_AFTER layer=-1 result=PASS\n");
    std::fflush(stderr);
    for (int il = 0; il < n_layer; ++il) {
        char start[32], end[32];
        std::snprintf(start, sizeof(start), "%d", il);
        std::snprintf(end, sizeof(end), "%d", il + 1);
        setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
        setenv("CORTEX_FLASHNEXT_LAYER_START", start, 1);
        setenv("CORTEX_FLASHNEXT_LAYER_END", end, 1);
        std::fprintf(stderr, "CORTEX_CHAT checkpoint=REBIND_BEFORE layer=%d\n", il);
        std::fflush(stderr);
        const bool rebound = llama_cortex_activate_layer(model, il);
        std::fprintf(stderr, "CORTEX_CHAT checkpoint=REBIND_AFTER layer=%d result=%s\n", il, rebound ? "PASS" : "FAIL");
        std::fflush(stderr);
        if (!rebound) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=rebind\n", il);
            std::fflush(stderr);
            return -1;
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 256; cp.n_batch = n_tokens;
        std::fprintf(stderr, "CORTEX_CHAT checkpoint=CONTEXT_BEFORE layer=%d\n", il);
        std::fflush(stderr);
        llama_context * ctx = llama_init_from_model(model, cp);
        std::fprintf(stderr, "CORTEX_CHAT checkpoint=CONTEXT_AFTER layer=%d result=%s\n", il, ctx ? "PASS" : "FAIL");
        std::fflush(stderr);
        if (!ctx) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=context_init\n", il);
            std::fflush(stderr);
            return -1;
        }
        const bool terminal_layer = il + 1 == n_layer;
        if (terminal_layer) llama_set_embeddings(ctx, true);
        else llama_set_embeddings_layer_inp(ctx, il + 1, true);
        if (!states[il].empty() && llama_state_set_data(ctx, states[il].data(), states[il].size()) == 0) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_restore\n", il);
            std::fflush(stderr);
            return -1;
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
            states[il].resize(state_size);
            if (llama_state_get_data(ctx, states[il].data(), state_size) != state_size) {
                llama_free(ctx); return -1;
            }
    std::fprintf(stderr, "CORTEX_CHAT STATE layer=%d kind=serialized bytes=%zu saved_position=%lld restored_position=%lld seq=0\n",
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
            std::fprintf(stderr, "CORTEX_CHAT TOKEN position=%lld id=%d logit=%g bytes=%zu scratch=%zu\n",
                         (long long) (pos + n_tokens - 1), projection.token, projection.logit,
                         projection.bytes_read, projection.scratch_bytes);
            if (pos + n_tokens - 1 == 8) for (size_t r = 0; r < projection.top_count; ++r) {
                const llama_vocab * top_vocab = llama_model_get_vocab(model);
                char piece[128]; int pn = llama_token_to_piece(top_vocab, projection.top_tokens[r], piece, sizeof(piece), 0, false);
                std::fprintf(stderr, "CORTEX_CHAT TOP rank=%zu token=%d logit=%g eog=%d control=%d piece=%.*s\n",
                    r + 1, projection.top_tokens[r], projection.top_logits[r],
                    llama_vocab_is_eog(top_vocab, projection.top_tokens[r]),
                    llama_vocab_is_control(top_vocab, projection.top_tokens[r]), std::max(0, pn), pn > 0 ? piece : "");
            }
            if (pos + n_tokens - 1 == 8) {
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
            if (sampler) llama_sampler_accept(sampler, token);
            return token;
        }
        llama_batch_free(batch);
        const size_t state_size = llama_state_get_size(ctx);
        states[il].resize(state_size);
        if (llama_state_get_data(ctx, states[il].data(), state_size) != state_size) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_serialize_intermediate expected=%zu\n", il, state_size);
            std::fflush(stderr);
            return -1;
        }
        std::fprintf(stderr, "CORTEX_CHAT STATE layer=%d kind=serialized bytes=%zu saved_position=%lld restored_position=%lld seq=0\n",
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
    if (argc < 2 || argc > 3) return 2;
    const int max_generate = argc == 3 ? std::atoi(argv[2]) : 8;
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
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD");
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY");
    if (std::getenv("CORTEX_FLASHNEXT_LOADER_ONLY") != nullptr) {
        std::fprintf(stderr, "CORTEX_LOADER_ONLY=PASS model_loaded=1 context_created=0 decode=0 graph_compute=0 ssm_conv=0\n");
        llama_free_model(model);
        return 0;
    }
    const int n_embd = llama_model_n_embd(model);
    std::vector<std::vector<uint8_t>> states(llama_model_n_layer(model));
    std::vector<float> hidden(n_embd);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.10f, 0.0f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
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
        if (std::getenv("CORTEX_LOGIT_DEBUG") != nullptr) {
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
            std::fprintf(stderr, "CORTEX_CHAT FEEDBACK position=%lld token=%d valid=%s piece=%.*s\n",
                         (long long) pos, next, llama_vocab_is_eog(vocab, next) || next >= 0 ? "YES" : "NO",
                         std::max(0, len), len > 0 ? piece : "");
            if (len > 0) { std::cout.write(piece, len) << std::flush; response.append(piece, len); }
            // For the Iteration-75 continuity test, a requested multi-token
            // run deliberately feeds the first generated ID back once even
            // when it is marked EOG.  Single-token/default generation keeps
            // the normal EOG stop behavior.
            if (llama_vocab_is_eog(vocab, next) && max_generate <= 1) break;
            llama_token feedback = next;
            const char * alternate = std::getenv("CORTEX_CHAT_ALTERNATE_TOKEN");
            if (alternate && i == 0) feedback = (llama_token) std::atoi(alternate);
            if (alternate && i == 0) {
                char alt_piece[256]; int alt_len = llama_token_to_piece(vocab, feedback, alt_piece, sizeof(alt_piece), 0, false);
                std::fprintf(stderr, "CORTEX_CHAT ALTERNATE position=%lld token=%d valid=%s piece=%.*s\n",
                             (long long) pos, feedback, feedback >= 0 && feedback < llama_vocab_n_tokens(vocab) ? "YES" : "NO",
                             std::max(0, alt_len), alt_len > 0 ? alt_piece : "");
            }
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
    return 0;
}
