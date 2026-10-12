#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
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
        const bool rebound = il == 0 ? llama_cortex_rebind_layer(model, il) : llama_cortex_activate_layer(model, il);
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
        llama_set_embeddings_layer_inp(ctx, il + 1, true);
        if (!states[il].empty() && llama_state_set_data(ctx, states[il].data(), states[il].size()) == 0) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_restore\n", il);
            std::fflush(stderr);
            return -1;
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
            batch.logits[i] = (i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;
        const int decode_rc = llama_decode(ctx, batch);
        if (decode_rc != 0) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=decode return_code=%d\n", il, decode_rc);
            std::fflush(stderr);
            return -1;
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
            float * logits = llama_get_logits(ctx);
            if (!logits) {
                std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=logits_output\n", il);
                std::fflush(stderr);
                return -1;
            }
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            if (sampler) {
                if (std::getenv("CORTEX_LOGIT_DEBUG") != nullptr) {
                    std::vector<int> ids(n_vocab);
                    std::iota(ids.begin(), ids.end(), 0);
                    std::partial_sort(ids.begin(), ids.begin() + std::min(8, n_vocab), ids.end(),
                        [logits](int a, int b) { return logits[a] > logits[b]; });
                    std::fprintf(stderr, "CORTEX_FINAL_LOGITS pos=%lld positions=%d vocab=%d top:", (long long) (pos + n_tokens - 1), n_tokens, n_vocab);
                    for (int i = 0; i < std::min(8, n_vocab); ++i) std::fprintf(stderr, " %d=%.8g", ids[i], logits[ids[i]]);
                    uint64_t hash = 1469598103934665603ULL;
                    for (int i = 0; i < n_vocab; ++i) {
                        uint32_t bits; std::memcpy(&bits, &logits[i], sizeof(bits));
                        hash ^= bits; hash *= 1099511628211ULL;
                    }
                    std::fprintf(stderr, " hash=%016llx\n", (unsigned long long) hash);
                }
                llama_token token = llama_sampler_sample(sampler, ctx, -1);
                if (std::getenv("CORTEX_LOGIT_DEBUG") != nullptr) std::fprintf(stderr, "CORTEX_SAMPLED token=%d\n", token);
                llama_sampler_accept(sampler, token);
                llama_batch_free(batch);
                const size_t state_size = llama_state_get_size(ctx);
                states[il].resize(state_size);
                if (llama_state_get_data(ctx, states[il].data(), state_size) != state_size) {
                    std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_serialize_sampler expected=%zu\n", il, state_size);
                    std::fflush(stderr);
                    return -1;
                }
                llama_free(ctx);
                return token;
            } else {
                const llama_token token = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
                llama_batch_free(batch);
                const size_t state_size = llama_state_get_size(ctx);
                states[il].resize(state_size);
                if (llama_state_get_data(ctx, states[il].data(), state_size) != state_size) {
                    std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_serialize_greedy expected=%zu\n", il, state_size);
                    std::fflush(stderr);
                    return -1;
                }
                llama_free(ctx);
                return token;
            }
        }
        llama_batch_free(batch);
        const size_t state_size = llama_state_get_size(ctx);
        states[il].resize(state_size);
        if (llama_state_get_data(ctx, states[il].data(), state_size) != state_size) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=PROCESS_FAILURE layer=%d reason=state_serialize_intermediate expected=%zu\n", il, state_size);
            std::fflush(stderr);
            return -1;
        }
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
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
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
        llama_token next = process_positions(model, states, prompt_tokens, pos, hidden, sampler);
        if (next < 0) {
            std::fprintf(stderr, "CORTEX_CHAT checkpoint=MAIN_EXIT result=FAIL exit_code=5 reason=process_positions\n");
            std::fflush(stderr);
            return 5;
        }
        pos += (llama_pos) prompt_tokens.size();
        if (prompt_tokens.size() > 1) {
            hidden.erase(hidden.begin(), hidden.end() - n_embd);
        }
        std::string response;
        for (int i = 0; i < max_generate; ++i) {
            char piece[256]; int len = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, false);
            if (len > 0) { std::cout.write(piece, len) << std::flush; response.append(piece, len); }
            if (llama_vocab_is_eog(vocab, next)) break;
            next = process_positions(model, states, {next}, pos++, hidden, sampler);
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
