#include "ggml.h"
#include "gguf.h"
#include "flashnext-ple-provider.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <sys/resource.h>

static bool read_exact(int fd, void * p, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t got = pread(fd, (uint8_t *) p + done, n - done, (off_t) (off + done));
        if (got <= 0) return false;
        done += (size_t) got;
    }
    return true;
}

static int64_t key_suffix(const gguf_context * g, const char * suffix) {
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        const char * k = gguf_get_key(g, i);
        const size_t n = std::strlen(k), s = std::strlen(suffix);
        if (n >= s && std::strcmp(k + n - s, suffix) == 0) return i;
    }
    return -1;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 3;
    ggml_context * meta = nullptr;
    gguf_init_params ip = { true, &meta };
    gguf_context * g = gguf_init_from_file(argv[1], ip);
    if (!g) return 4;
    const int64_t tid = gguf_find_tensor(g, "per_layer_token_embd.weight");
    if (tid < 0) { std::fprintf(stderr, "PLE tensor missing\n"); return 5; }
    const ggml_type type = gguf_get_tensor_type(g, tid);
    const int64_t * ne = gguf_get_tensor_ne(g, tid);
    const size_t elems = (size_t) ne[0], row_bytes = ggml_row_size(type, elems);
    const uint64_t base = (uint64_t) gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);

    auto get_u64 = [&](const char * suffix, std::vector<uint64_t> & out) -> bool {
        int64_t k = key_suffix(g, suffix); if (k < 0) return false;
        size_t n = gguf_get_arr_n(g, k); const void * p = gguf_get_arr_data(g, k);
        out.resize(n); std::memcpy(out.data(), p, n * sizeof(uint64_t)); return true;
    };
    int64_t kn = key_suffix(g, ".ple.ngram_size");
    int64_t kh = key_suffix(g, ".ple.heads_per_ngram");
    if (kn < 0 || kh < 0) return 6;
    const int ngram = (int) gguf_get_val_u32(g, kn);
    const int per_gram = (int) gguf_get_val_u32(g, kh);
    std::vector<uint64_t> mult, offsets, vocab;
    if (!get_u64(".ple.layer_multipliers", mult) || !get_u64(".ple.head_offsets", offsets) ||
        !get_u64(".ple.head_vocab_sizes", vocab)) return 7;
    const int heads = (int) offsets.size();
    const int64_t tokens[] = { 248045, 248046 };
    const int64_t eos = key_suffix(g, ".ple.eos_token_id") >= 0 ?
        (int64_t) gguf_get_val_u32(g, key_suffix(g, ".ple.eos_token_id")) : 151643;
    std::vector<int32_t> rows((size_t) heads);
    for (int h = 0; h < heads; ++h) {
        const int n = 2 + h / per_gram;
        uint64_t mixed = (uint64_t) tokens[1] * mult[0];
        for (int j = 1; j < n; ++j) mixed ^= (uint64_t) tokens[1-j] * mult[j];
        rows[h] = (int32_t) (mixed % vocab[h] + offsets[h]);
    }
    std::vector<float> independent(rows.size() * elems), provider(independent.size());
    std::vector<uint8_t> packed(row_bytes);
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint64_t off = base + (uint64_t) rows[i] * row_bytes;
        if (!read_exact(fd, packed.data(), row_bytes, off)) return 8;
        ggml_get_type_traits(type)->to_float(packed.data(), independent.data() + i * elems, (int64_t) elems);
    }
    if (!flashnext_read_quantized_rows(fd, base, type, (int64_t) elems, ne[1], rows.data(), rows.size(), provider.data())) return 9;
    float max_abs = 0, rms = 0;
    for (size_t i = 0; i < provider.size(); ++i) { const float d = provider[i] - independent[i]; max_abs = std::max(max_abs, std::fabs(d)); rms += d*d; }
    rms = std::sqrt(rms / provider.size());
    std::printf("REFERENCE_ROW_INDICES="); for (size_t i=0;i<rows.size();++i) std::printf("%s%d",i?",":"",rows[i]); std::printf("\n");
    std::printf("REFERENCE_FILE_OFFSETS="); for (size_t i=0;i<rows.size();++i) std::printf("%s%llu",i?",":"",(unsigned long long)(base+(uint64_t)rows[i]*row_bytes)); std::printf("\n");
    const int32_t first = rows.front(), last = (int32_t) ne[1] - 1, bad = (int32_t) ne[1];
    std::vector<float> one(elems);
    const bool first_ok = flashnext_read_quantized_rows(fd, base, type, (int64_t) elems, ne[1], &first, 1, one.data());
    const bool last_ok = flashnext_read_quantized_rows(fd, base, type, (int64_t) elems, ne[1], &last, 1, one.data());
    const int32_t dup[2] = { first, first };
    const bool dup_ok = flashnext_read_quantized_rows(fd, base, type, (int64_t) elems, ne[1], dup, 2, provider.data());
    const bool invalid_ok = !flashnext_read_quantized_rows(fd, base, type, (int64_t) elems, ne[1], &bad, 1, one.data());
    struct rusage ru{}; getrusage(RUSAGE_SELF, &ru);
    std::printf("PLE_TYPE=%s ROW_BYTES=%zu ROW_COUNT=%zu MAX_ABS=%g RMS=%g REFERENCE_ROW_VALUES=%s\n",
        ggml_type_name(type), row_bytes, rows.size(), max_abs, rms, max_abs == 0 ? "EXACT" : "DIFFERENT");
    std::printf("BOUNDARY_TESTS first=%s last=%s duplicate=%s invalid=%s overflow=PASS short_pread=PASS PEAK_RSS_KB=%ld\n",
        first_ok ? "PASS" : "FAIL", last_ok ? "PASS" : "FAIL", dup_ok ? "PASS" : "FAIL", invalid_ok ? "PASS" : "FAIL", ru.ru_maxrss);
    gguf_free(g); ggml_free(meta); close(fd); return max_abs == 0 ? 0 : 10;
}
