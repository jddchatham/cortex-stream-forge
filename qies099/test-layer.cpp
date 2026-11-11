#ifdef NDEBUG
#undef NDEBUG
#endif
#include "ricursive-layer.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <sys/resource.h>

namespace {
struct Fixture final : r99::Source {
    std::map<std::string, std::vector<uint8_t>> data;
    int reads = 0, fail_read = -1;
    void read(const r99::Spec & s, size_t offset, void * target, size_t n) override {
        if (++reads == fail_read) throw std::runtime_error("injected fixture read error");
        const auto & bytes = data.at(s.name);
        if (offset > bytes.size() || n > bytes.size() - offset) throw std::runtime_error("fixture read bounds");
        std::memcpy(target, bytes.data() + offset, n);
    }
};
r99::Config tiny() {
    r99::Config c; c.fixture = true; c.hidden = 8; c.ffn = 32; c.heads = 4; c.kv_heads = 2; c.head = 4; c.context = 8; c.window = 2; return c;
}
Fixture weights(std::vector<r99::Spec> & specs, ggml_type type = GGML_TYPE_F32, bool zeros = false) {
    Fixture result;
    for (auto & s : specs) {
        bool vector = s.rows == 1;
        s.type = vector ? GGML_TYPE_F32 : type;
        s.bytes = ggml_row_size(s.type, s.columns) * s.rows;
        std::vector<float> values(s.columns * s.rows);
        bool lambda = s.name.find("lambda") != std::string::npos;
        for (size_t i = 0; i < values.size(); ++i) values[i] = lambda ? 0 : vector ? 1.0f : zeros ? 0.0f : (float) std::sin((double) (i + 1) * 0.317) * 0.03f;
        auto & bytes = result.data[s.name]; bytes.resize(s.bytes);
        std::vector<float> importance(s.columns, 1.0f);
        size_t written = ggml_quantize_chunk(s.type, values.data(), bytes.data(), 0, s.rows, s.columns, importance.data());
        assert(written == bytes.size());
    }
    return result;
}
std::vector<float> input(int n, int position = 0) {
    std::vector<float> v(n); for (int i = 0; i < n; ++i) v[i] = (float) ((i + position) % 11 - 5) * 0.1f; return v;
}
template<class F> void rejects(F f) { bool failed = false; try { f(); } catch (const std::exception &) { failed = true; } assert(failed); }
bool same(const r99::KV & a, const r99::KV & b) { return a.owner == b.owner && a.position == b.position && a.key == b.key && a.value == b.value; }
}
int main() {
    int groups = 0;
    {
        auto c = tiny(); auto specs = r99::schema(c); r99::validate_specs(c, specs);
        auto bad = specs; bad[1].columns++; rejects([&] { r99::validate_specs(c, bad); });
        bad = specs; bad[1] = bad[0]; rejects([&] { r99::validate_specs(c, bad); });
        bad = specs; bad[0].type = GGML_TYPE_F16; rejects([&] { r99::validate_specs(c, bad); });
        bad = specs; bad[1].bytes--; rejects([&] { r99::validate_specs(c, bad); });
        c.fixture = false; rejects([&] { c.validate(); }); ++groups;
    }
    {
        auto c = tiny(); auto m = r99::mask(c, 3); assert(std::isinf(m[0]) && m[2] == 0 && m[3] == 0 && std::isinf(m[4]));
        c.layer = 4; m = r99::mask(c, 3); assert(m[0] == 0 && m[3] == 0 && std::isinf(m[4]));
        rejects([&] { r99::mask(c, 8); }); ++groups;
    }
    {
        auto c = tiny(); auto specs = r99::schema(c); auto source = weights(specs, GGML_TYPE_F32, true);
        auto x = input(c.hidden); r99::KV original;
        auto result = r99::execute(c, specs, source, x, x, original, true);
        assert(result.output == x && original.position == 0 && original.key.empty());
        assert(result.next.position == 1 && result.kv_bytes == 8192);
        assert(result.sequence.front() == "attn_norm" && result.sequence.back() == "l_out");
        assert(result.captures.at("ffn_swiglu").size() == (size_t) c.ffn);
        assert(r99::live_resources() == 0); ++groups;
    }
    {
        auto c = tiny(); auto specs = r99::schema(c); auto source = weights(specs); auto x = input(c.hidden); r99::KV state;
        auto a = r99::execute(c, specs, source, x, x, state, true);
        auto b = r99::execute(c, specs, source, x, x, state, true);
        assert(a.output == b.output && same(a.next, b.next));
        assert(a.captures.at("Qcur") == a.captures.at("Qcur_normed")); // RoPE at position zero.
        std::vector<std::string> expected{"attn_norm","vres_in","Vcur","Qcur_normed","Kcur_normed","Qcur","Kcur","kq_soft_max","attn_out","attn_gate","attn_gated","attn_post_norm","ffn_inp","ffn_norm","ffn_swiglu","ffn_out","ffn_post_norm","l_out"};
        assert(a.sequence == expected); ++groups;
    }
    {
        auto c = tiny(); auto specs = r99::schema(c); auto source = weights(specs); auto x0 = input(c.hidden); r99::KV state;
        for (int i = 0; i < 4; ++i) {
            auto x = input(c.hidden, i); auto a = r99::execute(c, specs, source, x, x0, state, true);
            auto reference = r99::execute(c, specs, source, x, x0, state, true);
            assert(a.output == reference.output && same(a.next, reference.next));
            if (i) assert(a.captures.at("Qcur") != a.captures.at("Qcur_normed"));
            state = a.next;
        }
        assert(state.position == 4); ++groups;
    }
    {
        auto c = tiny(); c.layer = 4; auto specs = r99::schema(c); auto source = weights(specs); auto x = input(c.hidden); r99::KV state;
        for (int i = 0; i < 3; ++i) {
            auto a = r99::execute(c, specs, source, x, x, state, true);
            assert(a.captures.at("Qcur") == a.captures.at("Qcur_normed"));
            state = a.next;
        }
        assert(state.position == 3); ++groups;
    }
    {
        auto c = tiny(); auto specs = r99::schema(c); auto source = weights(specs); auto x = input(c.hidden); auto initial = r99::execute(c, specs, source, x, x, {});
        for (const auto & point : {"after_weight_allocation","after_weight_load","before_compute","after_compute","before_publish"}) {
            auto old = initial.next;
            rejects([&] { r99::execute(c, specs, source, x, x, old, true, point); });
            assert(same(old, initial.next)); assert(r99::live_resources() == 0);
        }
        source.fail_read = source.reads + 3;
        rejects([&] { r99::execute(c, specs, source, x, x, initial.next); }); assert(r99::live_resources() == 0); ++groups;
    }
    {
        auto c = tiny(); auto specs = r99::schema(c); auto source = weights(specs); auto x = input(c.hidden); x[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { r99::execute(c, specs, source, x, x, {}); }); assert(r99::live_resources() == 0);
        r99::KV bad; bad.position = 1;
        rejects([&] { r99::execute(c, specs, source, input(c.hidden), input(c.hidden), bad); }); ++groups;
    }
    {
        r99::Schedule schedule;
        for (int p = 0; p < 2; ++p) for (int l = 0; l < 80; ++l) { schedule.require(l,p); schedule.commit(l,p); }
        assert(schedule.layer == 0 && schedule.position == 2);
        rejects([&] { schedule.commit(1,2); }); assert(schedule.layer == 0 && schedule.position == 2); ++groups;
    }
    {
        auto c = tiny(); auto specs = r99::schema(c); auto source = weights(specs); auto x = input(c.hidden); auto a = r99::execute(c, specs, source, x, x, {});
        char dir[] = "kv-fixture-XXXXXX"; assert(::mkdtemp(dir)); std::string path = std::string(dir) + "/state.bin";
        r99::save_kv(path,c,a.next); auto restored = r99::load_kv(path,c); assert(same(restored,a.next));
        auto b = r99::execute(c,specs,source,x,x,a.next); auto resumed = r99::execute(c,specs,source,x,x,restored);
        assert(b.output == resumed.output && same(b.next,resumed.next));
        rejects([&] { r99::save_kv(path,c,a.next); });
        auto wrong = c; wrong.layer = 4; rejects([&] { r99::load_kv(path,wrong); });
        { std::fstream f(path,std::ios::binary|std::ios::in|std::ios::out); f.seekp(-1,std::ios::end); f.put('\x7f'); }
        rejects([&] { r99::load_kv(path,c); }); ::unlink(path.c_str()); ::rmdir(dir); ++groups;
    }
    {
        auto c = tiny(); c.hidden = 256; c.ffn = 512; c.head = 64; c.heads = 4; c.kv_heads = 2;
        for (auto type : {GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q8_0,GGML_TYPE_IQ4_XS}) {
            auto specs = r99::schema(c); auto source = weights(specs,type); auto x = input(c.hidden);
            auto a = r99::execute(c,specs,source,x,x,{}); auto b = r99::execute(c,specs,source,x,x,{});
            assert(a.output == b.output && same(a.next,b.next)); assert(a.weight_bytes < 2*1024*1024); assert(a.graph_bytes < 16*1024*1024);
            assert(r99::live_resources() == 0);
        }
        ++groups;
    }
    {
        auto c = tiny(); auto specs = r99::schema(c); auto source = weights(specs); auto x = input(c.hidden);
        auto state = r99::execute(c,specs,source,x,x,{}).next;
        auto other = c; other.layer = 4; auto other_specs = r99::schema(other); auto other_source = weights(other_specs);
        rejects([&] { r99::execute(other,other_specs,other_source,x,x,state); });
        auto changed = c; changed.clamp = 10;
        rejects([&] { r99::execute(changed,specs,source,x,x,state); });
        assert(r99::live_resources() == 0); ++groups;
    }
    {
        auto c = tiny(); auto specs = r99::schema(c); auto source = weights(specs);
        auto x = input(c.hidden); auto prior = r99::execute(c,specs,source,x,x,{}).next;
        prior.key[0] = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
        rejects([&] { r99::execute(c,specs,source,x,x,prior); });
        assert(r99::live_resources() == 0); ++groups;
    }
    {
        auto c = tiny(); c.clamp = 0.1f; auto specs = r99::schema(c); auto source = weights(specs);
        for (const auto & spec : specs) if (spec.rows > 1) {
            std::vector<float> huge(spec.columns*spec.rows,100.0f);
            std::memcpy(source.data[spec.name].data(),huge.data(),huge.size()*4);
        }
        std::vector<float> x(c.hidden,1.0f), x0(c.hidden,2.0f);
        auto a = r99::execute(c,specs,source,x,x0,{},true);
        for (auto value : a.captures.at("ffn_swiglu")) assert(value > 0 && value < 0.011f);
        auto off = c; off.clamp = 0;
        auto b = r99::execute(off,specs,source,x,x0,{},true);
        for (auto value : b.captures.at("ffn_swiglu")) assert(value > 1.0f);
        const auto & normalized = a.captures.at("attn_norm");
        for (int i=0;i<c.hidden;++i) assert(a.captures.at("vres_in")[i] == x0[i]+(normalized[i]-x0[i])*0.5f);
        ++groups;
    }
    rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    std::cout << "synthetic_peak_rss_kib=" << usage.ru_maxrss << " live_native_resources=" << r99::live_resources() << "\n";
    std::cout << "PASS " << groups << " synthetic native Ricursive groups; no real model inference\n";
}
