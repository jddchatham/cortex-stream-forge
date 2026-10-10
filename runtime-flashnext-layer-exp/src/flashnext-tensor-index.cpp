#include "flashnext-tensor-index.h"

#include <fstream>
#include <algorithm>
#include <regex>
#include <sstream>
#include <vector>

namespace {
struct entry { flashnext_tensor_descriptor d; int64_t shape0=0, shape1=0, shape2=0; };
static std::vector<entry> entries;
static std::string model_dir;
static bool parse_num(const std::string & s, const char * key, uint64_t & v) {
    std::regex r(std::string("\\\"") + key + "\\\"\\s*:\\s*([0-9]+)"); std::smatch m;
    if (!std::regex_search(s,m,r)) return false; v=std::stoull(m[1]); return true;
}
static std::string parse_str(const std::string & s, const char * key) {
    std::regex r(std::string("\\\"") + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\""); std::smatch m;
    return std::regex_search(s,m,r) ? m[1].str() : std::string();
}
static int64_t stride_for(const entry & e) {
    return e.d.bytes / 512;
}
}

bool flashnext_tensor_index::load(const std::string & path, std::string * error) {
    std::ifstream f(path); if (!f) { if(error)*error="cannot open index"; return false; }
    std::stringstream ss; ss << f.rdbuf(); const std::string all=ss.str();
    model_dir=parse_str(all,"model_dir"); entries.clear();
    std::regex obj("\\{\\s*\\\"name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"[\\s\\S]*?\\}", std::regex::icase);
    for (std::sregex_iterator it(all.begin(),all.end(),obj), end; it!=end; ++it) {
        entry e; const std::string s=it->str(); e.d.name=(*it)[1].str(); e.d.shard=parse_str(s,"shard");
        if (!parse_num(s,"offset",e.d.offset) || !parse_num(s,"bytes",e.d.bytes)) continue;
        uint64_t t=0; if (!parse_num(s,"type",t)) continue; e.d.type=(int)t;
        std::regex sh("\\\"shape\\\"\\s*:\\s*\\[\\s*([0-9]+)(?:\\s*,\\s*([0-9]+))?(?:\\s*,\\s*([0-9]+))?"); std::smatch m;
        if (std::regex_search(s,m,sh)) {
            e.shape0 = std::stoll(m[1].str());
            e.shape1 = m[2].matched ? std::stoll(m[2].str()) : 0;
            e.shape2 = m[3].matched ? std::stoll(m[3].str()) : 0;
        }
        e.d.shape0=e.shape0; e.d.shape1=e.shape1; e.d.shape2=e.shape2;
        e.d.row_bytes=e.shape0>0 ? e.d.bytes/(uint64_t) std::max<int64_t>(1,e.shape1*std::max<int64_t>(1,e.shape2)) : 0;
        e.d.block_size=e.shape0>0 ? (uint32_t)e.shape0 : 0;
        e.d.expert_stride = stride_for(e); e.d.path=model_dir+"/"+e.d.shard; entries.push_back(e);
    }
    if (entries.empty()) { if(error)*error="no tensor entries parsed"; return false; }
    return true;
}

static const char * suffix(flashnext_projection p) { return p==flashnext_projection::gate?"ffn_gate_exps":p==flashnext_projection::up?"ffn_up_exps":"ffn_down_exps"; }
static const char * shared_suffix(flashnext_projection p) { return p==flashnext_projection::gate?"ffn_gate_shexp":p==flashnext_projection::up?"ffn_up_shexp":p==flashnext_projection::down?"ffn_down_shexp":"ffn_gate_inp_shexp"; }
static const char * hc_suffix(flashnext_projection p) { return p==flashnext_projection::hc_norm?"hc_attn_norm":p==flashnext_projection::hc_down?"hc_attn_down":p==flashnext_projection::hc_up?"hc_attn_up":"hc_attn_inject"; }

bool flashnext_tensor_index::resolve_expert(int layer, int expert, flashnext_projection p,
                                             flashnext_tensor_descriptor & out, std::string * error) const {
    const std::string n="blk."+std::to_string(layer)+"."+suffix(p)+".weight";
    for (const auto & e:entries) if(e.d.name==n) { if(expert<0||expert>=512) break; out=e.d; out.offset += uint64_t(expert)*uint64_t(e.d.expert_stride); return true; }
    if(error)*error="missing "+n; return false;
}
bool flashnext_tensor_index::resolve_shared(int layer, flashnext_projection p,
                                             flashnext_tensor_descriptor & out, std::string * error) const {
    const std::string n="blk."+std::to_string(layer)+"."+shared_suffix(p)+".weight";
    for (const auto & e:entries) if(e.d.name==n) { out=e.d; return true; }
    if(error)*error="missing "+n; return false;
}
bool flashnext_tensor_index::resolve_hc_attention(int layer, flashnext_projection p,
                                                   flashnext_tensor_descriptor & out, std::string * error) const {
    const std::string n="blk."+std::to_string(layer)+"."+hc_suffix(p)+".weight";
    for (const auto & e:entries) if(e.d.name==n) {
        if (e.d.offset + e.d.bytes < e.d.offset) { if(error)*error="offset overflow"; return false; }
        out=e.d; return true;
    }
    if(error)*error="missing "+n; return false;
}

bool flashnext_tensor_index::resolve_named(const std::string & name,
                                            flashnext_tensor_descriptor & out,
                                            std::string * error) const {
    for (const auto & e : entries) if (e.d.name == name) { out = e.d; return true; }
    if (error) *error = "missing " + name;
    return false;
}
