#pragma once
#include "ricursive-layer.h"
#include <array>
namespace r102 {
struct Entry {r99::Spec spec;std::string path,sha256;std::array<uint64_t,5> identity;};
struct Bundle {r99::Config config;std::vector<Entry> entries;bool real=false;};
Bundle read_bundle(const std::string&);
class Reader {
 const Bundle & bundle;
public:
 size_t bytes_read=0,max_read=0;
 explicit Reader(const Bundle & b):bundle(b){}
 void read(const r99::Spec&,size_t,void*,size_t);
};
class LayerSource final:public r99::Source {
 Reader reader;bool real;
public:
 explicit LayerSource(const Bundle&);
 bool is_model()const override{return real;}
 void read(const r99::Spec&s,size_t o,void*d,size_t n)override;
};
std::vector<r99::Spec> specs(const Bundle&);
int open_files();
r99::Result execute_layer(const Bundle&,const std::vector<float>&,const std::vector<float>&,const r99::KV&,bool capture=true);
}
