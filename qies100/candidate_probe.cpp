#include "ricursive-layer.h"
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include <cstring>
struct Fixture: r99::Source {
 std::string root;
 void read(const r99::Spec&s,size_t off,void*p,size_t n)override{std::ifstream f(root+"/"+s.name+".bin",std::ios::binary);f.seekg(off);f.read((char*)p,n);if(!f)throw std::runtime_error("fixture read");}
};
int main(int argc,char**argv)try{
 if(argc!=3)return 2;
 Fixture source;source.root=argv[1];std::string root=argv[2];std::filesystem::create_directory(root);std::vector<r99::KV>states(5);
 for(int p=0;p<4;++p){std::vector<float>x(256);for(int i=0;i<256;++i)x[i]=float((i+p)%11-5)*.1f;const auto x0=x;
 for(int l=0;l<5;++l){r99::Config c;c.fixture=true;c.hidden=256;c.ffn=512;c.heads=4;c.kv_heads=2;c.head=64;c.context=256;c.window=2;c.layer=l;
 auto specs=r99::schema(c);for(auto &s:specs){std::ifstream tf(source.root+"/"+s.name+".type");int type;if(tf>>type){s.type=(ggml_type)type;s.bytes=ggml_row_size(s.type,s.columns)*s.rows;}}auto out=r99::execute(c,specs,source,x,x0,states[l],true);states[l]=out.next;x=out.output;
 for(auto&[name,v]:out.captures){std::ofstream f(root+"/p"+std::to_string(p)+"-"+name+"-"+std::to_string(l)+".bin",std::ios::binary);f.write((char*)v.data(),v.size()*4);}
 }}std::cout<<"PASS candidate synthetic five layers, four positions\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
