#include "ricursive-layer.h"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace {
struct Plan { r99::Config config; std::string model; std::vector<r99::Spec> specs; uint64_t dev, inode, size, mtime, ctime; };
void check(bool ok,const char * why) { if(!ok)throw std::runtime_error(why); }
Plan read_plan(const std::string & path) {
    std::ifstream f(path); check((bool)f,"cannot open plan"); std::string magic,line,sha; Plan p;
    std::getline(f,magic); check(magic=="QIES099-RICURSIVE-PLAN-1","wrong plan format");
    std::getline(f,line); std::istringstream dimensions(line); dimensions>>p.config.layer>>p.config.context; check((bool)dimensions,"bad layer/context"); std::string extra;check(!(dimensions>>extra),"extra config fields");
    std::getline(f,p.model);std::getline(f,sha);
    check(sha=="9016e260f93992f0acc7cb590b012fe21fffddf767b53774387fac1b30ca7bfc","wrong model checksum identity");
    std::getline(f,line);std::istringstream identity(line);identity>>p.dev>>p.inode>>p.size>>p.mtime>>p.ctime;check((bool)identity && !(identity>>extra),"bad file identity");
    while(std::getline(f,line)) {
        std::istringstream row(line);r99::Spec s;int type;
        row>>s.name>>type>>s.columns>>s.rows>>s.offset>>s.bytes;
        check((bool)row && !(row>>extra),"invalid tensor row");s.type=(ggml_type)type;p.specs.push_back(s);
        check(p.specs.size()<=15,"too many layer tensors");
    }
    r99::validate_specs(p.config,p.specs);
    struct stat st{};check(::stat(p.model.c_str(),&st)==0,"model stat failed");
    check((uint64_t)st.st_dev==p.dev && (uint64_t)st.st_ino==p.inode && (uint64_t)st.st_size==p.size && p.size==40426079712ULL &&
          (uint64_t)st.st_mtim.tv_sec*1000000000+st.st_mtim.tv_nsec==p.mtime && (uint64_t)st.st_ctim.tv_sec*1000000000+st.st_ctim.tv_nsec==p.ctime,"model identity changed; reverify SHA256");
    return p;
}
std::vector<float> read_input(const std::string & path,int count) {
    std::ifstream f(path,std::ios::binary);check((bool)f,"input open failed");std::vector<float> out(count);
    f.read((char*)out.data(),out.size()*4);check((bool)f && f.peek()==std::char_traits<char>::eof(),"wrong F32 input size");return out;
}
void write_blob(const std::string & path,const std::vector<float>& values) {
    // The output directory was created exclusively for this attempt.
    std::ofstream f(path,std::ios::binary);check((bool)f,"output create failed");f.write((const char*)values.data(),values.size()*4);f.close();check((bool)f,"output write failed");
}
}
int main(int argc,char **argv) try {
    if(argc==3 && std::string(argv[1])=="--describe") {
        auto p=read_plan(argv[2]);size_t bytes=0;for(auto&s:p.specs)bytes+=s.bytes;
        std::cout<<"layer="<<p.config.layer<<" attention="<<(p.config.sliding()?"sliding":"global")<<" tensors=15 weight_bytes="<<bytes<<" context="<<p.config.context<<" model_payload_read=no\n";return 0;
    }
    if(argc!=7 || std::string(argv[1])!="--run") {
        std::cerr<<"Usage: ricursive-layer --describe PLAN\n       ricursive-layer --run PLAN INPUT.f32 X0.f32 PREVIOUS.kv|- NEW_OUTPUT_DIR\n";return 2;
    }
    if(r99::mem_available()<6ULL*1024*1024*1024) {std::cerr<<"BLOCKED: unchanged 6 GiB MemAvailable gate; no model payload opened\n";return 77;}
    check(std::getenv("QIES099_RESOURCE_SCOPE") && std::string(std::getenv("QIES099_RESOURCE_SCOPE"))=="1","launch only through reviewed unchanged resource-guard adapter");
    auto p=read_plan(argv[2]);auto x=read_input(argv[3],p.config.hidden);auto x0=read_input(argv[4],p.config.hidden);
    r99::KV state;if(std::string(argv[5])!="-")state=r99::load_kv(argv[5],p.config);
    std::string dir=argv[6];check(::mkdir(dir.c_str(),0700)==0,"output directory must be new");
    r99::ModelFile source(p.model);
    // Compare the actual opened inode too, closing the stat/open replacement window.
    struct stat st{};check(::fstat(source.fd,&st)==0 && (uint64_t)st.st_ino==p.inode && (uint64_t)st.st_dev==p.dev &&
      (uint64_t)st.st_size==p.size && (uint64_t)st.st_mtim.tv_sec*1000000000+st.st_mtim.tv_nsec==p.mtime &&
      (uint64_t)st.st_ctim.tv_sec*1000000000+st.st_ctim.tv_nsec==p.ctime,"opened model identity mismatch");
    auto out=r99::execute(p.config,p.specs,source,x,x0,state,true);
    write_blob(dir+"/layer-output.f32",out.output);r99::save_kv(dir+"/next.kv",p.config,out.next);
    for(const auto & capture:out.captures)write_blob(dir+"/"+capture.first+".f32",capture.second);
    std::cout<<"layer="<<p.config.layer<<" input_position="<<state.position<<" next_position="<<out.next.position
             <<" weight_bytes="<<out.weight_bytes<<" graph_buffer_bytes="<<out.graph_bytes<<" kv_bytes="<<out.kv_bytes<<"\n";
    return 0;
} catch(const std::exception & e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
