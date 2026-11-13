#include "common.h"
#include "../qies110/range_source.h"
#include <sys/stat.h>
using namespace a109;
int main(int argc,char**argv)try{
 if(argc!=2)q104::real("QIES111 synthetic fixture construction only");q104::synthetic();need(std::string(argv[1])=="0"||std::string(argv[1])=="15","fixed synthetic profiles only");int profile=std::stoi(argv[1]);const std::string file="/outputs/tensors.bin";int fd=open(file.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);need(fd>=0,"exclusive fixture create");
 auto writeall=[&](const void*b,size_t n){auto*p=(const char*)b;while(n){auto v=write(fd,p,n);need(v>0,"fixture write");p+=v;n-=v;}};char header[64]{};memcpy(header,"QIES110_SYNTHETIC_TENSOR_DATA_V1",30);writeall(header,64);uint64_t offset=64;
 struct Item{io110::Entry entry;};std::vector<Item>items;
 for(auto&[name,s]:io110::schema(profile)){
  if(offset%32){std::array<char,32>zero{};size_t n=32-offset%32;writeall(zero.data(),n);offset+=n;}
  io110::Entry e{name,"",int(s[0]),s[1],s[2],offset,io110::row_size(s[0],s[1])*s[2]};Digest hash;
  if(e.type==0){std::vector<float>v(e.columns);if(name=="attn_vres_lambda.weight")v[0]=-.375f;else v=gamma(e.columns,(name=="attn_q_norm.weight"||name=="post_attention_norm.weight"||name=="post_ffw_norm.weight")?1:0);writeall(v.data(),e.bytes);hash.add(v.data(),e.bytes);}
  else{int projection=name=="attn_q.weight"||name=="ffn_gate.weight"?0:name=="attn_k.weight"||name=="ffn_up.weight"?1:name=="attn_v.weight"||name=="ffn_down.weight"?2:name=="attn_gate.weight"?3:4;InputBlocks source(ggml_type(e.type),projection);std::vector<uint8_t>row(io110::row_size(e.type,e.columns));for(uint64_t r=0;r<e.rows;++r){source.row(r,e.columns,row.data());writeall(row.data(),row.size());hash.add(row.data(),row.size());}}
  offset+=e.bytes;e.hash=hash.finish();items.push_back({e});
 }
 need(fsync(fd)==0,"fixture fsync");close(fd);struct stat stat{};need(::stat(file.c_str(),&stat)==0,"fixture stat");auto id=io110::id(stat);std::ofstream plan("/outputs/manifest");plan<<"QIES110-SOURCE-1\nSYNTHETIC\n"<<profile<<"\n\"/fixtures/tensors.bin\"";for(auto n:id)plan<<' '<<n;plan<<"\n15\n";for(auto&i:items){auto&e=i.entry;plan<<std::quoted(e.name)<<' '<<e.type<<' '<<e.columns<<' '<<e.rows<<' '<<e.offset<<' '<<e.bytes<<' '<<e.hash<<'\n';}plan.close();need(bool(plan),"fixture manifest write");std::cout<<"Created bounded-memory synthetic tensor container, bytes="<<offset<<"; no model read\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
