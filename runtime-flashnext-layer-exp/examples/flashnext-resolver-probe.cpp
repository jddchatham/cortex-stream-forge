#include "../src/flashnext-tensor-index.h"
#include <cstdio>
#include <string>
int main(int argc,char **argv) {
    if(argc!=2)return 2; flashnext_tensor_index idx; std::string e;
    if(!idx.load(argv[1],&e)){std::fprintf(stderr,"%s\n",e.c_str());return 3;}
    const int ids[]={136,77,317,257,226,142,175,414,373,131};
    for(int id:ids) for(auto p:{flashnext_projection::gate,flashnext_projection::up,flashnext_projection::down}) {
        flashnext_tensor_descriptor d; if(!idx.resolve_expert(0,id,p,d,&e)){std::fprintf(stderr,"%s\n",e.c_str());return 4;}
        std::printf("expert=%d tensor=%s shard=%s offset=%llu bytes=%llu type=%d\n",id,d.name.c_str(),d.shard.c_str(),(unsigned long long)d.offset,(unsigned long long)d.bytes,d.type);
    }
    std::puts("FLASHNEXT DYNAMIC EXPERT RESOLVER: PASS"); return 0;
}
