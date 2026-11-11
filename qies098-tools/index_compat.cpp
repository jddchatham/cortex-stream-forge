#include "flashnext-tensor-index.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sys/resource.h>
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    rlimit memory{512ULL*1024*1024,512ULL*1024*1024};if(setrlimit(RLIMIT_AS,&memory))return 3;
    rlimit cpu{30,30};if(setrlimit(RLIMIT_CPU,&cpu))return 3;
    flashnext_tensor_index index;std::string error;
    if(!index.load(argv[1],&error))return 4;
    std::ifstream names(argv[2]);std::string name;
    while(std::getline(names,name)) {
        flashnext_tensor_descriptor d;
        if(!index.resolve_named(name,d,&error))return 5;
        std::cout<<d.name<<'\t'<<d.offset<<'\t'<<d.bytes<<'\t'<<d.type<<'\t'<<d.shape0<<'\t'<<d.shape1<<'\n';
    }
    return names.bad()?6:0;
}
