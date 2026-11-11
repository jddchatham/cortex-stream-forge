#include "ricursive-layer.h"
#include "admission.h"
#include <iostream>
struct Faux: r99::Source{void read(const r99::Spec&,size_t,void*,size_t)override{throw std::runtime_error("unexpected fake fixture read");}};
struct Unknown: r99::Source{bool is_model()const override{return true;}void read(const r99::Spec&,size_t,void*,size_t)override{throw std::runtime_error("unexpected data read");}};
int main(int argc,char**argv)try{
 if(argc>1&&std::string(argv[1])=="file"){r99::ModelFile file("/nonexistent-model");return 8;}
 if(argc>1&&std::string(argv[1])=="fake-fixture"){r99::Config c;c.fixture=true;Faux source;r99::execute(c,{},source,{},{},{});return 11;}
 r99::Config c;Unknown source;r99::execute(c,{},source,{}, {},{});return 9;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";if(r99::live_resources()!=0)return 10;return 1;}
