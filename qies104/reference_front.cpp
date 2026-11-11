#include "admission.h"
#include <iostream>
#include <stdexcept>
#include <unistd.h>
int main(int argc,char**argv)try{
 if(argc>1&&std::string(argv[1])=="--real")q104::real("independent reference real frontend");
 q104::synthetic();
 if(argc!=3)throw std::runtime_error("QIES104 BLOCKED: reference frontend requires pinned synthetic model and output directory");
 q104::fixture_file(argv[1]);
 execl("/stock-engine/reference-engine","reference-engine",argv[1],argv[2],(char*)nullptr);
 throw std::runtime_error("QIES104 BLOCKED: private reference backend unavailable outside sealed domain");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
