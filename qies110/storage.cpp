#include "range_source.h"
#include "telemetry.h"
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("PLAN hash|schedule");
 {std::ifstream f(argv[1]);std::string a,b;std::getline(f,a);std::getline(f,b);bool real=b=="REAL";io110::File source(argv[1],real);diag106::sample("storage-before",0,0,0,0,0,0,0,0);
 for(auto&[name,t]:source.entries){auto h=io110::hash_tensor(source,name);io110::require(h==t.hash,"tensor SHA256 mismatch");std::cout<<"{\"tensor\":\""<<name<<"\",\"bytes\":"<<t.bytes<<",\"sha256\":\""<<h<<"\",\"real_storage_only\":"<<(real?"true":"false")<<"}\n";
 if(std::string(argv[2])=="schedule"){std::vector<uint8_t>row(io110::row_size(t.type,t.columns));for(uint64_t r=0;r<t.rows;++r)source.row(name,int(r),int(t.columns),row.data());}}
 std::cout<<"{\"bytes_read\":"<<source.bytes_read<<",\"max_request\":"<<source.max_request<<",\"cache_peak\":"<<source.cache_peak<<",\"arithmetic\":false}\n";}
 diag106::sample("storage-after-cleanup",0,0,0,0,0,0,0,io110::open_files);return io110::open_files==0?0:2;
 }catch(const std::exception&e){std::cerr<<e.what()<<" open_files="<<io110::open_files<<'\n';return io110::open_files==0?1:2;}}
