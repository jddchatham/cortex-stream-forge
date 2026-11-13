#include "range_source.h"
#include <functional>
#include <iostream>
#include <filesystem>
int injected=0,calls=0;
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int f,void*p,size_t n,off_t o){++calls;if(injected==1){injected=0;errno=EINTR;return -1;}if(injected==2)n=std::min<size_t>(n,17);if(injected==3)return 0;return __real_pread(f,p,n,o);}
int main(int argc,char**argv){try{
 io110::require(argc==2,"scratch directory required");std::filesystem::path root=argv[1];std::filesystem::create_directories(root);std::string data=(root/"sparse-synthetic.bin").string(),plan=(root/"source.plan").string();
 auto schema=io110::schema(0);uint64_t end=64;std::map<std::string,io110::Entry>entries;
 for(auto&[name,s]:schema){end=(end+31)/32*32;io110::Entry e{name,std::string(64,'0'),int(s[0]),s[1],s[2],end,io110::row_size(s[0],s[1])*s[2]};entries[name]=e;end+=e.bytes;}
 int fd=open(data.c_str(),O_CREAT|O_EXCL|O_RDWR,0600);io110::require(fd>=0,"exclusive fixture create");char magic[64]="QIES110_SYNTHETIC_TENSOR_DATA_V1";io110::require(write(fd,magic,64)==64&&ftruncate(fd,end)==0,"sparse fixture initialize");close(fd);
 auto write_plan=[&](std::function<void(std::map<std::string,io110::Entry>&)>change={}){struct stat s{};io110::require(stat(data.c_str(),&s)==0,"stat fixture");auto es=entries;if(change)change(es);std::ofstream f(plan);f<<"QIES110-SOURCE-1\nSYNTHETIC\n0\n"<<std::quoted(data);for(auto n:io110::id(s))f<<' '<<n;f<<"\n15\n";for(auto&[name,e]:es)f<<std::quoted(name)<<' '<<e.type<<' '<<e.columns<<' '<<e.rows<<' '<<e.offset<<' '<<e.bytes<<' '<<e.hash<<'\n';};
 int passed=0;auto test=[&](const char*name,std::function<void()>fn,bool reject){bool threw=false;try{fn();}catch(const std::exception&){threw=true;}injected=0;io110::require(threw==reject&&io110::open_files==0,"fault outcome/cleanup mismatch");++passed;std::cout<<"PASS "<<name<<'\n';};
 write_plan();test("normal bounded read",[&]{io110::File f(plan,false);auto v=f.floats("attn_norm.weight");io110::require(v.size()==5120,"float shape");},false);
 for(int mode:{1,2,3}){write_plan();test(mode==1?"EINTR retry":mode==2?"short-read retry":"zero-read rejection",[&]{io110::File f(plan,false);injected=mode;std::vector<char>b(4096);f.read("attn_q.weight",0,b.data(),b.size());},mode==3);}
 write_plan();test("failed cache refill must not become a cache hit",[&]{io110::File f(plan,false);std::vector<char>b(io110::row_size(12,5120));f.row("attn_q.weight",0,5120,b.data());injected=3;bool threw=false;try{f.row("attn_q.weight",1000,5120,b.data());}catch(const std::exception&){threw=true;}injected=0;io110::require(threw,"refill fault required");int before=calls;f.row("attn_q.weight",1000,5120,b.data());io110::require(calls>before,"retry must reread source after failed refill");},false);
 auto bad_read=[&](const char*name,uint64_t off,size_t count){write_plan();test(name,[&]{io110::File f(plan,false);char b[32]{};f.read("attn_q.weight",off,b,count);},true);};
 bad_read("read overflow",UINT64_MAX,1);bad_read("past tensor end",entries.at("attn_q.weight").bytes,1);bad_read("request exceeds staging budget",0,io110::CHUNK+1);
 write_plan();test("missing tensor",[&]{io110::File f(plan,false);f.get("unknown");},true);
 write_plan();test("row outside tensor",[&]{io110::File f(plan,false);char b[32];f.row("attn_q.weight",10240,5120,b);},true);
 write_plan();test("wrong row width",[&]{io110::File f(plan,false);char b[32];f.row("attn_q.weight",0,5121,b);},true);
 write_plan();test("real-mode mismatch",[&]{io110::File f(plan,true);},true);
 for(auto name:{"overlap","alignment","size overflow","wrong quantization","wrong shape"}){write_plan([&](auto&es){auto&e=es.at("attn_q.weight");std::string n=name;if(n=="overlap")e.offset=es.at("attn_gate.weight").offset;else if(n=="alignment")e.offset+=1;else if(n=="size overflow")e.offset=UINT64_MAX-31;else if(n=="wrong quantization")e.type=13;else e.columns=5119;});test(name,[&]{io110::File f(plan,false);},true);}
 write_plan();test("identity change after open",[&]{io110::File f(plan,false);io110::require(truncate(data.c_str(),end-1)==0,"truncate fixture");char b[4];f.read("attn_q.weight",0,b,4);},true);
 // Recover only our synthetic sparse file; no external/model files are opened.
 io110::require(truncate(data.c_str(),end)==0,"restore fixture size");write_plan();test("path replacement after open",[&]{io110::File f(plan,false);std::filesystem::rename(data,data+".old");std::ofstream g(data);g<<"replacement";g.close();char b[4];f.read("attn_q.weight",0,b,4);},true);
 std::filesystem::remove(data);std::filesystem::rename(data+".old",data);write_plan();std::filesystem::rename(data,data+".old");std::filesystem::create_symlink(data+".old",data);test("symlink source rejection",[&]{io110::File f(plan,false);},true);
 std::filesystem::remove(data);std::filesystem::rename(data+".old",data);std::cout<<"{\"passed\":"<<passed<<",\"open_files\":"<<io110::open_files<<",\"real_weights\":false}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
