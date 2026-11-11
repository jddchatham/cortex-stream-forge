#ifdef NDEBUG
#undef NDEBUG
#endif
#include "bundle.h"
#include <cassert>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <functional>
int main(int argc,char**argv){
 assert(argc==3);auto fixture=r102::read_bundle(argv[1]);auto real=r102::read_bundle(argv[2]);int passed=0;
 auto rejects=[&](std::function<void()>fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}assert(failed);assert(r102::open_files()==0&&r99::live_resources()==0);++passed;};
 std::vector<unsigned char>chunk(1024*1024);auto spec=fixture.entries[0].spec;
 {r102::Reader reader(fixture);reader.read(spec,0,chunk.data(),1);assert(reader.bytes_read==1&&reader.max_read==1);++passed;}
 rejects([&]{auto s=spec;s.name="missing";r102::Reader reader(fixture);reader.read(s,0,chunk.data(),1);});
 rejects([&]{auto s=spec;++s.columns;r102::Reader reader(fixture);reader.read(s,0,chunk.data(),1);});
 rejects([&]{auto s=spec;s.type=GGML_TYPE_Q8_0;r102::Reader reader(fixture);reader.read(s,0,chunk.data(),1);});
 rejects([&]{r102::Reader reader(fixture);reader.read(spec,spec.bytes,chunk.data(),1);});
 rejects([&]{r102::Reader reader(fixture);reader.read(spec,0,chunk.data(),1024*1024+1);});
 rejects([&]{auto b=fixture;b.entries[0].identity[1]++;r102::Reader reader(b);reader.read(spec,0,chunk.data(),1);});
 rejects([&]{auto b=fixture;b.entries[0].path="/nonexistent/qies102-fixture";r102::Reader reader(b);reader.read(spec,0,chunk.data(),1);});
 rejects([&]{r102::LayerSource source(real);}); // No gate overrides or real graph execution.
 std::vector<float>x(256,.1f);r99::KV previous;
 rejects([&]{auto b=fixture;b.entries[0].path="/nonexistent/qies102-fixture";r102::execute_layer(b,x,x,previous);});
 assert(previous.position==0&&previous.key.empty());++passed;
 rejects([&]{r102::LayerSource source(fixture);r99::execute(fixture.config,r102::specs(fixture),source,x,x,previous,false,"after_weight_load");});
 std::ifstream input(argv[1]);std::string original((std::istreambuf_iterator<char>(input)),{});
 auto path=std::filesystem::path(argv[1]).parent_path()/"malformed-test.plan";
 auto malformed=[&](std::string data){std::ofstream f(path);f<<data;f.close();rejects([&]{r102::read_bundle(path.string());});std::filesystem::remove(path);};
 malformed("wrong\n");malformed(std::string(200000,'x'));malformed(original.substr(0,original.find_last_of('\n',original.size()-2)+1));
 std::string altered=original;auto where=altered.find("\"blk.0.");assert(where!=std::string::npos);altered.replace(where,7,"\"bad.0.");malformed(altered);
 std::cout<<"PASS "<<passed<<" adapter schema/range/identity/admission/failure-cleanup tests; real numerical execution=false\n";
}
