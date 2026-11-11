#include "admission.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <array>
#include <iomanip>
#include <openssl/evp.h>
namespace q104 {
namespace {
[[noreturn]] void deny(const std::string&s){throw std::runtime_error("QIES104 BLOCKED: "+s);}
std::string digest(const std::string&p){
 std::ifstream f(p,std::ios::binary);if(!f)deny("fixture open failed");
 EVP_MD_CTX*c=EVP_MD_CTX_new();if(!c)deny("digest allocation failed");
 if(EVP_DigestInit_ex(c,EVP_sha256(),nullptr)!=1){EVP_MD_CTX_free(c);deny("digest init");}
 std::array<char,65536>chunk{};
 while(f){f.read(chunk.data(),chunk.size());if(EVP_DigestUpdate(c,chunk.data(),f.gcount())!=1){EVP_MD_CTX_free(c);deny("digest update");}}
 unsigned char out[32];unsigned n=0;bool ok=f.eof()&&EVP_DigestFinal_ex(c,out,&n)==1&&n==32;EVP_MD_CTX_free(c);if(!ok)deny("fixture digest failed");
 std::ostringstream s;for(auto b:out)s<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)b;return s.str();
}
const std::map<std::string,std::string>&catalog(){static const std::map<std::string,std::string>m{
#include "fixture-catalog.inc"
};return m;}
}
void real(const char*entry){
 std::ifstream f("/proc/meminfo");std::string line;unsigned long long available=0;
 while(std::getline(f,line)){std::istringstream row(line);std::string key;unsigned long long value;if(row>>key>>value&&key=="MemAvailable:")available=value*1024;}
 if(available<6ULL*1024*1024*1024)deny(std::string(entry)+": unchanged 6 GiB MemAvailable gate; before model tensor allocation");
 // Fail closed even on a larger host. No receipt or environment flag enables this release.
 deny(std::string(entry)+": real execution locked pending reference/live guard qualification; no model allocation approved");
}
void synthetic(){
 const char*p=std::getenv("QIES104_SYNTHETIC");if(!p||std::string(p)!="1")deny("synthetic interface requires sealed no-model filesystem");
 std::ifstream f("/qies104.synthetic");std::string s;std::getline(f,s);if(s!="QIES104_SEALED_SYNTHETIC_DOMAIN_V1")deny("synthetic namespace marker missing");
}
void fixture_file(const std::string&p){
 synthetic();auto path=std::filesystem::canonical(p).string();auto it=catalog().find(path);
 if(it==catalog().end())deny("unregistered synthetic fixture: "+path);
 if(digest(path)!=it->second)deny("synthetic fixture bytes changed");
}
void fixture_directory(const std::string&p){synthetic();size_t count=0;for(auto&e:std::filesystem::directory_iterator(p)){if(e.is_regular_file()){fixture_file(e.path().string());++count;}}if(!count)deny("empty fixture directory");}
}
