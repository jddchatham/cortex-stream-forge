// Independent upstream GGUF metadata parser + read-only range digest. Data only.
#include "gguf.h"
#include "ggml.h"
#include <openssl/evp.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <memory>
int main(int argc,char**argv)try{
 if(argc!=3)throw std::runtime_error("MODEL LAYER expected");int layer=std::stoi(argv[2]);if(layer<0||layer>=80)throw std::runtime_error("layer range");
 auto*g=gguf_init_from_file(argv[1],{true,nullptr});if(!g)throw std::runtime_error("independent GGUF parse");std::unique_ptr<gguf_context,decltype(&gguf_free)>owner(g,gguf_free);int fd=open(argv[1],O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)throw std::runtime_error("read-only model open");std::unique_ptr<int,void(*)(int*)>closed(&fd,[](int*p){close(*p);});struct stat before{},after{};if(fstat(fd,&before))throw std::runtime_error("model stat");std::vector<char>buffer(1024*1024);int count=0;
 for(int i=0;i<gguf_get_n_tensors(g);++i){std::string name=gguf_get_tensor_name(g,i);if(name.rfind("blk."+std::to_string(layer)+".",0)!=0)continue;uint64_t offset=gguf_get_data_offset(g)+gguf_get_tensor_offset(g,i),bytes=gguf_get_tensor_size(g,i);if(offset>uint64_t(before.st_size)||bytes>uint64_t(before.st_size)-offset)throw std::runtime_error("reference tensor range");EVP_MD_CTX*h=EVP_MD_CTX_new();std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)>hash_owner(h,EVP_MD_CTX_free);if(!h)throw std::runtime_error("digest allocation");EVP_DigestInit_ex(h,EVP_sha256(),nullptr);
 for(uint64_t done=0;done<bytes;){size_t want=std::min<uint64_t>(buffer.size(),bytes-done);ssize_t n=pread(fd,buffer.data(),want,offset+done);if(n<=0)throw std::runtime_error("reference short read");EVP_DigestUpdate(h,buffer.data(),n);done+=n;}
 unsigned char hash[32];unsigned len=0;EVP_DigestFinal_ex(h,hash,&len);std::ostringstream text;for(auto x:hash)text<<std::hex<<std::setw(2)<<std::setfill('0')<<int(x);std::cout<<name<<'\t'<<offset<<'\t'<<bytes<<'\t'<<gguf_get_tensor_type(g,i)<<'\t'<<text.str()<<'\n';++count;
 }
 if(fstat(fd,&after)||before.st_size!=after.st_size||before.st_mtim.tv_sec!=after.st_mtim.tv_sec||before.st_mtim.tv_nsec!=after.st_mtim.tv_nsec||before.st_ctim.tv_sec!=after.st_ctim.tv_sec||before.st_ctim.tv_nsec!=after.st_ctim.tv_nsec)throw std::runtime_error("reference model identity changed");if(count!=15)throw std::runtime_error("reference layer count");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
