#include "gguf.h"
#include <map>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 auto*g=gguf_init_from_file(argv[1],{true,nullptr});if(!g)return 3;
 auto i=gguf_find_key(g,"general.architecture");if(i<0){gguf_free(g);return 4;}
 std::map<int,size_t>counts;bool supported=true;
 for(int64_t n=0;n<gguf_get_n_tensors(g);++n){auto t=gguf_get_tensor_type(g,n);counts[t]++;supported &= ggml_get_type_traits(t)->to_float!=nullptr || t==GGML_TYPE_F32;}
 std::cout<<"architecture="<<gguf_get_val_str(g,i)<<" tensor_count="<<gguf_get_n_tensors(g)<<" conversion_traits="<<supported<<" payload_loaded=no\n";
 for(auto[t,n]:counts)std::cout<<"type="<<t<<" name="<<ggml_type_name((ggml_type)t)<<" count="<<n<<'\n';
 gguf_free(g);return supported?0:5;
}
