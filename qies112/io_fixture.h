#pragma once
// INPUT definition only. Neither worker shares graph construction or a projector.
#include "../qies111/common.h"
#include <chrono>
namespace f112 {
using namespace a109;
constexpr int V=100352;
constexpr float SCALE=71.55417633056640625f;
inline double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}
inline int token(const char*s){std::string v=s;need(v.size()>0&&v.size()<8&&v.find_first_not_of("0123456789")==std::string::npos,"token ID syntax");int t=std::stoi(v);need(t>=0&&t<V,"token ID range");return t;}
struct EmbeddingBlocks {
 size_t block_bytes=ggml_row_size(GGML_TYPE_IQ4_XS,256);std::vector<uint8_t>bytes;
 EmbeddingBlocks():bytes(257*block_bytes){std::vector<float>x(256),importance(256,1.f);for(int c=0;c<257;++c){for(int i=0;i<256;++i)x[i]=fixture107::block_value(9,c,i);need(ggml_quantize_chunk(GGML_TYPE_IQ4_XS,x.data(),bytes.data()+c*block_bytes,0,1,256,importance.data())==block_bytes,"IQ4_XS fixture block");}++live;}
 ~EmbeddingBlocks(){--live;}
 void row(int id,void*out)const{need(id>=0&&id<V,"embedding lookup ID");for(int b=0;b<H/256;++b)memcpy((char*)out+b*block_bytes,bytes.data()+fixture107::code(9,id,b)*block_bytes,block_bytes);}
};
inline std::vector<float> input(){return supplied("input.f32");}
inline void setf(ggml_tensor*t,const std::vector<float>&x){set(t,x);}
inline void timing(const char*op,double init,double compute){std::cout<<"{\"operation\":\""<<op<<"\",\"initialization_including_synthetic_weight_fill_seconds\":"<<init<<",\"compute_seconds\":"<<compute<<"}\n";}
inline int choose(const std::vector<float>&v){need(v.size()==V,"logit count");int best=0;for(int i=0;i<V;++i){need(std::isfinite(v[i]),"nonfinite logits");if(v[i]>v[best])best=i;}return best;}
inline void selected(Captures&out,const std::vector<float>&v){int32_t id=choose(v);raw(out,"selected.i32",&id,4);}
inline void clean_error(const std::exception&e){std::cerr<<e.what()<<'\n';mark("after-failure-cleanup");summary();}
}
