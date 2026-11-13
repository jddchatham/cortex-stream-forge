#include "common.h"
using namespace a109;
int main(int argc,char**)try{
 if(argc!=1)q104::real("QIES109 fixture check only");q104::synthetic();mark("before");
 for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q8_0,GGML_TYPE_Q6_K})for(int columns:{5120,10240,40960}){
  InputBlocks blocks(type,1);std::vector<float>full(columns),importance(columns,1.f);size_t expected=ggml_row_size(type,columns);std::vector<uint8_t>from_blocks(expected),independent(expected+64,0xAD);
  for(int row:{0,17,1023}){
   blocks.row(row,columns,from_blocks.data());for(int b=0;b<columns/256;++b)for(int i=0;i<256;++i)full[b*256+i]=fixture107::block_value(1,fixture107::code(1,row,b),i);
   need(ggml_quantize_chunk(type,full.data(),independent.data(),0,1,columns,importance.data())==expected,"whole-row byte count");need(memcmp(from_blocks.data(),independent.data(),expected)==0,"block/whole-row fixture mismatch");for(size_t i=expected;i<independent.size();++i)need(independent[i]==0xAD,"fixture buffer canary");
  }
  std::cout<<"{\"fixture_type\":"<<type<<",\"columns\":"<<columns<<",\"bytes\":"<<expected<<",\"three_rows_exact\":true,\"canary_intact\":true}\n";
 }
 mark("after-cleanup");summary();return 0;
}catch(const std::exception&e){return failed(e);}
