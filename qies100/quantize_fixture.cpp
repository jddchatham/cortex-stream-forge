#include "ggml.h"
#include <vector>
#include <fstream>
#include <cstdlib>
int main(int argc,char**argv){if(argc!=7)return 2;auto type=(ggml_type)std::atoi(argv[1]);int cols=std::atoi(argv[2]),rows=std::atoi(argv[3]);if(cols<=0||cols>512||rows<=0||rows>512)return 3;std::vector<float>x(cols*rows),im(cols,1);std::ifstream f(argv[4],std::ios::binary);f.read((char*)x.data(),x.size()*4);if(!f)return 4;std::vector<char>bytes(ggml_row_size(type,cols)*rows);auto n=ggml_quantize_chunk(type,x.data(),bytes.data(),0,rows,cols,im.data());if(n!=bytes.size())return 5;std::ofstream out(argv[5],std::ios::binary);out.write(bytes.data(),bytes.size());auto traits=ggml_get_type_traits(type);std::vector<float>decoded(x.size());traits->to_float(bytes.data(),decoded.data(),decoded.size());std::ofstream d(argv[6],std::ios::binary);d.write((char*)decoded.data(),decoded.size()*4);return !out||!d;}
