#include "ggml.h"
#include <vector>
#include <fstream>
#include <cstdlib>
int main(int argc,char**argv){if(argc!=6)return 2;auto t=(ggml_type)std::atoi(argv[1]);int cols=std::atoi(argv[2]),rows=std::atoi(argv[3]);if(cols<=0||cols>512||rows<=0||rows>512)return 3;std::vector<char>b(ggml_row_size(t,cols)*rows);std::ifstream f(argv[4],std::ios::binary);f.read(b.data(),b.size());if(!f)return 4;std::vector<float>out(cols*rows);ggml_get_type_traits(t)->to_float(b.data(),out.data(),out.size());std::ofstream o(argv[5],std::ios::binary);o.write((char*)out.data(),out.size()*4);return !o;}
