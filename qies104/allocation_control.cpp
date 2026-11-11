#include "ggml.h"
int main(){auto*c=ggml_init({4096,nullptr,true});if(c)ggml_free(c);return 0;}
