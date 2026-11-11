// Test-only trap: any GGML context/allocation before rejection is a test FAILURE.
#include <unistd.h>
#include <cstddef>
namespace { [[noreturn]] void bad(){const char s[]="QIES104_TEST_FAILURE: unapproved GGML allocation\n";write(2,s,sizeof(s)-1);_exit(97);} }
extern "C" {struct Params{size_t size;void*buffer;bool no_alloc;};void*ggml_init(Params){bad();}void*ggml_backend_alloc_ctx_tensors(void*,void*){bad();}void*ggml_backend_alloc_buffer(void*,size_t){bad();}}
