#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits) {
                const bool trace = std::getenv("CORTEX_MOE_KERNEL_TRACE") != nullptr &&
                    op->op == GGML_OP_MUL_MAT_ID && op->src[0] != nullptr &&
                    std::strstr(op->src[0]->name, "blk.2.ffn_gate_exps.weight") != nullptr;
                if (trace && params->ith == 0) {
                    std::fprintf(stderr,
                        "CORTEX_MOE_EXTRA_ENTRY op=%d name=%s tensor=%p data=%p src0=%p src0_data=%p src0_buffer=%p ne=[%lld,%lld,%lld,%lld] src1=%p src2=%p nth=%d\n",
                        (int) op->op, op->name, (void *) op, op->data,
                        (void *) op->src[0], op->src[0]->data, (void *) op->src[0]->buffer,
                        (long long) op->ne[0], (long long) op->ne[1],
                        (long long) op->ne[2], (long long) op->ne[3],
                        (void *) op->src[1], (void *) op->src[2], params->nth);
                }
                const bool handled = tensor_traits->compute_forward(params, op);
                if (trace && params->ith == 0) {
                    std::fprintf(stderr,
                        "CORTEX_MOE_EXTRA_RETURN handled=%d tensor=%p data=%p first=%g\n",
                        handled ? 1 : 0, (void *) op, op->data,
                        op->data != nullptr ? *(const float *) op->data : 0.0f);
                }
                if (handled) return true;
            }
        }
    }
    return false;
}

bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}
