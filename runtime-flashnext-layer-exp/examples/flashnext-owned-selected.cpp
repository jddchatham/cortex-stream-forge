#include "../src/flashnext-selected-moe.h"
#include "../src/flashnext-shared-expert.h"
#include "../src/flashnext-segment-b.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

static uint64_t hash_f32(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { uint32_t u; std::memcpy(&u, p + i, 4); h ^= u; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    // hidden.f32 router-artifact.f32 shard.gguf, 30 expert offsets, 4 shared offsets
    if (argc != 38) return 2;
    flashnext_layer_transaction tx;
    std::ifstream hf(argv[1], std::ios::binary), rf(argv[2], std::ios::binary);
    if (!hf || !rf) return 3;
    hf.read(reinterpret_cast<char *>(tx.pre_moe_hidden.data()), tx.pre_moe_hidden.size() * sizeof(float));
    std::vector<float> route(248320); rf.read(reinterpret_cast<char *>(route.data()), route.size() * sizeof(float));
    if (!hf || !rf) return 4;
    std::memcpy(tx.residual_hc.data(), route.data() + 2560, tx.residual_hc.size() * sizeof(float));
    std::memcpy(tx.inject.data(), route.data() + 2560 + tx.residual_hc.size(), tx.inject.size() * sizeof(float));
    std::array<int, 512> order{};
    for (int i = 0; i < 512; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 10, order.end(),
        [&route](int a, int b) { return route[12804 + a] > route[12804 + b]; });
    float mx = route[12804 + order[0]];
    double z = 0.0;
    std::array<double, 10> ex{};
    for (int i = 0; i < 10; ++i) {
        ex[i] = std::exp(static_cast<double>(route[12804 + order[i]]) - static_cast<double>(mx));
        z += ex[i];
    }
    for (int i = 0; i < 10; ++i) { tx.selected_expert_ids[i] = order[i]; tx.routing_weights[i] = ex[i] / z; }
    std::printf("weights="); for (int i = 0; i < 10; ++i) std::printf("%s%.12g", i ? "," : "", tx.routing_weights[i]); std::printf("\n");

    const int fd = ::open(argv[3], O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 5;
    std::array<flashnext_selected_expert_slice, 10> slices{};
    for (int i = 0; i < 10; ++i) {
        slices[i].fd = fd;
        for (int r = 0; r < 3; ++r) slices[i].offsets[r] = std::strtoull(argv[4 + i*3 + r], nullptr, 0);
    }
    if (!flashnext_execute_selected_moe(slices, tx, 16 * 1024)) { ::close(fd); return 6; }
    flashnext_shared_expert_slices shared;
    shared.fd = fd;
    shared.gate_offset = std::strtoull(argv[34], nullptr, 0);
    shared.up_offset = std::strtoull(argv[35], nullptr, 0);
    shared.down_offset = std::strtoull(argv[36], nullptr, 0);
    shared.sigmoid_offset = std::strtoull(argv[37], nullptr, 0);
    if (!flashnext_execute_shared_expert(shared, tx)) { ::close(fd); return 7; }
    for (size_t i = 0; i < tx.complete_moe.size(); ++i) {
        tx.complete_moe[i] = tx.selected_moe[i] + tx.shared_moe[i];
    }
    if (!flashnext_execute_segment_b(tx)) { ::close(fd); return 8; }
    std::printf("CORTEX FLASHNEXT OWNED SELECTED MOE: PASS\n");
    std::printf("ids="); for (int i = 0; i < 10; ++i) std::printf("%s%d", i ? "," : "", tx.selected_expert_ids[i]);
    std::printf("\nselected_hash=%016llx selected_bytes=%llu unselected=%llu window=%llu pread_calls=%llu\n",
        (unsigned long long) hash_f32(tx.selected_moe.data(), tx.selected_moe.size()),
        (unsigned long long) tx.io.selected_expert_bytes_read,
        (unsigned long long) tx.io.unselected_expert_reads,
        (unsigned long long) tx.io.tensor_window_bytes,
        (unsigned long long) tx.io.pread_calls);
    std::printf("CORTEX FLASHNEXT OWNED SHARED EXPERT: PASS shared_hash=%016llx\n",
        (unsigned long long) hash_f32(tx.shared_moe.data(), tx.shared_moe.size()));
    std::printf("CORTEX FLASHNEXT OWNED COMPLETE MOE: PASS complete_hash=%016llx\n",
        (unsigned long long) hash_f32(tx.complete_moe.data(), tx.complete_moe.size()));
    std::printf("CORTEX FLASHNEXT OWNED SEGMENT B: PASS segment_b_hash=%016llx\n",
        (unsigned long long) hash_f32(tx.layer_output.data(), tx.layer_output.size()));
    ::close(fd);
    return 0;
}
