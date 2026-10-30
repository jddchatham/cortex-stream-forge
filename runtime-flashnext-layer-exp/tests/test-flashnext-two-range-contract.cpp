#include "../src/llama-batch.h"

#include <cstdio>
#include <memory>
#include <vector>

// Lightweight lifecycle fixture for the decode integration contract. It does
// not execute a model; it verifies the ordering and ownership invariants that
// the gated llama_context path must preserve.
int main() {
    constexpr size_t count = 10240;
    std::vector<float> layer0(count), layer1(count);
    for (size_t i = 0; i < count; ++i) layer0[i] = (float) i + 0.5f;

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->embd = layer0;
    llama_ubatch batch{};
    batch.n_tokens = 1;
    batch.embd = data->embd.data();
    batch.data = data;

    int prepare = 0;
    int find_slot = 0;
    int next = 0;
    std::vector<int> ranges;
    ++prepare;
    ++find_slot;
    ranges.push_back(0);
    layer1 = batch.data->embd;
    auto range1_data = std::make_shared<llama_ubatch::data_t>(*batch.data);
    range1_data->embd = layer1;
    llama_ubatch range1 = batch;
    range1.data = range1_data;
    range1.embd = range1_data->embd.data();
    range1.token = nullptr;
    ranges.push_back(1);

    if (prepare != 1 || find_slot != 1 || next != 0 || ranges != std::vector<int>({0, 1})) return 1;
    if (range1.data->embd != layer0 || range1.embd != range1.data->embd.data()) return 2;
    range1_data.reset();
    if (!range1.data || range1.embd != range1.data->embd.data()) return 3;
    std::puts("flashnext two-range contract: PASS");
    return 0;
}
