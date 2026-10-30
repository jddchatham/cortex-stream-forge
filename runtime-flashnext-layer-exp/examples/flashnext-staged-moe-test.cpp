#include "flashnext-staged-moe.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    flashnext_staged_moe staged;
    std::string error;
    assert(staged.begin(0, 0, error));
    assert(!staged.begin(0, 0, error));
    const bool ok = staged.run(
        [](flashnext_layer_transaction & tx, std::string &) {
            for (size_t i = 0; i < tx.pre_moe_hidden.size(); ++i) tx.pre_moe_hidden[i] = float(i % 17) * .125f;
            return true;
        },
        [](flashnext_layer_transaction & tx, std::string &) {
            for (size_t i = 0; i < tx.complete_moe.size(); ++i) {
                tx.selected_moe[i] = tx.pre_moe_hidden[i] * 2.0f;
                tx.shared_moe[i] = 1.0f;
                tx.complete_moe[i] = tx.selected_moe[i] + tx.shared_moe[i];
            }
            return true;
        },
        [](const flashnext_layer_transaction & tx, std::string &) {
            for (size_t i = 0; i < tx.layer_output.size(); ++i)
                if (tx.layer_output[i] != 0.0f) return false;
            return true;
        }, error);
    assert(ok);
    assert(staged.close(error));
    assert(!staged.close(error));
    std::cout << "staged FlashNext transaction PASS\n";
}
