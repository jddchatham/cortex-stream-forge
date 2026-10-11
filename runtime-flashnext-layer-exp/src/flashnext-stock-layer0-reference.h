#pragma once

#include "flashnext-layer-transaction.h"

// Source-derived reference for the stock Qwen4Exp final per-layer
// hyper-connection combine.  This is deliberately separate from the
// Cortex Segment-B implementation and returns the post-layer residual
// stream, not the final vocabulary head output.
bool flashnext_stock_layer0_reference(const flashnext_layer_transaction & tx,
                                      std::array<float, 10240> & output);
