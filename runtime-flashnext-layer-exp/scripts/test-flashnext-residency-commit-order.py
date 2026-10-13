#!/usr/bin/env python3
"""Static non-inference guard for INVALID -> bind -> VALID activation order."""
from pathlib import Path

source = Path(__file__).parents[1] / "src" / "llama-model.cpp"
text = source.read_text()
start = text.index("bool llama_model_base::cortex_activate_layer")
end = text.index("ggml_tensor * llama_model_base::create_tensor", start)
body = text[start:end]

assert "cortex_residency_valid = false" in body
assert "cortex_staged_names = std::move(next)" in body
bind = body.index("cortex_rebind_layer(il, true)")
publish = body.index("cortex_selected_names = std::move(pimpl->cortex_staged_names)")
valid = body.index("cortex_residency_valid = true", publish)
assert bind < publish < valid
assert "cortex_selected_names.clear()" in body
print("FLASHNEXT_RESIDENCY_COMMIT_ORDER PASS")
print("INVALID -> internal bind/validate -> manifest publish -> VALID")
