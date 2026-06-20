"""Introspect the Qwen2 transformers API so the split matches it exactly."""
import inspect
import torch
import transformers
from transformers.models.qwen2 import modeling_qwen2 as M

print("transformers", transformers.__version__, "| torch", torch.__version__)
print("\n=== Qwen2DecoderLayer.forward ===")
print(inspect.signature(M.Qwen2DecoderLayer.forward))
print("\n=== Qwen2Model.forward ===")
print(inspect.signature(M.Qwen2Model.forward))

src = inspect.getsource(M.Qwen2Model.forward)
print("\n=== features present in Qwen2Model.forward ===")
for kw in ["rotary_emb", "position_embeddings", "_update_causal_mask",
           "cache_position", "create_causal_mask", "make_flex_block",
           "past_key_values", "DynamicCache"]:
    present = "yes" if kw in src else "no"
    print(f"  {kw:24s}: {present}")

print("\n=== how layers are called (the for-loop body) ===")
for line in src.splitlines():
    s = line.strip()
    if "decoder_layer(" in s or "layer_outputs" in s or "for decoder_layer" in s or "in self.layers" in s:
        print("  " + s)

print("\n=== default attn implementation ===")
try:
    cfg = transformers.AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    print("  config._attn_implementation:", getattr(cfg, "_attn_implementation", "?"))
except Exception as e:
    print("  (config fetch failed, hub access?):", repr(e)[:200])
