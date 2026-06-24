"""
model.py — load a Qwen2 model + tokenizer.

Phase 0 loads the whole (small) model in every process and each stage uses only
its layer block. Selective shard loading (so a stage on one L4 holds only its
layers of a 32B) is a Phase 3 task — it doesn't affect correctness, only memory.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_model(model_id: str, dtype=None, device: str = "cpu", attn_implementation=None):
    # GPU: fp16 + device_map (required for quantized AWQ/GPTQ — can't .to() after
    # load). CPU: fp32 + .to(). dtype defaults by device unless overridden.
    # attn_implementation: e.g. "sdpa" — tree drafting passes a 4D (non-causal) mask to
    # the draft, which flash-attn can't take; sdpa can.
    if dtype is None:
        dtype = torch.float16 if device != "cpu" else torch.float32
    kw = dict(dtype=dtype)
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **kw)
    return model.eval()


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)


def _shard_device_map(config, start: int, end: int, keep_head: bool,
                      gpu: str, other: str):
    """device_map placing only owned layers (+ head if keep_head) on the GPU."""
    n = config.num_hidden_layers
    dm = {
        "model.embed_tokens": gpu if keep_head else other,
        "model.norm": gpu if keep_head else other,
        "model.rotary_emb": gpu,            # tiny, no weights; our stage needs it
        "lm_head": gpu if keep_head else other,
    }
    for i in range(n):
        dm[f"model.layers.{i}"] = gpu if (start <= i < end) else other
    return dm


def load_model_shard_bnb(model_id: str, start: int, end: int, keep_head: bool = False,
                         device: str = "cuda:0"):
    """Selective shard load for a 4-bit (bitsandbytes NF4) model: a device_map places
    ONLY the owned layers (+ embed/norm/lm_head if keep_head) on the GPU and every other
    layer on `meta` — so un-owned layers are never materialized and only this stage's
    ~share of the 4-bit weights lands in VRAM.

    This is what lets a model too big for one card (e.g. a 70B) be split across GPUs.
    AWQ/GPTQ can't shard this way: their Marlin kernel's post_init runs on every layer
    and requires each to be on cuda (un-owned on meta/cpu -> "Expected a cuda device").
    bitsandbytes has no such constraint. Proven bitwise-correct on a GPU (the sharded
    stage output matches the full model). Requires `bitsandbytes` installed.

    IMPORTANT: shard-loading only works from a *fp16* checkpoint (quantize-at-load — the
    un-owned layers are never materialized). A *pre-quantized* bnb 4-bit checkpoint can
    NOT be sharded: bnb deserializes every stored 4-bit tensor, including un-owned ones
    bound to meta -> "Bnb4bitDeserialize ... uint8 on meta". So point this at the fp16
    repo (e.g. Qwen/Qwen2.5-72B-Instruct), not a *-bnb-4bit one.
    """
    from transformers import BitsAndBytesConfig
    config = AutoConfig.from_pretrained(model_id)
    dm = _shard_device_map(config, start, end, keep_head=keep_head, gpu=device, other="meta")
    kwargs = dict(device_map=dm, torch_dtype=torch.float16)
    # If the checkpoint is ALREADY 4-bit (a pre-quantized bnb repo), its config carries the
    # quantization_config — the uint8 weights load as-is. Passing our own bnb config would
    # try to RE-quantize them ("expected a floating-point dtype, but got uint8"). So only
    # attach a bnb config to quantize a *fp16* checkpoint at load time.
    if not getattr(config, "quantization_config", None):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model.eval()


def load_model_shard(model_id: str, start: int, end: int, keep_head: bool = False,
                     device: str = "cuda:0", other_device: str = "cpu"):
    """Selective shard load: build the FULL model skeleton on `meta` (no weights),
    replace every module this stage does NOT own with Identity (so its weights are
    never read or materialized), then load ONLY the owned layers' (+ head, if
    keep_head) tensors from the checkpoint's safetensors shards onto `device`.

    Peak memory is ~this stage's share, not the whole model — which is what lets a
    model too large to load whole on one card be split across machines (load-then-
    prune can't help: the whole model won't even fit to be pruned). Owned layers
    keep their GLOBAL layer_idx, so `stage_for_range(model, start, end)` works
    unchanged afterwards (the un-owned slots are Identity, never indexed).

    Works for standard (non-quantized) checkpoints. Quantized AWQ/GPTQ needs
    quant-aware layer construction (the dense `from_config` skeleton builds plain
    Linear layers, so AWQ qweight/scales/qzeros won't bind) — that path is built on
    top of this and validated on a GPU separately; see docs/ARCHITECTURE.md.
    """
    from accelerate import init_empty_weights, load_checkpoint_in_model
    from huggingface_hub import snapshot_download

    config = AutoConfig.from_pretrained(model_id)
    dtype = torch.float16 if str(device) != "cpu" else torch.float32

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model = model.to(dtype)

    inner = model.model
    n = config.num_hidden_layers
    for i in range(n):
        if not (start <= i < end):
            inner.layers[i] = torch.nn.Identity()      # un-owned: drop the module
    if not keep_head:
        inner.embed_tokens = torch.nn.Identity()
        inner.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()

    # Load only the tensors the now-pruned skeleton still has parameters for. The
    # un-owned modules are Identity (no params), so their checkpoint tensors are
    # simply skipped — never read, never placed on `device`.
    ckpt = snapshot_download(model_id)
    load_checkpoint_in_model(model, ckpt, device_map={"": device}, dtype=dtype)

    # small models tie lm_head to the embedding — re-tie after loading so the head
    # is correct (only relevant when this stage keeps the head)
    if keep_head and getattr(config, "tie_word_embeddings", False):
        model.tie_weights()

    # Rotary inv_freq is a computed (non-persistent) buffer, not in the checkpoint,
    # so it's still on `meta` — rebuild the rotary module on-device so its forward
    # works.
    rope_cls = type(inner.rotary_emb)
    try:
        inner.rotary_emb = rope_cls(config=config).to(device)
    except TypeError:
        inner.rotary_emb = rope_cls(config).to(device)

    # Materialize any other leftover meta buffers (defensive) as empty on-device.
    for mod in model.modules():
        for bname, buf in list(mod._buffers.items()):
            if buf is not None and buf.is_meta:
                mod._buffers[bname] = torch.zeros(buf.shape, dtype=buf.dtype, device=device)

    return model.eval()


def prune_to_layers(model, start: int, end: int, keep_head: bool = True):
    """Free the VRAM of layers/head this stage doesn't own.

    A stage holding layers [start, end) doesn't need the others, and a
    non-coordinator stage needs neither embedding nor lm_head. We replace the
    unneeded modules with Identity and empty the CUDA allocator so their weights
    are freed: peak memory during load is the whole (4-bit) model, steady-state
    is ~this stage's share — letting two L4s comfortably hold a 32B with KV room.

    The kept layers retain their *global* layer_idx (StageKV handles the offset),
    so stage_for_range(model, start, end) works unchanged afterward.
    """
    import gc
    inner = model.model
    for i in range(len(inner.layers)):
        if not (start <= i < end):
            inner.layers[i] = torch.nn.Identity()
    if not keep_head:
        inner.embed_tokens = torch.nn.Identity()
        inner.norm = torch.nn.Identity()
        model.lm_head = torch.nn.Identity()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
