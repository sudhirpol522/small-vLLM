"""Step 6: integrate the existing quantization/ files into TinyServe serve.py.

Run from your TinyServe repository's root:

    python integrate_serve_quantization.py serve.py

It saves a backup named serve.py.before_quantization and modifies serve.py.
It deliberately makes *only* the model construction/weight-loading changes:
none of the scheduler, attention math, KV-cache, or CUDA graph code is edited.

Why a patcher rather than distributing a whole new serve.py? This protects
TinyServe's 2300+ lines of code and modifies only the verified source blocks.
The script rejects unexpected source versions rather than patching blindly.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(src: str, old: str, new: str, description: str) -> str:
    """Refuse accidental double patches or changing the wrong function."""
    count = src.count(old)
    if count != 1:
        raise RuntimeError(
            f"Cannot safely patch {description}: expected one exact source "
            f"block, found {count}. Compare your serve.py with TinyServe main."
        )
    return src.replace(old, new, 1)


def transform(source: str) -> str:
    """Return an updated serve.py, never modifying the caller's input."""
    if 'from quantization.config import parse_quant_config' in source:
        raise RuntimeError('serve.py already imports quantization; refusing a second patch')

    # 1. IMPORTS: serve.py is the entry point. These imports refer to the five
    # small modules we previously created; none imports vLLM at module import
    # time. vLLM is loaded lazily by MarlinBackend on an actual quantized GPU.
    source = replace_once(
        source,
        'from transformers import AutoModelForCausalLM\n',
        '''from transformers import AutoModelForCausalLM

# Step 6: quantization *policy* (config) is separate from packed weights
# (GPTQ/AWQ) and from execution (Marlin). A dense model uses only the
# DenseLinearMethod; installing vLLM is not required for that path.
from quantization.config import parse_quant_config
from quantization.linear import Linear, make_linear
from quantization.gptq import GPTQMethod
from quantization.awq import AWQMethod
from quantization.marlin import MarlinBackend
''',
        'quantization imports',
    )

    # 2. PROJECTION FACTORY: there are seven linear projections per decoder
    # layer. Put the format choice in *one* place to avoid seven if/else trees.
    # Example: q_proj of a GPTQ model gets GPTQMethod + MarlinBackend, but
    # a checkpoint-excluded q_proj gets an ordinary dense weight.
    source = replace_once(
        source,
        'class Attention(nn.Module):\n',
        '''def build_projection(in_features, out_features, *, bias, name, quant_config):
    """Choose how ONE named projection stores/executes its weight matrix.

    Example: name='layers.0.self_attn.q_proj', K=N=4096:
      dense checkpoint -> parameter `weight` with shape [4096,4096].
      GPTQ INT4       -> `qweight` with shape [512,4096].
      AWQ INT4        -> `qweight` with shape [4096,512].

    Attention.forward() still does exactly `self.q_proj(x)` in all cases.
    """
    # The checkpoint may intentionally skip specific projections, so choose
    # per layer name, not merely per model. An excluded layer stays dense.
    quantized = (
        quant_config is not None
        and quant_config.should_quantize_layer(name)
    )

    method = None
    if quantized:
        # Same Marlin backend for GPTQ and AWQ on A100/H100. What differs is
        # the checkpoint's packed layout, handled by GPTQMethod/AWQMethod.
        backend = MarlinBackend()
        if quant_config.method == "gptq":
            method = GPTQMethod(quant_config, backend=backend)
        elif quant_config.method == "awq":
            method = AWQMethod(quant_config, backend=backend)
        else:
            raise ValueError(f"Unsupported quantization: {quant_config.method}")

    # `make_linear` creates checkpoint-shaped *parameters now*, before loading.
    # It delegates to DenseLinearMethod when method=None.
    return make_linear(
        in_features, out_features,
        bias=bias,
        name=name,
        quant_config=quant_config,
        quant_method=method,
    )


class Attention(nn.Module):
''',
        'projection factory',
    )

    # 3. ATTENTION: propagate metadata to constructor but keep all actual
    # attention calculations in forward() unchanged. Names exactly match
    # what TinyServe's existing loader gets after stripping model. prefix.
    source = replace_once(
        source,
        '    def __init__(self, cfg, layer_idx: int):\n        super().__init__()\n',
        '    def __init__(self, cfg, layer_idx: int, quant_config=None):\n        super().__init__()\n',
        'Attention constructor',
    )
    source = replace_once(
        source,
        '''        self.q_proj = nn.Linear(cfg.hidden_size, self.H * self.D, bias=qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias)
        self.o_proj = nn.Linear(self.H * self.D, cfg.hidden_size, bias=False)
''',
        '''        # These are the four LINEAR parts of attention. q/k/v and output
        # shapes do not change when the weights are compressed; only storage
        # and multiplication change. Example: x=[10,4096] -> q=[10,4096].
        prefix = f"layers.{layer_idx}.self_attn"
        self.q_proj = build_projection(
            cfg.hidden_size, self.H * self.D, bias=qkv_bias,
            name=f"{prefix}.q_proj", quant_config=quant_config,
        )
        self.k_proj = build_projection(
            cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias,
            name=f"{prefix}.k_proj", quant_config=quant_config,
        )
        self.v_proj = build_projection(
            cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias,
            name=f"{prefix}.v_proj", quant_config=quant_config,
        )
        self.o_proj = build_projection(
            self.H * self.D, cfg.hidden_size, bias=False,
            name=f"{prefix}.o_proj", quant_config=quant_config,
        )
''',
        'Attention projections',
    )

    # 4. MLP: same decision as attention. MLP.forward() remains completely
    # unchanged: F.silu(gate_proj(x)) * up_proj(x), then down_proj(...).
    source = replace_once(
        source,
        '''class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
''',
        '''class MLP(nn.Module):
    def __init__(self, cfg, layer_idx: int, quant_config=None):
        super().__init__()
        # Same checkpoint/Marlin decision as attention, now for the 3 MLP
        # projections. E.g. gate_proj: [hidden_size -> intermediate_size].
        prefix = f"layers.{layer_idx}.mlp"
        self.gate_proj = build_projection(
            cfg.hidden_size, cfg.intermediate_size, bias=False,
            name=f"{prefix}.gate_proj", quant_config=quant_config,
        )
        self.up_proj = build_projection(
            cfg.hidden_size, cfg.intermediate_size, bias=False,
            name=f"{prefix}.up_proj", quant_config=quant_config,
        )
        self.down_proj = build_projection(
            cfg.intermediate_size, cfg.hidden_size, bias=False,
            name=f"{prefix}.down_proj", quant_config=quant_config,
        )
''',
        'MLP projections',
    )

    # 5. PROPAGATE THE CONFIG: we don't store this as a global variable:
    # explicit constructor parameters make each model's settings obvious.
    source = replace_once(
        source,
        '''class DecoderLayer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.self_attn = Attention(cfg, i)
        self.mlp = MLP(cfg)
''',
        '''class DecoderLayer(nn.Module):
    def __init__(self, cfg, i, quant_config=None):
        super().__init__()
        # Pass the per-model policy to each projection factory. For example,
        # layer 0 and layer 1 both get GPTQMethod, but hold separate weights.
        self.self_attn = Attention(cfg, i, quant_config)
        self.mlp = MLP(cfg, i, quant_config)
''',
        'DecoderLayer propagation',
    )
    source = replace_once(
        source,
        '''class Model(nn.Module):
    def __init__(self, cfg, max_position: int):
        super().__init__()

        self.cfg = cfg
''',
        '''class Model(nn.Module):
    def __init__(self, cfg, max_position: int, quant_config=None):
        super().__init__()

        self.cfg = cfg
        self.quant_config = quant_config  # None for regular dense checkpoints.
''',
        'Model constructor',
    )
    source = replace_once(
        source,
        'self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))',
        'self.layers = nn.ModuleList(DecoderLayer(cfg, i, quant_config) for i in range(cfg.num_hidden_layers))',
        'Model layer creation',
    )

    # 6. LOADER: the critical lifecycle:
    #       parse -> construct -> load -> validate -> repack -> return
    # A missing qweight is *dangerous*: torch.empty would be uninitialized.
    # Treat checkpoint metadata g_idx explicitly. Don't ignore it blindly.
    before_loader = '''def load_model(name: str, dtype, max_position: int):
    """quick helper method to load weights into our model"""
    path = name if os.path.isdir(name) else snapshot_download(
        name, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    cfg = AutoConfig.from_pretrained(path)
    torch.set_default_dtype(dtype)
    with torch.device("cuda"):
        model = Model(cfg, max_position)
    torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, "pt", device="cuda") as fh:
                for key in fh.keys():
                    name_ = key.removeprefix("model.")
                    if name_ in params:
                        params[name_].copy_(fh.get_tensor(key))
    return model.eval()
'''

    after_loader = '''def _validate_gptq_g_idx(name, tensor, modules):
    """Only accept GPTQ `g_idx` when it describes ordinary static grouping.

    Example: group_size=128 means input positions 0..127 map to group 0,
    positions 128..255 map to group 1, etc. A shuffled g_idx would require
    activation-order-aware kernels, which our v1 implementation rejects.
    """
    owner_name = name.removesuffix(".g_idx")
    owner = modules.get(owner_name)
    if not isinstance(owner, Linear) or not isinstance(owner.method, GPTQMethod):
        raise ValueError(f"Unexpected GPTQ g_idx for {owner_name!r}")
    K = owner.in_features
    group_size = owner.method.config.group_size
    if tensor.ndim != 1 or tensor.numel() != K or tensor.dtype not in (
        torch.int32, torch.int64,
    ):
        raise ValueError(f"Invalid GPTQ g_idx shape/dtype for {owner_name}")
    expected = (
        torch.zeros(K, dtype=torch.int64)
        if group_size == -1 else torch.arange(K, dtype=torch.int64) // group_size
    )
    if not torch.equal(tensor.to(dtype=torch.int64, device="cpu"), expected):
        raise ValueError(
            f"GPTQ g_idx for {owner_name} is not standard static grouping; "
            "activation ordering isn't supported by TinyServe v1"
        )


@torch.inference_mode()
def load_model(name: str, dtype, max_position: int):
    """Construct checkpoint-matched weights, load strictly, then prepare Marlin.

    Why this order? Model() allocates *shapes* before the loader can copy data.
    Marlin repacking uses the *values*, so it must happen AFTER loading and
    BEFORE ModelRunner's existing memory profiling / CUDA-graph capture.
    """
    path = name if os.path.isdir(name) else snapshot_download(
        name, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"])
    cfg = AutoConfig.from_pretrained(path)

    # Dense checkpoint -> None. GPTQ/AWQ -> validated policy with bits/groups.
    # Example: GPTQ group_size=128, bits=4 -> pack_factor=8.
    quant_config = parse_quant_config(cfg)

    # Preserve the caller's default dtype even on failure. TinyServe normally
    # requests BF16, but INT4 qweight/qzeros explicitly allocate int32.
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("cuda"):
            model = Model(cfg, max_position, quant_config=quant_config)
    finally:
        torch.set_default_dtype(original_dtype)

    params = dict(model.named_parameters())
    modules = dict(model.named_modules())
    loaded = set()
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors model weights found in {path}")

    # Read checkpoint tensors from CPU and copy only one at a time to GPU.
    # This avoids an extra GPU copy of every safetensors shard during loading.
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                param_name = key.removeprefix("model.")
                value = fh.get_tensor(key)

                if param_name.endswith(".g_idx"):
                    # GPTQ sometimes saves input-group metadata which is not a
                    # model parameter. Validate it; don't silently drop it.
                    _validate_gptq_g_idx(param_name, value, modules)
                    continue

                if param_name not in params:
                    # Older checkpoints may serialize non-persistent RoPE
                    # frequencies. TinyServe recomputes its RoPE table.
                    if param_name.endswith(".rotary_emb.inv_freq"):
                        continue
                    # A tied model has NO separate lm_head parameter: it uses
                    # embed_tokens.weight in Model.logits().
                    if param_name == "lm_head.weight" and model.tie:
                        continue
                    if quant_config is None:
                        # Match old TinyServe's permissive handling of extra
                        # dense keys. Quantized checkpoints must be stricter.
                        continue
                    raise ValueError(f"Unexpected checkpoint tensor {key!r}")

                if param_name in loaded:
                    raise ValueError(f"Duplicate checkpoint tensor {key!r}")
                target = params[param_name]

                # Unlike packed integers, floating-point checkpoint tensors
                # can be intentionally converted from FP16 to TinyServe BF16.
                # Example: scales FP16 on disk -> BF16 on GPU.
                if param_name.endswith((".qweight", ".qzeros")):
                    if value.dtype != torch.int32:
                        raise TypeError(f"{key} must be int32, got {value.dtype}")
                elif value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise TypeError(f"{key} has unsupported dtype {value.dtype}")

                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError(
                        f"{key}: checkpoint shape {tuple(value.shape)} does "
                        f"not match layer shape {tuple(target.shape)}"
                    )

                target.copy_(value)
                loaded.add(param_name)
                # Do not keep the on-disk tensor alive unnecessarily.
                del value

    missing = set(params) - loaded
    if missing:
        raise RuntimeError(
            "Checkpoint missing parameters: " + ", ".join(sorted(missing)[:20])
        )

    # Drop references to old checkpoint-layout nn.Parameters BEFORE repacking.
    # Otherwise `params` keeps qweight/scales/qzeros alive after the Marlin
    # backend replaces them, inflating peak GPU memory and starving the KV cache.
    del params
    if "target" in locals():
        del target

    # Visit all linear layers only AFTER every safetensors file was loaded.
    # Dense layers are no-ops; GPTQ/AWQ methods transform loaded packed data.
    for module in model.modules():
        if isinstance(module, Linear):
            module.process_weights_after_loading()

    return model.eval()
'''
    source = replace_once(source, before_loader, after_loader, 'strict loader')
    return source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path, help='Path to TinyServe serve.py')
    ap.add_argument('--output', type=Path, default=None,
                    help='Write to a different output path; default is in-place')
    args = ap.parse_args()
    original = args.source.read_text()
    updated = transform(original)  # All checks run BEFORE we write any file.
    dest = args.output or args.source
    if dest == args.source:
        backup = args.source.with_name(args.source.name + '.before_quantization')
        if backup.exists():
            raise RuntimeError(f'Backup {backup} already exists; refusing overwrite')
        backup.write_text(original)
    dest.write_text(updated)
    print(f'Updated {dest} with GPTQ/AWQ model construction + strict loading.')
    if dest == args.source:
        print(f'Original saved in {backup}.')


if __name__ == '__main__':
    main()
