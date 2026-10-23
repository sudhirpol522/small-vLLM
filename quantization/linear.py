"""Step 2: Give TinyServe one interface for dense and quantized linear layers.

FIRST PRINCIPLE
---------------
A transformer projection asks for a mathematical operation, not a weight format:

    y = linear(x)                    # e.g. [2, 4] -> [2, 3]

The operation is the same whether its weights came from a dense, GPTQ, or AWQ
checkpoint. What differs is (1) what tensors we allocate, and (2) how we do
the multiplication. Therefore the model owns a *Linear* object, and that object
delegates these differences to a *method*.

THIS FILE'S SCOPE
-----------------
- Dense weights work today (so we can test the abstraction independently).
- GPTQ/AWQ checkpoint detection uses QuantConfig from config.py.
- Quantized layers deliberately raise NotImplementedError until we implement
  GPTQMethod and AWQMethod. We must not silently turn a 4-bit checkpoint into
  a dense layer: its qweight/scales/qzeros would not load correctly.
- No CUDA kernels, vLLM imports, or checkpoint loading are implemented here.

TinyServe will eventually construct a projection like this:

    layer = make_linear(
        4096, 4096, bias=False,
        name="layers.0.self_attn.q_proj",
        quant_config=qcfg,
    )

The call site will still simply do ``q = layer(x)`` in Attention.forward().
"""

from __future__ import annotations  # Allows us to use Linear in method type hints.

from typing import Protocol  # An interface: any compatible method can implement it.

import torch  # Tensor objects and their dtypes (float16/bfloat16/int32).
from torch import nn  # nn.Module registers named parameters for TinyServe's loader.
from torch.nn import functional as F  # F.linear is the reference dense GEMM.

from .config import QuantConfig  # Step 1 already decided which layers are quantized.


class LinearMethod(Protocol):
    """Contract shared by dense, GPTQ, and AWQ implementations.

    Why three operations instead of one forward()?
      1. create_weights: choose checkpoint-shaped parameters BEFORE loading.
      2. process_weights_after_loading: transform loaded weights for a kernel.
      3. apply: multiply input activations by the prepared weights.

    Example: GPTQ will create qweight instead of weight, then repack it for
    Marlin, then use Marlin for apply(). Dense needs none of those conversions.

    Protocol avoids forcing the later GPTQ/AWQ method classes to inherit from
    a complicated base class; they only need these three methods.
    """

    def create_weights(
        self, layer: Linear, in_features: int, out_features: int,
        params_dtype: torch.dtype,
    ) -> None:
        ...  # Create checkpoint-compatible nn.Parameters on 'layer'.

    def process_weights_after_loading(self, layer: Linear) -> None:
        ...  # Optional post-load packing; performed once, not each forward.

    def apply(
        self, layer: Linear, x: torch.Tensor, bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ...  # Must return the same output shape as ordinary nn.Linear.


class DenseLinearMethod:
    """The simple reference method for a normal FP16/BF16 checkpoint."""

    def create_weights(
        self, layer: Linear, in_features: int, out_features: int,
        params_dtype: torch.dtype,
    ) -> None:
        # PyTorch's nn.Linear stores weight as [out_features, in_features].
        # Example: Linear(4, 3) -> weight shape [3, 4], NOT [4, 3].
        weight = torch.empty(
            (out_features, in_features), dtype=params_dtype,
        )  # Honors surrounding `with torch.device("cuda")` in serve.py.

        # TinyServe loads weights via dict(model.named_parameters()).
        # Registering `weight` means the key `...q_proj.weight` still matches.
        # requires_grad=False because TinyServe does inference, not training.
        layer.register_parameter(
            "weight", nn.Parameter(weight, requires_grad=False)
        )

    def process_weights_after_loading(self, layer: Linear) -> None:
        # Dense weights already have the exact layout expected by F.linear.
        # Example: a safetensors [3, 4] weight can be used without repacking.
        pass

    def apply(
        self, layer: Linear, x: torch.Tensor, bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # x [2, 4] @ weight.T [4, 3] -> result [2, 3]; bias is broadcast.
        # This is mathematically equivalent to torch.nn.Linear.forward().
        return F.linear(x, layer.weight, bias)


class Linear(nn.Module):
    """One layer exposed to Attention/MLP, independent of how W is stored.

    Example: q_proj = Linear(4, 3, bias=False, method=DenseLinearMethod())
             q_proj(torch.ones(2, 4)) produces shape [2, 3].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        method: LinearMethod,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()  # Required for assigning nn.Parameters to a module.

        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear dimensions must be positive")

        # Store the *logical* dimensions, even if the weights will be packed.
        # Example: GPTQ may have qweight [512, 4096], but K=N=4096 logically.
        self.in_features = in_features
        self.out_features = out_features
        self.method = method  # E.g. DenseLinearMethod today, GPTQMethod later.

        # `serve.py` currently sets the default dtype before Model(...) runs.
        # Example: with default_dtype(bfloat16), params_dtype is bfloat16.
        dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()

        # The method creates the correct *names* and *shapes* before load_model
        # builds its named_parameters() dictionary.
        # Dense: weight [N, K]. GPTQ: qweight [K/8, N], scales, qzeros.
        self.method.create_weights(self, in_features, out_features, dtype)

        # Bias is a normal (unquantized) vector for all methods.
        # Example: q_proj with bias=True needs bias [N], not packed INT4.
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype), requires_grad=False
            )
        else:
            # Register None so self.bias is always accessible without
            # including a nonexistent bias tensor in named_parameters().
            self.register_parameter("bias", None)

        # We intentionally do not initialize these tensors with random values.
        # TinyServe's weight loader must fill *every* parameter before inference.
        # A later step will make that loader strict; until then, DO NOT replace
        # serve.py's working dense layers with this class in production.

    def process_weights_after_loading(self) -> None:
        # Called ONCE after all safetensors files are loaded, before profiling.
        # GPTQ/AWQ will convert checkpoint packing to the kernel's layout here.
        self.method.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The model continues using self.q_proj(x), self.up_proj(x), etc.
        # Example: x [2, 4] -> y [2, 3], regardless of the weight format.
        return self.method.apply(self, x, self.bias)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    name: str,
    quant_config: QuantConfig | None,
    params_dtype: torch.dtype | None = None,
    quant_method: LinearMethod | None = None,
) -> Linear:
    """Choose a method for one named projection using Step 1's QuantConfig.

    DENSE EXAMPLE:
        make_linear(4, 3, bias=False,
                    name="layers.0.self_attn.q_proj", quant_config=None)
        -> Linear using DenseLinearMethod.

    GPTQ EXAMPLE (once gptq.py is implemented):
        make_linear(4096, 4096, bias=False,
                    name="layers.0.self_attn.q_proj", quant_config=qcfg,
                    quant_method=GPTQMethod(qcfg, backend))
        -> Linear using GPTQMethod.

    Why allow quant_method to be passed in? The next files will construct
    GPTQMethod/AWQMethod; this file should only know their shared interface.
    """
    # A model may be entirely dense, or a quantized checkpoint may exclude
    # one projection. In either case the layer needs an ordinary dense weight.
    needs_quant = (
        quant_config is not None and quant_config.should_quantize_layer(name)
    )

    if needs_quant:
        if quant_method is None:
            # We haven't implemented GPTQMethod or AWQMethod yet. Fail clearly
            # rather than silently allocating an incorrect dense weight.
            raise NotImplementedError(
                f"{name!r} needs {quant_config.method.upper()} weights. "
                "Implement its quantization method in the next step and "
                "pass it as quant_method."
            )
        chosen_method = quant_method
    else:
        if quant_method is not None:
            # A programmer error: accidentally passing GPTQMethod for lm_head
            # would make the model/checkpoint disagree. Catch it immediately.
            raise ValueError(f"{name!r} is dense, but a quant_method was supplied")
        chosen_method = DenseLinearMethod()

    return Linear(
        in_features,
        out_features,
        bias=bias,
        method=chosen_method,
        params_dtype=params_dtype,
    )
