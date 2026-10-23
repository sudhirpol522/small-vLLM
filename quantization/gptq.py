"""Step 3: Teach TinyServe what a *GPTQ checkpoint* stores.

FIRST PRINCIPLE
---------------
A standard nn.Linear(4096, 4096) stores one dense matrix called `weight`.
A 4-bit GPTQ checkpoint stores compressed integers plus scaling information:

    qweight  [512, 4096]  int32     # 4096 input weights / 8 values per int32
    qzeros   [ 32,  512]  int32     # 32 groups, 4096 outputs / 8 per int32
    scales   [ 32, 4096]  fp16     # one scale per group and output channel

Why 8? An int32 holds 32 bits, and 32 / 4 = 8 four-bit values.
Why 32 groups? Input dimension 4096 / group_size 128 = 32.

This file creates tensors matching that *on-disk* representation. That way
TinyServe's future strict safetensors loader can match keys like
"layers.0.self_attn.q_proj.qweight" in model.named_parameters().

It does NOT run Marlin, quantize floating-point weights, load safetensors, or
implement CUDA. The Marlin backend will be a separate file. Until a backend is
provided, running a GPTQ layer fails loudly instead of producing garbage.

HOW TO USE WITH THE PREVIOUS FILE
---------------------------------
    from quantization.config import QuantConfig
    from quantization.gptq import GPTQMethod
    from quantization.linear import make_linear

    cfg = QuantConfig(method="gptq", bits=4, group_size=128,
                      runtime_zero_points=False)
    layer = make_linear(4096, 4096, bias=False,
                        name="layers.0.self_attn.q_proj",
                        quant_config=cfg,
                        quant_method=GPTQMethod(cfg))
    print(layer.qweight.shape)  # torch.Size([512, 4096])

Weight allocation works now. Forward execution awaits the Marlin adapter.
"""

from __future__ import annotations  # Resolve type annotations when needed, not immediately.

from typing import TYPE_CHECKING, Protocol  # Describe the future backend's interface.

import torch  # Creates tensors and defines int32/float16/bfloat16 dtypes.
from torch import nn  # nn.Parameter is discoverable through named_parameters().

from .config import QuantConfig  # Reuse the validated checkpoint settings from Step 1.

if TYPE_CHECKING:
    # Type-only import avoids a runtime import cycle with quantization.linear.
    # The Linear class can be constructed without importing this file itself.
    from .linear import Linear


class GPTQBackend(Protocol):
    """The *interface* that our later Marlin backend will implement.

    Why separate this from GPTQMethod?
      - GPTQ answers: how was the checkpoint PACKED?
      - Marlin answers: how should those integers be PREPARED and MULTIPLIED?

    Example: a future MarlinBackend.prepare_gptq() will repack `qweight` and
    permute `scales` once; MarlinBackend.run() will call the compiled GEMM.
    Neither function is implemented here, intentionally.
    """

    def prepare_gptq(self, layer: Linear, config: QuantConfig) -> None:
        ...  # Convert checkpoint representation to a GPU-kernel representation.

    def run(
        self, layer: Linear, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        ...  # Compute e.g. [2, 4096] -> [2, 4096] using packed weights.


class GPTQMethod:
    """Create 4-bit symmetric GPTQ checkpoint weights for one Linear layer.

    This follows the three operations expected by quantization/linear.py:
      create_weights()                called while building the model
      process_weights_after_loading() called after strict checkpoint loading
      apply()                         called for every projection in inference

    We support *static-group* symmetric GPTQ only for v1. The corresponding
    QuantConfig is checked by config.py; layer-specific shapes are checked here.
    """

    def __init__(
        self, config: QuantConfig, backend: GPTQBackend | None = None
    ) -> None:
        # GPTQMethod must not accidentally be used with AWQ's different packing.
        # Example: QuantConfig(method="awq", ...) -> immediate ValueError.
        if config.method != "gptq":
            raise ValueError("GPTQMethod requires QuantConfig(method='gptq')")

        # We deliberately implement only the most common Marlin v1 format.
        # Example: 8-bit GPTQ needs a different pack factor and backend rules.
        if config.bits != 4 or config.runtime_zero_points:
            raise ValueError("v1 GPTQ requires 4 bits and symmetric quantization")

        self.config = config  # Holds bits=4 and group_size=128 (for example).
        self.backend = backend  # None until our later marlin.py is implemented.

    def create_weights(
        self,
        layer: Linear,
        in_features: int,
        out_features: int,
        params_dtype: torch.dtype,
    ) -> None:
        """Register packed *checkpoint-layout* tensors with the given layer.

        Example: K=4096, N=4096, group_size=128, params_dtype=bfloat16
            qweight: [512, 4096], int32
            qzeros:  [32,  512], int32
            scales:  [32, 4096], bfloat16

        IMPORTANT: These are uninitialized parameters. Do not run inference
        before the loader has filled them and the backend has prepared them.
        """
        # Names K and N match the GEMM view: x is [..., K], y is [..., N].
        K = in_features  # Example: hidden_size = 4096 input features.
        N = out_features  # Example: q_proj outputs 4096 features.
        pack = self.config.pack_factor  # 32 bits / 4 bits = 8 values per int32.
        group_size = self.config.group_size  # Example: 128 input values per group.

        # The weights are integers, but activation/scaling math is floating-point.
        # TinyServe typically runs BF16; FP16 is also supported by Marlin.
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"GPTQ requires float16/bfloat16 activations and scales, got {params_dtype}"
            )

        # GPTQ packs qweight along the INPUT axis (K), unlike AWQ GEMM.
        # Example: K=4096 -> 4096 / 8 = 512 rows in packed qweight.
        if K % pack != 0:
            raise ValueError(f"GPTQ in_features={K} must be divisible by {pack}")

        # qzeros packs 8 OUTPUT zero-point entries inside each int32.
        # Example: N=4096 -> 4096 / 8 = 512 packed zero-point columns.
        if N % pack != 0:
            raise ValueError(f"GPTQ out_features={N} must be divisible by {pack}")

        # A group is a consecutive set of K input weights sharing scales.
        # Example: group_size=128 and K=4096 -> 4096 / 128 = 32 groups.
        # group_size=-1 means one group spans all K inputs: groups=1.
        if group_size == -1:
            groups = 1
        else:
            if group_size <= 0 or K % group_size != 0:
                raise ValueError(
                    f"GPTQ in_features={K} must be divisible by group_size={group_size}"
                )
            groups = K // group_size

        # torch.empty does not initialize weights: the loader MUST fill them.
        # Because serve.py constructs Model inside `with torch.device('cuda')`,
        # these tensors will be allocated on the right device automatically.
        qweight = torch.empty((K // pack, N), dtype=torch.int32)
        qzeros = torch.empty((groups, N // pack), dtype=torch.int32)
        scales = torch.empty((groups, N), dtype=params_dtype)

        # nn.Parameter(requires_grad=False) makes checkpoint names discoverable
        # through model.named_parameters(); the current TinyServe loader uses it.
        # Example: "layers.0.self_attn.q_proj.qweight" becomes a valid key.
        layer.register_parameter(
            "qweight", nn.Parameter(qweight, requires_grad=False)
        )
        layer.register_parameter(
            "qzeros", nn.Parameter(qzeros, requires_grad=False)
        )
        layer.register_parameter(
            "scales", nn.Parameter(scales, requires_grad=False)
        )

        # Mark THIS layer unprepared. Stored on the layer (not on GPTQMethod),
        # so readiness is never accidentally shared across different layers.
        # Example: layer 0 being prepared must not make layer 1 runnable.
        layer._gptq_prepared = False

    def process_weights_after_loading(self, layer: Linear) -> None:
        """Repack loaded tensors ONCE, after the checkpoint loader validates them.

        The backend will eventually replace checkpoint-layout tensors with the
        layout Marlin needs. This cannot happen before checkpoint loading:
        `torch.empty` qweight still contains uninitialized values then.
        """
        # Marlin preparation changes the weight layout. Doing it twice would
        # treat an already-repacked tensor as if it were still checkpoint GPTQ.
        # Example: after repacking qweight [512, 4096], a second repack is invalid.
        if getattr(layer, "_gptq_prepared", False):
            raise RuntimeError("GPTQ layer has already been prepared")

        if self.backend is None:
            # Step 3 deliberately stops at checkpoint-compatible allocation.
            # Future Step: MarlinBackend.prepare_gptq(layer, config).
            raise NotImplementedError(
                "GPTQ weight allocation is ready, but the Marlin backend "
                "is not connected yet; do not run quantized inference."
            )

        # Let the backend handle Marlin padding, repacking and scale transforms.
        # Example: qweight [512, 4096] may change to a Marlin tile layout.
        self.backend.prepare_gptq(layer, self.config)

        # Only mark prepared AFTER the backend returns without error.
        layer._gptq_prepared = True

    def apply(
        self, layer: Linear, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Delegate the math to the existing GPU kernel rather than dequantizing.

        Example: input x [10, 4096] produces output [10, 4096].
        The actual computation will live in the later Marlin adapter.
        """
        if not getattr(layer, "_gptq_prepared", False):
            # Otherwise we'd interpret raw/uninitialized checkpoint weights
            # as Marlin-packed weights and obtain incorrect results.
            raise RuntimeError(
                "GPTQ layer is not prepared; load and validate its checkpoint "
                "weights, then call process_weights_after_loading()."
            )

        if self.backend is None:
            raise RuntimeError("GPTQ backend is missing")

        # The kernel expects K features, regardless of whether x is 2D or 3D.
        # Example: x [2, 5, 4096] -> output [2, 5, 4096].
        if x.shape[-1] != layer.in_features:
            raise ValueError(
                f"Expected last input dimension {layer.in_features}, got {x.shape[-1]}"
            )

        return self.backend.run(layer, x, bias)
