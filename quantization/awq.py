"""Step 4: Give TinyServe the AWQ *checkpoint* representation.

FIRST PRINCIPLE
---------------
A linear layer computes y = x @ W.T, regardless of whether W is dense,
GPTQ-packed, or AWQ-packed. Our existing quantization/linear.py provides that
common ``layer(x)`` interface. This file answers a different question:

    "What parameters must an AWQ layer allocate so its checkpoint can load?"

For example, a 4096-input, 4096-output AWQ GEMM 4-bit projection with groups
of 128 expects checkpoint tensors named:

    qweight: [4096,  512] int32    # Packed along OUTPUT axis (N / 8).
    qzeros:  [  32,  512] int32    # 4096 / 128 = 32 groups.
    scales:  [  32, 4096] fp16    # Or bf16 storage in TinyServe, if requested.

WHY NOT REUSE GPTQMethod?
-------------------------
The GPTQ *logical* matrix is also [4096, 4096], but its checkpoint qweight is
[512, 4096]. GPTQ packs along the INPUT axis; AWQ GEMM packs along the OUTPUT
axis and also uses a different order for the 4-bit values inside int32.

IMPORTANT: This file does NOT convert AWQ packing, load checkpoints, or run
Marlin. Those are later steps. A backend must be supplied before inference.

EXAMPLE (runs on CPU, without vLLM or CUDA)
-------------------------------------------
    from quantization.config import QuantConfig
    from quantization.linear import make_linear
    from quantization.awq import AWQMethod

    qcfg = QuantConfig(method="awq", bits=4, group_size=128,
                       runtime_zero_points=True)
    layer = make_linear(4096, 4096, bias=False,
                        name="layers.0.self_attn.q_proj",
                        quant_config=qcfg,
                        quant_method=AWQMethod(qcfg))
    print(layer.qweight.shape)  # torch.Size([4096, 512])

No forward pass is possible yet: the checkpoint loader and Marlin backend are
not wired up, and the torch.empty() parameters contain uninitialized data.
"""

from __future__ import annotations  # Allow the Linear type hint without importing it now.

from typing import TYPE_CHECKING, Protocol  # Describes a backend, not an implementation.

import torch  # Tensor allocation, float16/bfloat16/int32 and Tensor typing.
from torch import nn  # nn.Parameter makes weights visible to named_parameters().

from .config import QuantConfig  # Step 1 already interpreted AWQ's config.json.

if TYPE_CHECKING:
    # Avoid importing Linear at runtime: linear.py needs methods like this one.
    # In type hints, Linear means our generic quantized/dense projection class.
    from .linear import Linear


class AWQBackend(Protocol):
    """Interface expected from our future Marlin backend.

    Checkpoint layout and kernel layout are DIFFERENT concerns.

    Example: prepare_awq() will eventually convert [K, N/8] AWQ qweight,
    including AWQ's nibble order, into the packed format Marlin requires.
    run() will then call an existing vLLM GEMM instead of a custom CUDA kernel.
    """

    def prepare_awq(self, layer: Linear, config: QuantConfig) -> None:
        ...  # Prepare loaded AWQ weights ONCE, before GPU memory profiling.

    def run(
        self, layer: Linear, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        ...  # Compute e.g. x [10,4096] -> y [10,4096] with prepared weights.


class AWQMethod:
    """A linear method for standard GEMM-format, 4-bit AWQ checkpoints.

    Three lifecycle methods match quantization/linear.py's LinearMethod:

    1. create_weights(): called during model construction, BEFORE loading.
    2. process_weights_after_loading(): called once AFTER strict loading.
    3. apply(): called when Attention/MLP evaluates e.g. self.q_proj(x).

    This mirrors GPTQMethod's lifecycle while using AWQ's own storage layout.
    """

    def __init__(
        self, config: QuantConfig, backend: AWQBackend | None = None
    ) -> None:
        # A GPTQ checkpoint has a different qweight orientation and bit order.
        # Example: passing QuantConfig(method="gptq", ...) is a programming error.
        if config.method != "awq":
            raise ValueError("AWQMethod requires QuantConfig(method='awq')")

        # TinyServe v1 deliberately supports AWQ GEMM INT4 with zero points.
        # Zero-point-free AWQ requires different backend assumptions.
        if config.bits != 4 or not config.runtime_zero_points:
            raise ValueError("v1 AWQ requires 4 bits and runtime zero points")

        self.config = config  # For example, bits=4 and group_size=128.
        self.backend = backend  # None until we write quantization/marlin.py.

    def create_weights(
        self,
        layer: Linear,
        in_features: int,
        out_features: int,
        params_dtype: torch.dtype,
    ) -> None:
        """Create AWQ GEMM checkpoint-shaped tensors on this Linear layer.

        Example: K=4096, N=4096, group_size=128, dtype=torch.bfloat16:
            qweight = [4096, 512]   int32
            qzeros  = [  32, 512]   int32
            scales  = [  32, 4096]  bfloat16

        The input K and output N are *logical* dimensions, even though qweight
        is packed. For a forward x=[2,4096], the result will be [2,4096].
        """
        K = in_features  # Example: self.q_proj receives 4096 features.
        N = out_features  # Example: self.q_proj produces 4096 features.
        pack = self.config.pack_factor  # 32-bit integer / 4-bit value = 8.
        group_size = self.config.group_size  # Example: 128.

        # Activations/scales are floating point while qweight/qzeros are int32.
        # Marlin's W4A16 path supports both FP16 and BF16 activations.
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "AWQ requires float16/bfloat16 activations and scales, "
                f"got {params_dtype}"
            )

        # AWQ packs 8 output-channel 4-bit values into one int32 value.
        # Example: N=4096 -> qweight second dimension is 4096/8=512.
        if N % pack != 0:
            raise ValueError(f"AWQ out_features={N} must be divisible by {pack}")

        # Our initial Marlin conversion also expects an input dimension
        # divisible by the pack factor: K=4096 works; K=4097 does not.
        if K % pack != 0:
            raise ValueError(f"AWQ in_features={K} must be divisible by {pack}")

        # Each group covers a consecutive run of input features.
        # Example: K=4096 / group_size=128 -> 32 rows in scales and qzeros.
        # group_size=-1 means one group covers the entire input dimension.
        if group_size == -1:
            groups = 1
        else:
            if group_size <= 0 or K % group_size != 0:
                raise ValueError(
                    f"AWQ in_features={K} must be divisible by "
                    f"group_size={group_size}"
                )
            groups = K // group_size

        # We allocate on PyTorch's current default device. Inside TinyServe's
        # `with torch.device("cuda")`, these parameters will be GPU-resident.
        # torch.empty() is uninitialized; a strict loader must fill them first.
        qweight = torch.empty((K, N // pack), dtype=torch.int32)
        qzeros = torch.empty((groups, N // pack), dtype=torch.int32)
        scales = torch.empty((groups, N), dtype=params_dtype)

        # Checkpoint keys are e.g. "layers.0.self_attn.q_proj.qweight".
        # Registering as nn.Parameter lets TinyServe's existing
        # dict(model.named_parameters()) discover and load these keys.
        # requires_grad=False because serving does not train the weights.
        layer.register_parameter(
            "qweight", nn.Parameter(qweight, requires_grad=False)
        )
        layer.register_parameter(
            "qzeros", nn.Parameter(qzeros, requires_grad=False)
        )
        layer.register_parameter(
            "scales", nn.Parameter(scales, requires_grad=False)
        )

        # Track preparation per LAYER: preparing layer 0 must not mark layer 1
        # as ready. Starting as False also blocks accidental use of garbage.
        layer._awq_prepared = False

    def process_weights_after_loading(self, layer: Linear) -> None:
        """Tell the future backend to convert AWQ weights exactly once.

        This method is NOT the conversion itself. The backend will eventually:
            1. Fix AWQ's nibble order [0,4,1,5,2,6,3,7].
            2. Convert AWQ's output-packed qweight to input-packed layout.
            3. Repack weights/scales/zero points for the Marlin GEMM.

        These steps happen AFTER checkpoint loading but BEFORE inference.
        """
        if getattr(layer, "_awq_prepared", False):
            # Applying preparation twice would reinterpret Marlin-packed bytes
            # as AWQ checkpoint bytes and produce a corrupted weight layout.
            raise RuntimeError("AWQ layer has already been prepared")

        if self.backend is None:
            # It's fine to create AWQ weights before we implement Marlin,
            # but it is NOT fine to run inference with unprepared tensors.
            raise NotImplementedError(
                "AWQ weight allocation is ready, but the Marlin backend "
                "is not connected yet; do not run quantized inference."
            )

        # The backend will work on the already-loaded values, not torch.empty().
        self.backend.prepare_awq(layer, self.config)

        # Mark ready ONLY if backend preparation completed successfully.
        layer._awq_prepared = True

    def apply(
        self, layer: Linear, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Run a prepared AWQ layer through the backend's existing GPU kernel.

        Example: x [2, 4096] => output [2, 4096]. No dequantization to a
        permanent dense [4096,4096] weight is needed in the final backend.
        """
        if not getattr(layer, "_awq_prepared", False):
            # An unprepared layer contains either garbage or AWQ-checkpoint
            # layout that Marlin cannot consume directly.
            raise RuntimeError(
                "AWQ layer is not prepared; load and validate its checkpoint "
                "weights, then call process_weights_after_loading()."
            )

        if self.backend is None:
            raise RuntimeError("AWQ backend is missing")

        # Input dimension remains logical K, even though the weight is packed.
        # Example: x=[3,2048] to a K=4096 layer cannot be multiplied.
        if x.shape[-1] != layer.in_features:
            raise ValueError(
                f"Expected last input dimension {layer.in_features}, got {x.shape[-1]}"
            )

        return self.backend.run(layer, x, bias)  # Existing vLLM kernel, later.
