"""Step 5: Reuse vLLM's *compiled* Marlin INT4 GEMM from TinyServe.

FIRST PRINCIPLE: checkpoint layout != GPU kernel layout.
-------------------------------------------------------
Imagine a 4096-input, 4096-output linear projection (K=N=4096):

  GPTQ checkpoint: qweight [512, 4096]   (8 INT4 values per int32 along K)
  AWQ  checkpoint: qweight [4096, 512]   (8 INT4 values per int32 along N)

Neither tensor can be fed directly to Marlin's GEMM. Marlin expects a
rearrangement of packed weights into GPU-friendly tiles, plus separately
permuted scaling/zero-point data.

We do NOT write the GEMM or the Marlin packer: those come from an installed,
CUDA-enabled vLLM build. The tiny AWQ unpack/repack below is plain PyTorch,
following vLLM's own conversion; it is run ONCE during model loading.

LIFECYCLE (in the order we'll eventually call it from TinyServe):

    method = GPTQMethod(qcfg, backend=MarlinBackend())
    layer  = make_linear(..., quant_method=method, quant_config=qcfg)
    # The yet-to-be-modified serve.py loader fills qweight/scales/qzeros.
    layer.process_weights_after_loading()    # Converts layout ONCE.
    y = layer(x)                              # Compiled Marlin GEMM.

The module imports without vLLM installed. Preparing or executing on a GPU
REQUIRES a compatible, pinned vLLM release with compiled CUDA extensions.

Reference code inspected while designing this adapter:
  vllm/model_executor/layers/quantization/auto_awq.py
      _convert_awq_to_standard_format
  vllm/model_executor/kernels/linear/mixed_precision/marlin.py
      MarlinLinearKernel.process_weights_after_loading
  vllm/model_executor/layers/quantization/utils/marlin_utils.py
      apply_gptq_marlin_linear

SCOPE: single GPU; INT4 GPTQ(sym=True, desc_act=False) and GEMM AWQ with
zero points; FP16/BF16 activations; A100/H100. No custom CUDA, TP, or Machete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .config import QuantConfig

if TYPE_CHECKING:
    from .linear import Linear


# AWQ has an unusual ordering *within each packed int32*.
# Example: instead of values 0,1,2,3,4,5,6,7 in that order, AWQ packs
# positions in a way that is corrected by indexing [0,4,1,5,2,6,3,7].
_AWQ_UNPACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _awq_to_standard(
    qweight: torch.Tensor, qzeros: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert AWQ GEMM checkpoint packing to standard input-packed packing.

    This is a PyTorch *layout conversion*, NOT a CUDA inference kernel.
    It mirrors vLLM's _convert_awq_to_standard_format() for INT4.

    Example with K=N=4096 and G=32:
      INPUT  qweight: [4096, 512], qzeros: [32, 512]
      OUTPUT qweight: [512, 4096], qzeros: [512, 32]

    Why does qzeros become [512, 32]? Marlin's shared preprocessing expects
    zero-point values packed along OUTPUT channels, then transposed into its
    standard [N/8, groups] intermediate representation.
    """
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError("AWQ qweight and qzeros must be torch.int32")
    if qweight.ndim != 2 or qzeros.ndim != 2:
        raise ValueError("AWQ qweight and qzeros must be 2D tensors")

    K, N_packed = qweight.shape  # E.g. (4096, 512).
    G, zero_N_packed = qzeros.shape  # E.g. (32, 512).
    pack = 8  # Eight 4-bit values in each 32-bit integer.
    if K % pack != 0 or N_packed != zero_N_packed:
        raise ValueError("AWQ packed dimensions are inconsistent")
    N = N_packed * pack  # E.g. 512 * 8 = 4096 logical outputs.

    # Shifts: (0,4,8,12,16,20,24,28) extracts successive 4-bit nibbles.
    # We use an INT32 tensor on the source device; PyTorch handles signed
    # int32 shifts, and `& 0xF` discards the sign-extended upper bits.
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    order = torch.tensor(_AWQ_UNPACK_ORDER, device=qweight.device)

    # Expand [K, N/8] -> [K, N/8, 8] -> [K, N] *logical* INT4 values.
    # Example: qweight[0,0] contains 8 weights; unpacking reveals those 8.
    weights = ((qweight.unsqueeze(-1) >> shifts) & 0xF)
    weights = weights[:, :, order].reshape(K, N)

    # Now pack groups of eight INPUT rows, not output columns:
    # [K, N] -> [K/8, 8, N] -> [K/8, N] packed INT32.
    weights = weights.reshape(K // pack, pack, N)
    standard_qweight = (weights << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32
    ).contiguous()

    # Undo the same unusual AWQ nibble order for the zero points.
    # [G, N/8] -> [G, N] -> [N, G] -> [N/8, 8, G].
    zeros = ((qzeros.unsqueeze(-1) >> shifts) & 0xF)
    zeros = zeros[:, :, order].reshape(G, N).t().contiguous()
    zeros = zeros.reshape(N // pack, pack, G)
    standard_qzeros = (zeros << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32
    ).contiguous()

    return standard_qweight, standard_qzeros


def _vllm_dependencies():
    """Import vLLM lazily so CPU-only checkpoint-layout tests still work.

    vLLM's APIs here are *internal*. Pin and test an exact installed version;
    if these utilities move or signatures change, this adapter must change.
    """
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        from vllm.scalar_type import scalar_types
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "MarlinBackend needs an installed CUDA-enabled vLLM build "
            "compatible with this adapter. Pin the tested vLLM version."
        ) from exc
    return ops, mu, scalar_types


def _validate_loaded_layer(layer: Linear, config: QuantConfig) -> tuple[int, int, int]:
    """Fail before touching weights if checkpoint tensors don't fit this layer.

    E.g. if layer is [4096 -> 4096] with group_size=128, the `scales`
    tensor must have [32,4096] entries. Wrong shapes usually mean the
    checkpoint is incompatible, not that we should silently reshape it.
    """
    K, N = layer.in_features, layer.out_features
    pack = config.pack_factor
    if config.bits != 4 or pack != 8:
        raise ValueError("MarlinBackend v1 supports only INT4 (pack factor 8)")
    if config.group_size not in (-1, 32, 64, 128):
        raise ValueError("Unsupported Marlin group size")
    if K % pack or N % pack:
        raise ValueError("K and N must be divisible by the INT4 pack factor")
    if config.group_size != -1 and K % config.group_size:
        raise ValueError("K must be divisible by group_size")
    groups = 1 if config.group_size == -1 else K // config.group_size

    expected_q = (K // pack, N) if config.method == "gptq" else (K, N // pack)
    expected_z = (groups, N // pack)
    expected_s = (groups, N)
    for name, shape, dtype in (
        ("qweight", expected_q, torch.int32),
        ("qzeros", expected_z, torch.int32),
    ):
        p = getattr(layer, name, None)
        if p is None or tuple(p.shape) != shape or p.dtype != dtype:
            raise ValueError(f"{name} must be {shape} {dtype}; got {p!r}")

    scale = getattr(layer, "scales", None)
    if scale is None or tuple(scale.shape) != expected_s:
        raise ValueError(f"scales must have shape {expected_s}")
    if scale.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("scales must be FP16 or BF16")

    if layer.qweight.device.type != "cuda":
        raise RuntimeError("Marlin preparation requires an NVIDIA CUDA GPU")
    if not (layer.qweight.device == layer.qzeros.device == layer.scales.device):
        raise ValueError("All quantized parameters must be on the same GPU")
    major, minor = torch.cuda.get_device_capability(layer.qweight.device)
    if major * 10 + minor < 75:
        raise RuntimeError("Marlin requires SM75 or newer; use A100/SM80 or H100/SM90")

    return K, N, groups


class MarlinBackend:
    """Shared post-load and forward backend for GPTQMethod and AWQMethod.

    Important: This class does not load safetensors. An *upcoming* change to
    serve.py will first load all checkpoint parameters and only then invoke
    layer.process_weights_after_loading().

    One instance may be passed to each layer. Mutable per-layer state is
    stored on the `layer` itself, particularly `marlin_workspace`.
    """

    def prepare_gptq(self, layer: Linear, config: QuantConfig) -> None:
        """Turn an already-loaded symmetric GPTQ INT4 projection into Marlin.

        The checkpoint qzeros is validated but the symmetric Marlin kernel
        uses quant_type=uint4b8 (implicit bias) rather than runtime zeros.
        That distinction matches vLLM's AutoGPTQLinearMethod configuration.
        """
        if config.method != "gptq" or config.runtime_zero_points:
            raise ValueError("prepare_gptq requires symmetric GPTQ settings")
        self._prepare(layer, config, is_awq=False)

    def prepare_awq(self, layer: Linear, config: QuantConfig) -> None:
        """First normalize AWQ's unusual packing, then prepare for Marlin.

        Unlike GPTQ, AWQ needs runtime zero points (wtype uint4).
        The conversion is in plain PyTorch and follows vLLM's AWQ method.
        """
        if config.method != "awq" or not config.runtime_zero_points:
            raise ValueError("prepare_awq requires AWQ with runtime zero points")
        self._prepare(layer, config, is_awq=True)

    @torch.no_grad()  # Conversion never participates in autograd.
    def _prepare(self, layer: Linear, config: QuantConfig, *, is_awq: bool) -> None:
        if hasattr(layer, "marlin_workspace"):
            raise RuntimeError("This layer was already prepared for Marlin")
        K, N, groups = _validate_loaded_layer(layer, config)
        ops, mu, scalar_types = _vllm_dependencies()
        wtype = scalar_types.uint4 if is_awq else scalar_types.uint4b8
        # `has_zp` is TRUE for AWQ, FALSE for symmetric GPTQ.
        mu.verify_marlin_supported(wtype, config.group_size, has_zp=is_awq)

        # Marlin GEMM uses fixed thread tiles. A projection such as N=4104
        # may be padded to a tile-compatible N; output will be sliced back.
        padded_n, padded_k = mu.marlin_padded_nk(N, K, config.group_size)
        if is_awq:
            # AWQ [K,N/8] -> standardized [K/8,N].
            # qzeros [groups,N/8] -> standardized [N/8,groups].
            std_qweight, std_qzeros = _awq_to_standard(
                layer.qweight.detach(), layer.qzeros.detach()
            )
        else:
            std_qweight = layer.qweight.detach()  # GPTQ already packs along K.
            std_qzeros = None  # Symmetric GPTQ does not need runtime ZP.

        # Repack GPTQ-layout data into Marlin's GPU tile layout. This calls
        # vLLM's COMPILED op, not a custom CUDA kernel written in TinyServe.
        packed = ops.gptq_marlin_repack(
            mu.marlin_pad_qweight(
                std_qweight.contiguous(), N, K, padded_n, padded_k
            ),
            size_k=padded_k,
            size_n=padded_n,
            num_bits=4,
            is_a_8bit=False,
        )

        # Each scale originally corresponds to (group, output channel).
        # Marlin's threads read channels in a different order. If we forget
        # this permutation, qweight is correct but outputs are still wrong.
        reordered_scales = mu.marlin_permute_scales(
            mu.marlin_pad_scales(
                layer.scales.detach().contiguous(),
                N, K, padded_n, padded_k, config.group_size
            ),
            size_k=padded_k,
            size_n=padded_n,
            group_size=config.group_size,
            is_a_8bit=False,
        )

        if is_awq:
            # AWQ needs explicit runtime zero points. The standardized
            # packed zeros [N/8,G] must be transposed/unpacked to [G,N],
            # padded and then permuted/repacked into Marlin's zero format.
            padded_g = padded_k // config.group_size if config.group_size != -1 else 1
            zeros_unpacked = mu.unpack_cols(
                std_qzeros.t().contiguous(), 4, groups, N
            )
            padded_zeros = mu.marlin_pad_scales(
                zeros_unpacked, N, K, padded_n, padded_k, config.group_size
            )
            kernel_zeros = mu.marlin_zero_points(
                padded_zeros,
                size_k=padded_g,
                size_n=padded_n,
                num_bits=4,
                is_a_8bit=False,
            )
        else:
            # Empty int32 signals the GPTQ symmetric path to Marlin. We can
            # discard checkpoint qzeros after validating/loading them.
            kernel_zeros = mu.marlin_make_empty(layer.qweight.device)

        # The optimized kernel permutes bias for the same reason as scales.
        # Compute it now (not during each generated token's forward).
        new_bias = None
        if layer.bias is not None:
            new_bias = mu.marlin_permute_bias(
                mu.marlin_pad_dim(layer.bias.detach(), N, padded_n)
            )

        # Marlin uses an int32 locks/workspace tensor. This uses vLLM's
        # workspace sizing helper (SM-count * maximum blocks per SM).
        # Example: it is allocated ONCE before TinyServe's CUDA graph capture;
        # stable addresses are important when replaying a CUDA graph.
        workspace = mu.marlin_make_workspace_new(
            layer.qweight.device, mu.MARLIN_MAX_BLOCKS_PER_SM
        )

        # Replace original checkpoint-layout parameters. Assigning an
        # nn.Parameter to an existing registered name replaces it cleanly.
        # The now-obsolete original storage can be released by PyTorch once
        # nothing references it, rather than occupying GPU memory forever.
        layer.qweight = nn.Parameter(packed, requires_grad=False)
        layer.scales = nn.Parameter(reordered_scales, requires_grad=False)
        layer.qzeros = nn.Parameter(kernel_zeros, requires_grad=False)
        if new_bias is not None:
            layer.bias = nn.Parameter(new_bias, requires_grad=False)

        # persistent=False means this runtime scratch space is not a model
        # checkpoint weight. It is still a real CUDA tensor with a fixed address.
        layer.register_buffer("marlin_workspace", workspace, persistent=False)
        layer._marlin_wtype = wtype

    def run(
        self, layer: Linear, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Execute the existing Marlin GEMM with already-prepared weights.

        Example: [10,4096] @ a compressed [4096,4096] projection ->
                 [10,4096]; a 3D input [2,5,4096] also works.
        """
        if not hasattr(layer, "marlin_workspace"):
            raise RuntimeError("Marlin weights not prepared: load and prepare first")
        if x.device != layer.qweight.device:
            raise ValueError("Input and prepared Marlin weights must be on same GPU")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Marlin INT4 requires FP16/BF16 activations")
        if x.shape[-1] != layer.in_features:
            raise ValueError("Input's last dimension must match the logical K")

        _, mu, _ = _vllm_dependencies()
        # Reuse vLLM's wrapper: it handles matrix reshape, padding the input
        # to Marlin's K-tile, dispatch, and trimming padded output columns.
        # Critically: pass OUR preallocated workspace instead of None, so
        # TinyServe's CUDA graphs do not require a new workspace allocation.
        return mu.apply_gptq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.qzeros,
            workspace=layer.marlin_workspace,
            wtype=layer._marlin_wtype,
            output_size_per_partition=layer.out_features,
            input_size_per_partition=layer.in_features,
            bias=bias,
            input_dtype=x.dtype,
        )
