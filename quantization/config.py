"""Step 1: Understand *which* weights TinyServe is supposed to load.

This module intentionally does NOT allocate weights, read safetensors, or run CUDA.
It only interprets Hugging Face checkpoint metadata.

Try the examples at the bottom of this docstring with ``parse_quant_config``:

    from types import SimpleNamespace
    hf_cfg = SimpleNamespace(quantization_config={
        "quant_method": "gptq", "bits": 4, "group_size": 128,
        "sym": True, "desc_act": False,
    })
    qcfg = parse_quant_config(hf_cfg)
    assert qcfg.method == "gptq"
    assert qcfg.should_quantize_layer("layers.0.self_attn.q_proj")
    assert not qcfg.should_quantize_layer("lm_head")

Why does this exist? TinyServe currently creates every projection using nn.Linear.
We must inspect the checkpoint FIRST, because a 4-bit GPTQ layer needs
qweight/scales/qzeros parameters, not nn.Linear's dense .weight parameter.
"""

from __future__ import annotations  # Allows modern type annotations to be read safely.

from collections.abc import Mapping  # E.g. a dict from HF config.json is a Mapping.
from dataclasses import dataclass  # Generates the small configuration object's __init__.
from typing import Literal  # Restricts our method field to the two v1 formats.


# TinyServe's initial model has exactly these seven transformer projections.
# Example: "layers.0.self_attn.q_proj" ends in "q_proj"; it can be quantized.
# An embedding or RMSNorm is NOT a linear projection and should stay untouched.
_PROJECTIONS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
})

# Start only with group sizes supported by vLLM's Marlin INT4 path.
# Example: group_size=128 means 128 consecutive input weights share a scale.
# -1 means one group spans the whole input dimension for each output channel.
_SUPPORTED_GROUP_SIZES = frozenset({-1, 32, 64, 128})


@dataclass(frozen=True)
class QuantConfig:
    """The few quantization facts the rest of TinyServe will need.

    Example: QuantConfig(method="awq", bits=4, group_size=128,
                         runtime_zero_points=True) describes an AWQ layer.

    ``runtime_zero_points`` is about the Marlin KERNEL, not checkpoint keys:
    a symmetric GPTQ checkpoint can contain ``qzeros`` even though its Marlin
    kernel represents the offset through an implicit quantization bias.
    """

    method: Literal["gptq", "awq"]  # Which checkpoint packing convention?
    bits: int  # Example: 4 means eight 4-bit weights fit in one int32.
    group_size: int  # Example: 128 input weights per quantization group.
    runtime_zero_points: bool  # True for standard AWQ, False for symmetric GPTQ.
    included_layers: tuple[str, ...] = ()  # GPTQ whitelist; empty = all projections.
    excluded_layers: tuple[str, ...] = ()  # AWQ exclusions, e.g. ("o_proj",).

    @property
    def pack_factor(self) -> int:
        # A checkpoint packs values into 32-bit integers: 32 / 4 = 8.
        return 32 // self.bits

    def should_quantize_layer(self, name: str) -> bool:
        """Decide whether one TinyServe layer needs packed weights.

        Example (a normal GPTQ checkpoint):
            layers.0.self_attn.q_proj -> True
            layers.0.input_layernorm -> False
            lm_head                  -> False

        We intentionally leave the LM head dense: TinyServe's Model.logits()
        reads ``lm_head.weight`` directly instead of calling its forward().
        """
        # The checkpoint calls it "model.layers.0..." while TinyServe registers
        # "layers.0..."; stripping an optional prefix gives a consistent name.
        name = name.removeprefix("model.")

        # Only the seven projections listed above are in our initial scope.
        # Example: "embed_tokens" is not in _PROJECTIONS, so return False.
        if name.rsplit(".", 1)[-1] not in _PROJECTIONS:
            return False

        # Some GPTQ checkpoints quantize only selected projections.
        # Example: included_layers=("self_attn.q_proj",) skips the MLP.
        if self.included_layers and not any(
            _matches_layer_rule(name, rule) for rule in self.included_layers
        ):
            return False

        # Some AWQ checkpoints explicitly keep certain projections dense.
        # Example: excluded_layers=("self_attn.o_proj",) skips every o_proj.
        if any(_matches_layer_rule(name, rule) for rule in self.excluded_layers):
            return False

        return True  # A supported projection without an exclusion is quantized.


def _matches_layer_rule(name: str, rule: str) -> bool:
    """Match complete dotted path components, not arbitrary substrings.

    Example: rule="q_proj" matches "layers.0.self_attn.q_proj".
    Example: rule="self_attn" matches all projections inside self_attn.
    Non-example: rule="q_proj" does NOT match "my_q_projection".
    """
    # Dots at both ends ensure a rule matches complete path components.
    return f".{rule}." in f".{name}."


def _layer_rules(raw_value: object, field: str) -> tuple[str, ...]:
    """Convert optional list-of-names (or GPTQ list-of-lists) to one tuple.

    Example: [["q_proj", "k_proj"], ["v_proj"]] ->
             ("q_proj", "k_proj", "v_proj").
    """
    if raw_value is None:
        return ()  # Missing setting means no special inclusion/exclusion rule.

    if not isinstance(raw_value, (list, tuple)):
        raise ValueError(f"{field} must be a list of layer names")

    names: list[str] = []
    for item in raw_value:
        # GPTQ metadata may group names in sublists. Flatten those groups.
        group = item if isinstance(item, (list, tuple)) else [item]
        for name in group:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{field} contains an invalid layer name: {name!r}")
            # E.g. "model.layers.0.mlp.up_proj" -> "layers.0.mlp.up_proj".
            names.append(name.strip().removeprefix("model."))

    # dict.fromkeys removes duplicates but preserves checkpoint order.
    return tuple(dict.fromkeys(names))


def _required_alias(raw: Mapping, first: str, second: str | None = None):
    """Read equivalent keys from different checkpoint producers.

    Example: AWQ may say ``w_bit=4`` or ``bits=4``; both mean the same thing.
    If both appear and disagree, fail instead of silently choosing a value.
    """
    keys = (first,) if second is None else (first, second)
    present = [key for key in keys if key in raw]
    if not present:
        raise ValueError(f"Quantization config must specify {first!r}")
    if len(present) == 2 and raw[present[0]] != raw[present[1]]:
        raise ValueError(f"Conflicting values for {first!r} and {second!r}")
    return raw[present[0]]


def _as_mapping(value: object) -> Mapping:
    """HF usually provides a dict; some configuration objects have to_dict()."""
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    raise ValueError("quantization_config must be a mapping or support to_dict()")


def parse_quant_config(hf_config: object) -> QuantConfig | None:
    """Read and validate the quantization_config of an HF model configuration.

    Returns None for a dense model; it does NOT mean unsupported quantization.
    Unsupported quantized formats raise ValueError so TinyServe cannot silently
    load a packed model as if it contained dense nn.Linear weights.

    Example: SimpleNamespace(quantization_config=None) -> None.
    """
    # The existing serve.py already calls AutoConfig.from_pretrained(path).
    # That object normally has a .quantization_config property for GPTQ/AWQ.
    raw_value = (
        hf_config.get("quantization_config")
        if isinstance(hf_config, Mapping)
        else getattr(hf_config, "quantization_config", None)
    )

    if raw_value is None:
        return None  # No quantization metadata: keep today's dense nn.Linear.

    raw = _as_mapping(raw_value)
    method = raw.get("quant_method")  # Example: "gptq" or "awq".
    if method not in ("gptq", "awq"):
        raise ValueError(f"Unsupported quantization method for v1: {method!r}")

    # Refuse a quantized LM head because current TinyServe Model.logits()
    # accesses lm_head.weight directly. We can support this in a later step.
    if raw.get("lm_head", False) is not False:
        raise ValueError("v1 requires lm_head to remain unquantized")

    if method == "gptq":
        bits = _required_alias(raw, "bits")  # E.g. 4 bits per weight.
        group_size = _required_alias(raw, "group_size")  # E.g. 128.

        # Marlin's first GPTQ version will support symmetric INT4 only.
        if raw.get("sym") is not True:
            raise ValueError("v1 supports GPTQ only with sym=True")
        # Activation ordering changes how input columns map to scale groups.
        # E.g. desc_act=True is NOT supported by our simple loader yet.
        if raw.get("desc_act", False) is not False:
            raise ValueError("v1 requires GPTQ desc_act=False")
        # Per-layer regex overrides can change bits/group_size between layers.
        # Example: dynamic={".*q_proj": {"bits": 8}} is out of v1 scope.
        if raw.get("dynamic"):
            raise ValueError("v1 does not support GPTQ dynamic layer overrides")

        included_layers = _layer_rules(
            raw.get("modules_in_block_to_quantize"),
            "modules_in_block_to_quantize",
        )
        excluded_layers = ()
        runtime_zero_points = False  # GPTQ-Marlin uses an implicit symmetric bias.

    else:  # method == "awq"
        # AWQ producers may use w_bit/q_group_size instead of bits/group_size.
        bits = _required_alias(raw, "w_bit", "bits")
        group_size = _required_alias(raw, "q_group_size", "group_size")

        # Some AWQ producers label different weight layouts with "version".
        # Example: version="gemv" is NOT the common GEMM-format checkpoint.
        # Never silently load an unfamiliar layout using GEMM packing rules.
        awq_version = raw.get("version", "gemm")
        if not isinstance(awq_version, str) or awq_version.lower() != "gemm":
            raise ValueError(f"v1 supports AWQ version='gemm', got {awq_version!r}")

        # Standard AWQ-Marlin uses explicit runtime zero points.
        if raw.get("zero_point") is not True:
            raise ValueError("v1 supports AWQ only with zero_point=True")

        excluded_layers = _layer_rules(
            raw.get("modules_to_not_convert"), "modules_to_not_convert"
        )
        included_layers = ()
        runtime_zero_points = True

    # Validate shared INT4 Marlin requirements once for both methods.
    if type(bits) is not int or bits != 4:
        raise ValueError(f"v1 supports only 4-bit quantization, got {bits!r}")
    if type(group_size) is not int or group_size not in _SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"Unsupported group_size={group_size!r}; expected -1, 32, 64, or 128"
        )

    # Construct an immutable decision object for later model construction.
    # Example: qcfg.pack_factor == 8 for all supported INT4 checkpoints.
    return QuantConfig(
        method=method,
        bits=bits,
        group_size=group_size,
        runtime_zero_points=runtime_zero_points,
        included_layers=included_layers,
        excluded_layers=excluded_layers,
    )
