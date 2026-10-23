"""TinyServe weight-only INT4 quantization integrations.

`config.py` chooses checkpoint format; `linear.py` exposes one projection
interface; `gptq.py` and `awq.py` create checkpoint-compatible weights;
`marlin.py` reuses vLLM's compiled GEMM without a custom CUDA kernel.
"""
