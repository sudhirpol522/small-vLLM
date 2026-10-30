# small-vLLM

A compact, vLLM-style LLM inference engine in Python, built to make the serving pipeline easier to understand through readable code.

## Features

- Paged KV cache with prefix reuse and reference-counted blocks.
- Continuous batching, chunked prefill, and request preemption.
- FlashAttention and a PyTorch reference attention backend.
- CUDA graphs for decode execution.
- OpenAI-compatible HTTP endpoints with streaming.
- GPTQ and AWQ weight-quantization integrations.

## Code

| File | Purpose |
| --- | --- |
| [`serve.py`](serve.py) | Requests, KV cache management, scheduling, model execution, and serving. |
| [`quantization/`](quantization/) | Quantization configuration, linear layers, and backend integration. |
| [`inference.py`](inference.py) | Example streaming client. |

## Blog

I’m documenting the engine class by class and function by function on [my technical blog](https://sudhirpol522.github.io/writing/).

The first article, [Understanding the KV Cache Manager: Prefix Caching and Chunked Prefill](https://sudhirpol522.github.io/blog/small-vllm-kv-cache-manager/), explains the manager with code examples and Excalidraw diagrams.
