# Read the engine from request to response

[Repository overview](../README.md) · [Architecture](architecture.md) · [Source index](api-index.md)

The guide starts with the behavior a reader can observe, then follows the code that implements it. You do not need to read `serve.py` from top to bottom. The foundations chapter assumes familiarity with continuous batching and introduces prefix caching and chunked prefill.

## Reading order

| Part | What it explains | Current status |
| --- | --- | --- |
| 1. Architecture | Entry points, component responsibilities, and one request's lifecycle. | [Overview available](architecture.md) |
| 2. Foundations | KV storage, prefix identity, and chunked prefill. | [Draft available](foundations.md) |
| 3. Request and configuration | `Config`, `SamplingParams`, `Request`, `ScheduledRequest`, and `Output`. | Planned |
| 4. KV cache manager | Hash lookup, capacity checks, cached-block claiming, allocation, registration, and release. | [Draft walkthrough available](kv-cache-manager.md) |
| 5. Scheduler | Waiting/running queues, token budgets, admission, preemption, and completion. | Planned |
| 6. Attention and model | `AttnMetadata`, attention backends, normalization, rotary embeddings, projections, and decoder layers. | Planned |
| 7. Model runner | Packed batches, slot mappings, GPU KV allocation, CUDA graphs, and sampling. | Planned |
| 8. Engine and text output | Model-step orchestration, token acceptance, stopping, and detokenization. | Planned |
| 9. HTTP serving | `AsyncEngine`, request parsing, streaming, cancellation, and the example client. | Planned |
| 10. Quantization | Configuration, the linear interface, checkpoint layouts, and Marlin integration. | Planned |
| 11. Demos, benchmarking, and CLI | Demo helpers, benchmark inputs, generation paths, and command dispatch. | Planned |
| 12. Integration utility and legacy snapshot | Source transformation and the earlier implementation. | Planned; indexed separately |

Only available chapters are linked. Planned chapters will become links as their explanations are written and checked against the implementation.

## Two ways to use the documentation

For a first read, start with [architecture](architecture.md), then [foundations](foundations.md), then the [KV cache manager](kv-cache-manager.md). Continue with the request and scheduler chapters as they are added.

For a specific symbol, use the [source index](api-index.md). Every explicitly defined class, function, method, property, and nested helper has a source link and a coverage status. Being listed means the definition was inventoried; it does not mean its walkthrough is complete.

## How chapters will be written

Each class starts with its responsibility, the state it owns, and the components it calls. Each method then follows the same pattern: purpose, relevant code, a worked example, and state before and after the call. Small methods stay short; allocation and scheduling receive more detail.

Diagrams stay next to the behavior they explain. The README contains the component map. Concept diagrams explain reuse and batching. Method diagrams show concrete block IDs, queues, and reference counts.

See the [authoring guide](authoring.md) for the chapter template and the [review notes](review-notes.md) for corrections needed before publication.
