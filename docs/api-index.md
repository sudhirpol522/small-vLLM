# Source and documentation index

[Guide index](README.md) · [Authoring guide](authoring.md)

Inventory of source revision [`62bfced`](https://github.com/sudhirpol522/small-vLLM/commit/62bfcedf01abcaa3df3d1905fd6028558e35dead). The active files contain **26 classes** and **115 named functions or methods**, including properties and nested helpers. The historical snapshot contains another 17 classes and 74 functions or methods, listed separately.

**Draft walkthrough** means an explanation exists but needs review. **Planned** means the source definition has been indexed and its walkthrough is still to be written. Only the KV cache manager currently has method-level walkthroughs. General architecture mentions do not count as completed coverage.

The index includes explicitly declared definitions, including private helpers and nested HTTP handlers. Generated dataclass methods and lambdas are not separate entries. `inference.py` and `quantization/__init__.py` contain no explicit class or function definitions; their file-level behavior still needs documentation.

## File coverage

| Source file | Classes | Functions and methods | Walkthrough status |
| --- | --- | --- | --- |
| [`inference.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/inference.py) | 0 | 0 | Planned |
| [`integrate_serve_quantization.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/integrate_serve_quantization.py) | 0 | 3 | Planned |
| [`quantization/__init__.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/__init__.py) | 0 | 0 | Planned |
| [`quantization/awq.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py) | 2 | 6 | Planned |
| [`quantization/config.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py) | 1 | 7 | Planned |
| [`quantization/gptq.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py) | 2 | 6 | Planned |
| [`quantization/linear.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py) | 3 | 10 | Planned |
| [`quantization/marlin.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py) | 1 | 7 | Planned |
| [`serve.py`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py) | 17 | 76 | KV manager draft; other components planned |
| [`serve.py.before_quantization`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization) | 17 | 74 | Legacy coverage planned |

## Definitions

Expand a file to find source locations and the intended chapter. The legacy snapshot is kept outside the active reading path.

<details>
<summary>inference.py</summary>

No explicitly declared classes or functions. File-level explanation planned.

</details>

<details>
<summary>integrate_serve_quantization.py</summary>

| Definition | Kind | Chapter | Status |
| --- | --- | --- | --- |
| [`replace_once`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/integrate_serve_quantization.py#L21) | Function | Integration utility | Planned |
| [`transform`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/integrate_serve_quantization.py#L32) | Function | Integration utility | Planned |
| [`main`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/integrate_serve_quantization.py#L389) | Function | Integration utility | Planned |

</details>

<details>
<summary>quantization/__init__.py</summary>

No explicitly declared classes or functions. File-level explanation planned.

</details>

<details>
<summary>quantization/awq.py</summary>

| Definition | Kind | Chapter | Status |
| --- | --- | --- | --- |
| [`AWQBackend`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py#L60) | Class | Quantization | Planned |
| [`AWQBackend.prepare_awq`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py#L70) | Method | Quantization | Planned |
| [`AWQBackend.run`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py#L73) | Method | Quantization | Planned |
| [`AWQMethod`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py#L79) | Class | Quantization | Planned |
| [`AWQMethod.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py#L91) | Method | Quantization | Planned |
| [`AWQMethod.create_weights`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py#L107) | Method | Quantization | Planned |
| [`AWQMethod.process_weights_after_loading`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py#L185) | Method | Quantization | Planned |
| [`AWQMethod.apply`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/awq.py#L214) | Method | Quantization | Planned |

</details>

<details>
<summary>quantization/config.py</summary>

| Definition | Kind | Chapter | Status |
| --- | --- | --- | --- |
| [`QuantConfig`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py#L45) | Class | Quantization | Planned |
| [`QuantConfig.pack_factor`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py#L64) | Property | Quantization | Planned |
| [`QuantConfig.should_quantize_layer`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py#L68) | Method | Quantization | Planned |
| [`_matches_layer_rule`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py#L103) | Function | Quantization | Planned |
| [`_layer_rules`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py#L114) | Function | Quantization | Planned |
| [`_required_alias`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py#L140) | Function | Quantization | Planned |
| [`_as_mapping`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py#L155) | Function | Quantization | Planned |
| [`parse_quant_config`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/config.py#L167) | Function | Quantization | Planned |

</details>

<details>
<summary>quantization/gptq.py</summary>

| Definition | Kind | Chapter | Status |
| --- | --- | --- | --- |
| [`GPTQBackend`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py#L55) | Class | Quantization | Planned |
| [`GPTQBackend.prepare_gptq`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py#L67) | Method | Quantization | Planned |
| [`GPTQBackend.run`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py#L70) | Method | Quantization | Planned |
| [`GPTQMethod`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py#L76) | Class | Quantization | Planned |
| [`GPTQMethod.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py#L88) | Method | Quantization | Planned |
| [`GPTQMethod.create_weights`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py#L104) | Method | Quantization | Planned |
| [`GPTQMethod.process_weights_after_loading`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py#L181) | Method | Quantization | Planned |
| [`GPTQMethod.apply`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/gptq.py#L209) | Method | Quantization | Planned |

</details>

<details>
<summary>quantization/linear.py</summary>

| Definition | Kind | Chapter | Status |
| --- | --- | --- | --- |
| [`LinearMethod`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L45) | Class | Quantization | Planned |
| [`LinearMethod.create_weights`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L60) | Method | Quantization | Planned |
| [`LinearMethod.process_weights_after_loading`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L66) | Method | Quantization | Planned |
| [`LinearMethod.apply`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L69) | Method | Quantization | Planned |
| [`DenseLinearMethod`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L75) | Class | Quantization | Planned |
| [`DenseLinearMethod.create_weights`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L78) | Method | Quantization | Planned |
| [`DenseLinearMethod.process_weights_after_loading`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L95) | Method | Quantization | Planned |
| [`DenseLinearMethod.apply`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L100) | Method | Quantization | Planned |
| [`Linear`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L108) | Class | Quantization | Planned |
| [`Linear.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L115) | Method | Quantization | Planned |
| [`Linear.process_weights_after_loading`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L160) | Method | Quantization | Planned |
| [`Linear.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L165) | Method | Quantization | Planned |
| [`make_linear`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/linear.py#L171) | Function | Quantization | Planned |

</details>

<details>
<summary>quantization/marlin.py</summary>

| Definition | Kind | Chapter | Status |
| --- | --- | --- | --- |
| [`_awq_to_standard`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py#L60) | Function | Quantization | Planned |
| [`_vllm_dependencies`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py#L118) | Function | Quantization | Planned |
| [`_validate_loaded_layer`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py#L136) | Function | Quantization | Planned |
| [`MarlinBackend`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py#L183) | Class | Quantization | Planned |
| [`MarlinBackend.prepare_gptq`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py#L194) | Method | Quantization | Planned |
| [`MarlinBackend.prepare_awq`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py#L205) | Method | Quantization | Planned |
| [`MarlinBackend._prepare`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py#L216) | Method | Quantization | Planned |
| [`MarlinBackend.run`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/quantization/marlin.py#L318) | Method | Quantization | Planned |

</details>

<details>
<summary>serve.py</summary>

| Definition | Kind | Chapter | Status |
| --- | --- | --- | --- |
| [`Config`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L58) | Class | Request and configuration | Planned |
| [`Config.__post_init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L80) | Method | Request and configuration | Planned |
| [`SamplingParams`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L95) | Class | Request and configuration | Planned |
| [`Request`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L106) | Class | Request and configuration | Planned |
| [`Request.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L119) | Method | Request and configuration | Planned |
| [`Request.num_tokens`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L147) | Property | Request and configuration | Planned |
| [`Request.output_ids`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L151) | Property | Request and configuration | Planned |
| [`KVCacheManager`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L155) | Class | KV cache manager | [Draft walkthrough](kv-cache-manager.md) |
| [`KVCacheManager.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L230) | Method | KV cache manager | [Draft walkthrough](kv-cache-manager.md#initialization) |
| [`KVCacheManager.num_free`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L256) | Property | KV cache manager | [Draft walkthrough](kv-cache-manager.md#counting-free-blocks) |
| [`KVCacheManager._update_hashes`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L259) | Method | KV cache manager | [Draft walkthrough](kv-cache-manager.md#updating-block-hashes) |
| [`KVCacheManager.find_cached_prefix`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L309) | Method | KV cache manager | [Draft walkthrough](kv-cache-manager.md#finding-a-cached-prefix) |
| [`KVCacheManager.allocate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L366) | Method | KV cache manager | [Draft walkthrough](kv-cache-manager.md#allocating-blocks) |
| [`KVCacheManager._register_full_blocks`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L456) | Method | KV cache manager | [Draft walkthrough](kv-cache-manager.md#registering-full-blocks) |
| [`KVCacheManager.free_request`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L482) | Method | KV cache manager | [Draft walkthrough](kv-cache-manager.md#releasing-a-request) |
| [`ScheduledRequest`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L493) | Class | Request and configuration | Planned |
| [`ScheduledRequest.completes_prompt`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L499) | Property | Request and configuration | Planned |
| [`Scheduler`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L514) | Class | Scheduler | Planned |
| [`Scheduler.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L525) | Method | Scheduler | Planned |
| [`Scheduler.add`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L531) | Method | Scheduler | Planned |
| [`Scheduler.has_work`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L534) | Method | Scheduler | Planned |
| [`Scheduler.schedule`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L537) | Method | Scheduler | Planned |
| [`Scheduler.finish`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L680) | Method | Scheduler | Planned |
| [`AttnMetadata`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L689) | Class | Attention and model | Planned |
| [`torch_ref_attention`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L753) | Function | Attention and model | Planned |
| [`flash_attention`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L829) | Function | Attention and model | Planned |
| [`RMSNorm`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L846) | Class | Attention and model | Planned |
| [`RMSNorm.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L847) | Method | Attention and model | Planned |
| [`RMSNorm.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L852) | Method | Attention and model | Planned |
| [`rotary`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L857) | Function | Attention and model | Planned |
| [`build_projection`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L862) | Function | Attention and model | Planned |
| [`Attention`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L902) | Class | Attention and model | Planned |
| [`Attention.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L903) | Method | Attention and model | Planned |
| [`Attention.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L940) | Method | Attention and model | Planned |
| [`MLP`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L976) | Class | Attention and model | Planned |
| [`MLP.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L977) | Method | Attention and model | Planned |
| [`MLP.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L995) | Method | Attention and model | Planned |
| [`DecoderLayer`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L998) | Class | Attention and model | Planned |
| [`DecoderLayer.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L999) | Method | Attention and model | Planned |
| [`DecoderLayer.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1008) | Method | Attention and model | Planned |
| [`Model`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1013) | Class | Attention and model | Planned |
| [`Model.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1014) | Method | Attention and model | Planned |
| [`Model.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1040) | Method | Attention and model | Planned |
| [`Model.logits`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1047) | Method | Attention and model | Planned |
| [`_validate_gptq_g_idx`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1051) | Function | Model loading | Planned |
| [`load_model`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1080) | Function | Model loading | Planned |
| [`sample`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1187) | Function | Model runner and sampling | Planned |
| [`ModelRunner`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1248) | Class | Model runner and sampling | Planned |
| [`ModelRunner.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1272) | Method | Model runner and sampling | Planned |
| [`ModelRunner.allocate_kv_cache`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1312) | Method | Model runner and sampling | Planned |
| [`ModelRunner._prepare`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1375) | Method | Model runner and sampling | Planned |
| [`ModelRunner.graph_sizes`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1477) | Method | Model runner and sampling | Planned |
| [`ModelRunner.capture_cuda_graphs`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1486) | Method | Model runner and sampling | Planned |
| [`ModelRunner.run`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1533) | Method | Model runner and sampling | Planned |
| [`Output`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1564) | Class | Request and configuration | Planned |
| [`Detokenizer`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1575) | Class | Engine and text output | Planned |
| [`Detokenizer.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1587) | Method | Engine and text output | Planned |
| [`Detokenizer.update`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1600) | Method | Engine and text output | Planned |
| [`Engine`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1658) | Class | Engine and text output | Planned |
| [`Engine.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1662) | Method | Engine and text output | Planned |
| [`Engine.add_request`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1693) | Method | Engine and text output | Planned |
| [`Engine.abort`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1718) | Method | Engine and text output | Planned |
| [`Engine.has_work`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1725) | Method | Engine and text output | Planned |
| [`Engine.step`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1728) | Method | Engine and text output | Planned |
| [`Engine.generate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1786) | Method | Engine and text output | Planned |
| [`AsyncEngine`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1798) | Class | HTTP serving | Planned |
| [`AsyncEngine.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1804) | Method | HTTP serving | Planned |
| [`AsyncEngine._loop`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1821) | Method | HTTP serving | Planned |
| [`AsyncEngine.generate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1876) | Async method | HTTP serving | Planned |
| [`build_app`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1909) | Function | HTTP serving | Planned |
| [`build_app.get`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1924) | Nested function | HTTP serving | Planned |
| [`build_app.params_from`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1928) | Nested function | HTTP serving | Planned |
| [`build_app.error`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1943) | Nested function | HTTP serving | Planned |
| [`build_app.respond`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1947) | Async nested function | HTTP serving | Planned |
| [`build_app.respond.choice`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1968) | Nested function | HTTP serving | Planned |
| [`build_app.respond.usage`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1977) | Nested function | HTTP serving | Planned |
| [`build_app.respond.events`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L1991) | Async nested function | HTTP serving | Planned |
| [`build_app.health`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2020) | Async nested function | HTTP serving | Planned |
| [`build_app.models`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2025) | Async nested function | HTTP serving | Planned |
| [`build_app.metrics`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2032) | Async nested function | HTTP serving | Planned |
| [`build_app.completions`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2042) | Async nested function | HTTP serving | Planned |
| [`build_app.chat`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2048) | Async nested function | HTTP serving | Planned |
| [`kv_cache_demo`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2072) | Function | Demos, benchmarks, and CLI | Planned |
| [`kv_cache_demo.cached_pages`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2076) | Nested function | Demos, benchmarks, and CLI | Planned |
| [`kv_cache_demo.dump`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2079) | Nested function | Demos, benchmarks, and CLI | Planned |
| [`attention_demo`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2146) | Function | Demos, benchmarks, and CLI | Planned |
| [`build_shared_prefix`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2237) | Function | Demos, benchmarks, and CLI | Planned |
| [`load_benchmark_prompts`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2254) | Function | Demos, benchmarks, and CLI | Planned |
| [`load_benchmark_prompts.encode`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2273) | Nested function | Demos, benchmarks, and CLI | Planned |
| [`hf_static_generate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2305) | Function | Demos, benchmarks, and CLI | Planned |
| [`tinyserve_generate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2357) | Function | Demos, benchmarks, and CLI | Planned |
| [`benchmark`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2379) | Function | Demos, benchmarks, and CLI | Planned |
| [`main`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L2474) | Function | Demos, benchmarks, and CLI | Planned |

</details>

<details>
<summary>serve.py.before_quantization</summary>

| Definition | Kind | Chapter | Status |
| --- | --- | --- | --- |
| [`Config`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L49) | Class | Legacy snapshot | Planned |
| [`Config.__post_init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L71) | Method | Legacy snapshot | Planned |
| [`SamplingParams`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L86) | Class | Legacy snapshot | Planned |
| [`Request`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L97) | Class | Legacy snapshot | Planned |
| [`Request.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L110) | Method | Legacy snapshot | Planned |
| [`Request.num_tokens`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L138) | Property | Legacy snapshot | Planned |
| [`Request.output_ids`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L142) | Property | Legacy snapshot | Planned |
| [`KVCacheManager`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L146) | Class | Legacy snapshot | Planned |
| [`KVCacheManager.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L221) | Method | Legacy snapshot | Planned |
| [`KVCacheManager.num_free`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L247) | Property | Legacy snapshot | Planned |
| [`KVCacheManager._update_hashes`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L250) | Method | Legacy snapshot | Planned |
| [`KVCacheManager.find_cached_prefix`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L300) | Method | Legacy snapshot | Planned |
| [`KVCacheManager.allocate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L357) | Method | Legacy snapshot | Planned |
| [`KVCacheManager._register_full_blocks`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L447) | Method | Legacy snapshot | Planned |
| [`KVCacheManager.free_request`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L473) | Method | Legacy snapshot | Planned |
| [`ScheduledRequest`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L484) | Class | Legacy snapshot | Planned |
| [`ScheduledRequest.completes_prompt`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L490) | Property | Legacy snapshot | Planned |
| [`Scheduler`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L505) | Class | Legacy snapshot | Planned |
| [`Scheduler.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L516) | Method | Legacy snapshot | Planned |
| [`Scheduler.add`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L522) | Method | Legacy snapshot | Planned |
| [`Scheduler.has_work`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L525) | Method | Legacy snapshot | Planned |
| [`Scheduler.schedule`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L528) | Method | Legacy snapshot | Planned |
| [`Scheduler.finish`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L671) | Method | Legacy snapshot | Planned |
| [`AttnMetadata`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L680) | Class | Legacy snapshot | Planned |
| [`torch_ref_attention`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L744) | Function | Legacy snapshot | Planned |
| [`flash_attention`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L820) | Function | Legacy snapshot | Planned |
| [`RMSNorm`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L837) | Class | Legacy snapshot | Planned |
| [`RMSNorm.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L838) | Method | Legacy snapshot | Planned |
| [`RMSNorm.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L843) | Method | Legacy snapshot | Planned |
| [`rotary`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L848) | Function | Legacy snapshot | Planned |
| [`Attention`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L853) | Class | Legacy snapshot | Planned |
| [`Attention.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L854) | Method | Legacy snapshot | Planned |
| [`Attention.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L875) | Method | Legacy snapshot | Planned |
| [`MLP`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L911) | Class | Legacy snapshot | Planned |
| [`MLP.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L912) | Method | Legacy snapshot | Planned |
| [`MLP.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L918) | Method | Legacy snapshot | Planned |
| [`DecoderLayer`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L921) | Class | Legacy snapshot | Planned |
| [`DecoderLayer.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L922) | Method | Legacy snapshot | Planned |
| [`DecoderLayer.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L929) | Method | Legacy snapshot | Planned |
| [`Model`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L934) | Class | Legacy snapshot | Planned |
| [`Model.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L935) | Method | Legacy snapshot | Planned |
| [`Model.forward`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L960) | Method | Legacy snapshot | Planned |
| [`Model.logits`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L967) | Method | Legacy snapshot | Planned |
| [`load_model`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L971) | Function | Legacy snapshot | Planned |
| [`sample`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L991) | Function | Legacy snapshot | Planned |
| [`ModelRunner`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1052) | Class | Legacy snapshot | Planned |
| [`ModelRunner.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1076) | Method | Legacy snapshot | Planned |
| [`ModelRunner.allocate_kv_cache`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1116) | Method | Legacy snapshot | Planned |
| [`ModelRunner._prepare`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1179) | Method | Legacy snapshot | Planned |
| [`ModelRunner.graph_sizes`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1281) | Method | Legacy snapshot | Planned |
| [`ModelRunner.capture_cuda_graphs`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1290) | Method | Legacy snapshot | Planned |
| [`ModelRunner.run`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1337) | Method | Legacy snapshot | Planned |
| [`Output`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1368) | Class | Legacy snapshot | Planned |
| [`Detokenizer`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1379) | Class | Legacy snapshot | Planned |
| [`Detokenizer.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1391) | Method | Legacy snapshot | Planned |
| [`Detokenizer.update`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1404) | Method | Legacy snapshot | Planned |
| [`Engine`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1462) | Class | Legacy snapshot | Planned |
| [`Engine.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1466) | Method | Legacy snapshot | Planned |
| [`Engine.add_request`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1497) | Method | Legacy snapshot | Planned |
| [`Engine.abort`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1522) | Method | Legacy snapshot | Planned |
| [`Engine.has_work`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1529) | Method | Legacy snapshot | Planned |
| [`Engine.step`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1532) | Method | Legacy snapshot | Planned |
| [`Engine.generate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1590) | Method | Legacy snapshot | Planned |
| [`AsyncEngine`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1602) | Class | Legacy snapshot | Planned |
| [`AsyncEngine.__init__`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1608) | Method | Legacy snapshot | Planned |
| [`AsyncEngine._loop`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1625) | Method | Legacy snapshot | Planned |
| [`AsyncEngine.generate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1680) | Async method | Legacy snapshot | Planned |
| [`build_app`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1713) | Function | Legacy snapshot | Planned |
| [`build_app.get`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1728) | Nested function | Legacy snapshot | Planned |
| [`build_app.params_from`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1732) | Nested function | Legacy snapshot | Planned |
| [`build_app.error`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1747) | Nested function | Legacy snapshot | Planned |
| [`build_app.respond`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1751) | Async nested function | Legacy snapshot | Planned |
| [`build_app.respond.choice`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1772) | Nested function | Legacy snapshot | Planned |
| [`build_app.respond.usage`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1781) | Nested function | Legacy snapshot | Planned |
| [`build_app.respond.events`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1795) | Async nested function | Legacy snapshot | Planned |
| [`build_app.health`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1824) | Async nested function | Legacy snapshot | Planned |
| [`build_app.models`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1829) | Async nested function | Legacy snapshot | Planned |
| [`build_app.metrics`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1836) | Async nested function | Legacy snapshot | Planned |
| [`build_app.completions`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1846) | Async nested function | Legacy snapshot | Planned |
| [`build_app.chat`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1852) | Async nested function | Legacy snapshot | Planned |
| [`kv_cache_demo`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1876) | Function | Legacy snapshot | Planned |
| [`kv_cache_demo.cached_pages`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1880) | Nested function | Legacy snapshot | Planned |
| [`kv_cache_demo.dump`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1883) | Nested function | Legacy snapshot | Planned |
| [`attention_demo`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L1950) | Function | Legacy snapshot | Planned |
| [`build_shared_prefix`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L2041) | Function | Legacy snapshot | Planned |
| [`load_benchmark_prompts`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L2058) | Function | Legacy snapshot | Planned |
| [`load_benchmark_prompts.encode`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L2077) | Nested function | Legacy snapshot | Planned |
| [`hf_static_generate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L2109) | Function | Legacy snapshot | Planned |
| [`tinyserve_generate`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L2161) | Function | Legacy snapshot | Planned |
| [`benchmark`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L2183) | Function | Legacy snapshot | Planned |
| [`main`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py.before_quantization#L2278) | Function | Legacy snapshot | Planned |

</details>
