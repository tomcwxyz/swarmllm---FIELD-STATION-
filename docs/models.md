# Models

SwarmLLM ships no weights. Browsers fetch tensors by HTTP range request from public Hugging Face repositories; local development and tests read the same files from `models/` (git-ignored).

FIELD STATION is broadening this from a fixed Qwen-oriented catalogue into a capability-based runtime. A GGUF file is a container, not a promise that the model architecture or every tensor quantisation is executable, so compatibility is checked across **architecture**, **tensor formats**, and **tokenizer/chat behaviour** separately.

## Supported models

| Model | File | Engine | Notes |
|---|---|---|---|
| Qwen 3.8 27B | GGUF Q4_0 (~15 GB), includes the `nextn` draft layer | `Qwen35Engine` | 64 layers: 48 Gated DeltaNet + 16 attention; MTP speculation |
| Qwen3 4B / 1.7B / 0.6B | GGUF Q4_0 / Q8_0 | `DenseEngine` | dense; 0.6B is the golden-test model |
| Qwen3 0.6B Q4_K_M | mixed K-quant GGUF | `DenseEngine` through Q8 conversion | experimental field model; real-model golden still required |
| SmolLM2 135M | safetensors f32 | `DenseEngine` | smallest demo; quantised to Q8 at load |

Architectures sharing Qwen 3.5/3.6/3.8's hybrid layout can be added mainly as catalogue/configuration work once their exact GGUF metadata and tensor names have been verified.

## Quantisation capability

| GGML tensor type | Current FIELD STATION path | Status |
|---|---|---|
| F32 / F16 | float path | supported |
| Q4_0 | native WebGPU representation | supported |
| Q8_0 | native WebGPU representation | supported |
| Q4_1 | decode then requantise to Q8 | supported with conversion |
| Q4_K | FIELD STATION build patch decodes then requantises to Q8 | supported with conversion; real-model golden pending |
| Q5_K / Q6_K | decode then requantise to Q8 | supported with conversion |

`Q4_K_M` is not a separate tensor encoding. It is a mixed quantisation recipe: individual tensors in one GGUF may use Q4_K, Q5_K, Q6_K or another type. FIELD STATION therefore decides compatibility tensor by tensor.

The first Q4_K implementation is deliberately a correctness path. It uses the ggml `block_q4_K` layout in `engine/q4k.js`, then feeds the existing Q8 GPU representation. It does not yet get Q4_0's low-memory direct network-to-GPU streaming path, so conversion-heavy models can have a higher load-time CPU/GPU memory peak. A native Q4_K WebGPU/streaming path is a later optimisation once real-model benchmarks show where it matters.

## Inspect another GGUF

The room now exposes **Inspect another GGUF…** below the built-in model selector.

Paste an HTTPS `.gguf` URL and FIELD STATION reads only enough of the file's header/index, using HTTP Range requests, to report:

- `general.architecture` and the candidate FIELD STATION adapter;
- layer and tensor counts;
- all GGML tensor types present;
- whether each tensor format is native, converted or unsupported;
- GGUF tensor bytes and estimated runtime weight bytes;
- an explicit verdict such as `supported`, `supported-with-conversion`, `architecture-needed`, or `unsupported`.

For a currently compatible **Qwen3 dense** GGUF, **Use this model** turns that single URL into an ephemeral room model. Starting it propagates only a sanitised descriptor—label, HTTPS GGUF URL and memory estimate—to the model dealer and workers. There is no server-side model registry and no arbitrary remote config object is accepted.

A remote host must allow browser CORS and HTTP Range requests. Header compatibility is deliberately not treated as proof of numerical correctness: a new architecture/quantisation combination still needs reference evidence before it becomes a built-in supported model.

## Self-describing Qwen3 GGUFs

`engine/model-adapters.js` is the first architecture-adapter boundary. For Qwen3 dense GGUFs it reconstructs the `DenseEngine` configuration from GGUF metadata and tensor shapes, including hidden size, layer count, attention/KV heads, head dimension, FFN size, vocabulary size, RMS epsilon and RoPE base.

This means the dense Qwen3 path no longer relies on separate `config.json` and `tokenizer.json` URLs. The host obtains tokenizer vocabulary/merges from GGUF metadata when it loads the embedding/head. Workers do not build a tokenizer because they only execute assigned hidden layers.

## Architecture roadmap

| Architecture | FIELD STATION status | Intended adapter |
|---|---|---|
| Qwen3 dense | supported; adapter boundary started | `qwen3-dense` / `DenseEngine` |
| Qwen 3.5/3.8 hybrid | supported; adapter migration next | `qwen35` |
| Llama dense | next implementation target | `llama-dense` |
| Gemma 2/3 text | planned | `gemma` |
| Gemma 4 | research | `gemma` / architecture-specific additions |
| Phi 3/4 | planned | `phi` |
| Mistral-family dense | planned | `mistral` |
| MoE | later research | expert placement / item 10 |

The finished adapter boundary will own model detection, config reconstruction, tensor mapping, layer/state plans, attention/RoPE/activation differences, tokenizer/chat templates, stop tokens and memory estimation. See [`roadmap/28-model-adapters-and-k-quants.md`](../roadmap/28-model-adapters-and-k-quants.md).

## Local layout for tests and benchmarks

```
models/
  qwen/    model.gguf (Qwen3-0.6B Q8_0), model-q4.gguf, tokenizer.json, config.json
  qwen17/  model.gguf, tokenizer.json, config.json
  q38/     model.gguf (Qwen 3.8 27B Q4_0)
  model/   SmolLM2-135M: model.safetensors, tokenizer.json, config.json
```

The external JSON files in the test layout remain useful to reference tests; the browser Qwen3 GGUF runtime no longer requires them.

## Adding a model today

1. Use **Inspect another GGUF…** or `engine/model-capabilities.js` to inspect metadata and tensor types before downloading the model body.
2. Confirm the architecture maps to an implemented adapter and that the adapter can reconstruct every runtime dimension it needs.
3. Confirm tensor names and dimensions against the architecture's loader/name map.
4. For a built-in catalogue entry, use a reviewed/pinned model source rather than relying permanently on `resolve/main`.
5. Generate a deterministic golden with `tests/reference/` and compare it with a trusted reference implementation such as llama.cpp.
6. Verify one-device output, then two- and three-device splits.
7. Record load time, memory and tok/s in `docs/bench-log.md`.

Do not label a model supported merely because its GGUF header parses. A new architecture is supported only once its forward pass, chat/tokenizer behaviour and distributed split have reference evidence.