# Models

SwarmLLM ships no weights. Browsers fetch tensors by HTTP range request from public Hugging Face repositories; local development and tests read the same files from `models/` (git-ignored).

FIELD STATION is broadening this from a fixed Qwen-oriented catalogue into a capability-based runtime. A GGUF file is a container, not a promise that the model architecture or every tensor quantisation is executable, so compatibility is checked across **architecture**, **tensor formats**, and **tokenizer/chat behaviour** separately.

## Supported and experimental models

| Model | File | Engine | Notes |
|---|---|---|---|
| Qwen 3.8 27B | GGUF Q4_0 (~15 GB), includes the `nextn` draft layer | `Qwen35Engine` | 64 layers: 48 Gated DeltaNet + 16 attention; MTP speculation |
| Qwen3 4B / 1.7B / 0.6B | GGUF Q4_0 / Q8_0 | `DenseEngine` | dense; 0.6B is the golden-test model |
| Qwen3 0.6B Q4_K_M | mixed K-quant GGUF | `DenseEngine` through Q8 conversion | experimental field model; real-model golden still required |
| Meta-Llama-3-8B-Instruct | GGUF Q4_0 (~4.7 GB) | `DenseEngine` with interleaved RoPE | experimental cross-architecture target; real-model reference and multi-device golden still required |
| SmolLM2 135M | safetensors f32 | `DenseEngine` | smallest demo; quantised to Q8 at load |

The Llama entry is intentionally **experimental rather than verified**. Its metadata, chat format, tensor shapes and canonical GGUF rotary layout are implemented, but support is not considered complete until the exact target file agrees with llama.cpp and produces equivalent one-, two- and three-device results.

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

The room exposes **Inspect another GGUF…** below the built-in model selector.

Paste an HTTPS `.gguf` URL and FIELD STATION reads only enough of the file's header/index, using HTTP Range requests, to report:

- `general.architecture` and the candidate FIELD STATION adapter;
- layer and tensor counts;
- all GGML tensor types present;
- whether each tensor format is native, converted or unsupported;
- GGUF tensor bytes and estimated runtime weight bytes;
- an explicit verdict such as `supported`, `supported-with-conversion`, `architecture-needed`, or `unsupported`.

For a currently executable **Qwen3 dense or original Llama 3** GGUF, **Use this model** turns that single URL into an ephemeral room model. Starting it propagates only a sanitised descriptor—label, HTTPS GGUF URL and memory estimate—to the model dealer and workers. There is no server-side model registry and no arbitrary remote config object is accepted.

A remote host must allow browser CORS and HTTP Range requests. Header compatibility is deliberately not treated as proof of numerical correctness: a new architecture/quantisation combination still needs reference evidence before it becomes a verified built-in model.

### Llama boundary

Original Llama 3 and later Llama-family checkpoints can all appear as `general.architecture = llama`, but they are not computationally identical. The current FIELD STATION adapter supports the original Llama 3 dense path and rejects GGUFs declaring non-`none` `llama.rope.scaling.type` metadata. That keeps Llama 3.1/3.2-style scaled RoPE out until the kernel implements and verifies the relevant frequency scaling.

Canonical llama.cpp GGUF conversion permutes Llama Q/K projection rows so the rotary pairs are adjacent. FIELD STATION therefore switches the shared DenseEngine RoPE kernel into an adapter-selected **interleaved** mode for Llama while leaving Qwen on its existing half-split layout. This preserves the GGUF weights exactly as stored and retains Q4_0 network-to-GPU streaming.

## Self-describing dense GGUFs

`engine/model-adapters.js` is the architecture-adapter boundary. For Qwen3 dense and original Llama 3 GGUFs it reconstructs the `DenseEngine` configuration from GGUF metadata and tensor shapes, including hidden size, layer count, attention/KV heads, head dimension, FFN size, vocabulary size, RMS epsilon, RoPE base and RoPE layout.

The host obtains tokenizer vocabulary/merges from GGUF metadata when it loads the embedding/head. Workers do not build a tokenizer because they only execute assigned hidden layers. Conversation representation is also adapter-owned: Qwen uses its ChatML envelope; Llama 3 uses one `<|begin_of_text|>` for the conversation plus role header and end-of-turn tokens.

## Architecture roadmap

| Architecture | FIELD STATION status | Intended adapter |
|---|---|---|
| Qwen3 dense | supported; adapter-owned config/chat | `qwen3-dense` / `DenseEngine` |
| Qwen 3.5/3.8 hybrid | supported; adapter migration continuing | `qwen35` |
| Original Llama 3 dense | experimental implementation; numerical + distributed verification next | `llama-dense` / `DenseEngine` |
| Scaled-RoPE Llama variants | deliberately blocked pending kernel + reference evidence | `llama-dense` |
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
  llama3/  model.gguf (Meta-Llama-3-8B-Instruct Q4_0; local/reference only)
  q38/     model.gguf (Qwen 3.8 27B Q4_0)
  model/   SmolLM2-135M: model.safetensors, tokenizer.json, config.json
```

The external JSON files in the test layout remain useful to reference tests; browser dense-GGUF execution derives its runtime description and tokenizer from the GGUF itself.

## Adding a model today

1. Use **Inspect another GGUF…** or `engine/model-capabilities.js` to inspect metadata and tensor types before downloading the model body.
2. Confirm the architecture maps to an implemented adapter and that the adapter can reconstruct every runtime dimension it needs.
3. Confirm tensor names, rotary layout and dimensions against the architecture's loader/name map.
4. For a built-in catalogue entry, use a reviewed/pinned model source rather than relying permanently on `resolve/main`.
5. Generate a deterministic golden with `tests/reference/` and compare it with a trusted reference implementation such as llama.cpp.
6. Verify one-device output, then two- and three-device splits.
7. Record load time, memory and tok/s in `docs/bench-log.md`.

Do not label a model verified merely because its GGUF header parses or it generates plausible text. A new architecture is verified only once its forward pass, chat/tokenizer behaviour and distributed split have reference evidence.
