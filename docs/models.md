# Models

SwarmLLM ships no weights. Browsers fetch tensors by HTTP range request from public Hugging Face repositories; local development and tests read the same files from `models/` (git-ignored).

FIELD STATION is broadening this from a fixed Qwen-oriented catalogue into a capability-based runtime. A GGUF file is a container, not a promise that the model architecture or every tensor quantisation is executable, so compatibility is checked across **architecture**, **tensor formats**, and **tokenizer/chat behaviour** separately.

## Supported models

| Model | File | Engine | Notes |
|---|---|---|---|
| Qwen 3.8 27B | GGUF Q4_0 (~15 GB), includes the `nextn` draft layer | `Qwen35Engine` | 64 layers: 48 Gated DeltaNet + 16 attention; MTP speculation |
| Qwen3 4B / 1.7B / 0.6B | GGUF Q4_0 / Q8_0 | `DenseEngine` | dense; 0.6B is the golden-test model |
| SmolLM2 135M | safetensors f32 | `DenseEngine` | smallest demo; quantized to Q8 at load |

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

The first Q4_K implementation is deliberately a correctness path. It uses the ggml `block_q4_K` layout in `engine/q4k.js`, then feeds the existing streaming Q8 GPU representation. A native Q4_K WebGPU matvec is a later optimisation once real-model benchmarks show where it matters.

## Compatibility preflight

`engine/model-capabilities.js` now inspects an already-parsed GGUF header without fetching tensor bodies. It reports:

- `general.architecture` and the candidate FIELD STATION adapter;
- layer and tensor counts;
- all GGML tensor types present;
- whether each is native, converted or unsupported;
- estimated tensor bytes;
- an explicit overall verdict such as `supported`, `supported-with-conversion`, `architecture-needed`, or `unsupported`.

This is currently a programmatic capability. The next UI step is **Other model… / Load from Hugging Face**, where the room reads the header first and shows the verdict before participants commit to downloading weights.

## Architecture roadmap

| Architecture | FIELD STATION status | Intended adapter |
|---|---|---|
| Qwen3 dense | supported | `dense` |
| Qwen 3.5/3.8 hybrid | supported | `qwen35` |
| Llama dense | planned | `llama` |
| Gemma 2/3 text | planned | `gemma` |
| Gemma 4 | research | `gemma` / architecture-specific additions |
| Phi 3/4 | planned | `phi` |
| Mistral-family dense | planned | `mistral` |
| MoE | later research | expert placement / item 10 |

The adapter boundary will own model detection, tensor mapping, layer/state plans, attention/RoPE/activation differences, tokenizer/chat templates, stop tokens and memory estimation. See [`roadmap/28-model-adapters-and-k-quants.md`](../roadmap/28-model-adapters-and-k-quants.md).

## Local layout for tests and benchmarks

```
models/
  qwen/    model.gguf (Qwen3-0.6B Q8_0), model-q4.gguf, tokenizer.json, config.json
  qwen17/  model.gguf, tokenizer.json, config.json
  q38/     model.gguf (Qwen 3.8 27B Q4_0)
  model/   SmolLM2-135M: model.safetensors, tokenizer.json, config.json
```

## Adding a model today

1. Inspect the GGUF metadata and tensor types using the capability layer; confirm the architecture maps to an implemented engine.
2. Confirm tensor names and dimensions against the architecture's loader/name map.
3. Add the model entry to `room/models.js` with its pinned/reviewed source, layer/memory information and tokenizer/config sources where needed.
4. Generate a deterministic golden with `tests/reference/` and compare it with a trusted reference implementation such as llama.cpp.
5. Verify one-device output, then two- and three-device splits.
6. Record load time, memory and tok/s in `docs/bench-log.md`.

Do not label a model supported merely because its GGUF header parses. A new architecture is supported only once its forward pass, chat/tokenizer behaviour and distributed split have reference evidence.
