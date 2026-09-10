# 28 · Model adapters, K-quants and Hugging Face preflight

**Phase:** now · **Status:** in progress

## Why

FIELD STATION can currently run a narrow set of Qwen-family GGUFs well, but "GGUF" is a container rather than an architecture guarantee. Broader model support has three separate boundaries: tensor/quantisation codecs, model architecture, and tokenizer/chat behaviour. Keeping those boundaries explicit lets us add Llama, Gemma, Phi, Mistral and future models without scattering model-name conditionals through the room runtime.

This is also part of the experiment: a room should be able to inspect a model before committing to a multi-gigabyte download and say what this collection of devices can actually run.

## Design

### A. Capability preflight

Inspect the GGUF header before weight download and report:

- `general.architecture` and the FIELD STATION adapter that would execute it;
- layer count and tensor count;
- every GGML tensor type present;
- whether each type is native, converted, codec-ready, or unsupported;
- GGUF tensor bytes and estimated runtime weight bytes;
- one explicit verdict: supported, supported-with-conversion, integration-needed, architecture-needed, or unsupported.

This now exists in the room behind **Inspect another GGUF…**. Inspection happens in the browser using HTTP Range requests. A supported Qwen3 GGUF can be selected directly from that single URL; the model is still not downloaded until Start is pressed.

### B. K-quant compatibility path

Support common mixed K-quant GGUFs in two stages:

1. **correctness path:** decode Q4_K/Q5_K/Q6_K and requantise unsupported GPU formats to the existing Q8 representation;
2. **performance path:** add native WebGPU matvec/streaming for the formats that materially reduce room memory or load time, starting with Q4_K.

`Q4_K_M` is treated as a mixed tensor recipe, not a new tensor encoding. Compatibility is decided tensor by tensor.

### C. Architecture adapters

Move model-family behaviour behind a small adapter contract rather than adding branches to the generation loop:

- detection / metadata keys;
- config reconstruction from GGUF metadata and tensor shapes;
- tensor name mapping;
- layer/state plan;
- attention/RoPE/activation quirks;
- tokenizer and chat prompt format;
- stop tokens;
- memory estimation.

`engine/model-adapters.js` is the first implementation of this boundary. Qwen3 dense can now reconstruct the `DenseEngine` config from the GGUF itself, including dimensions that cannot safely be guessed from hidden size alone. The host no longer needs a separate `config.json` or `tokenizer.json` for dense Qwen3 GGUFs; tokenizer data comes from GGUF metadata when the host loads its embedding/head.

Target order:

1. Qwen3 dense and Qwen3.5/3.8 hybrid expressed through the adapter boundary;
2. **Llama dense** as the first proof that the boundary is genuinely reusable;
3. Gemma dense/text path (Gemma 2/3 first; Gemma 4 separately where its architecture needs it);
4. Phi and Mistral-family variants;
5. MoE expert placement as a later distributed-compute experiment (links to item 10).

### D. Room-safe custom model descriptors

Custom models are ephemeral room state, not server state. When a participant selects an inspected GGUF, FIELD STATION now propagates a deliberately small descriptor through the existing authenticated room control channel:

- model label;
- HTTPS `.gguf` URL;
- `kind: gguf` / custom marker;
- estimated memory need.

The receiving device validates that descriptor before adding it to its in-memory catalogue. Arbitrary model configuration is not accepted over the control channel. This lets the largest device become the dealer and lets workers/rejoining peers reconstruct the same custom model without introducing an account or backend model registry.

### E. Reference verification

Every architecture × quantisation combination needs deterministic reference evidence:

- first-token/top-logit comparison against llama.cpp or another trusted implementation;
- deterministic greedy output on one device;
- one-device output agrees with two- and three-device splits within the documented tolerance;
- benchmark row records load time, peak memory and tok/s.

## Implementation state

- [x] Add standalone Q4_K block decoder and byte accounting from the ggml layout.
- [x] Add a model capability registry / GGUF header inspector.
- [x] Add no-GPU unit coverage for Q4_K layout and architecture-vs-quant verdicts.
- [x] Wire Q4_K into the FIELD STATION distributable `engine/gguf.js` at build time; eligible 2D tensors take the existing requant-to-Q8 path.
- [x] Add a small Qwen3 0.6B `Q4_K_M` model as an explicitly experimental field test.
- [x] Add the first architecture adapter boundary and reconstruct Qwen3 dense runtime config from GGUF metadata/tensor shapes.
- [x] Remove the separate config/tokenizer dependency from the dense Qwen3 GGUF runtime path.
- [x] Surface GGUF compatibility preflight in the room model picker.
- [x] Propagate sanitised custom GGUF descriptors to the model dealer, workers and rejoining peers.
- [ ] Validate the real Qwen3 `Q4_K_M` GGUF end-to-end against llama.cpp and add a golden.
- [ ] Move Qwen prompt/stop-token behaviour behind the adapter contract.
- [ ] Land the first Llama-family adapter and reference model.
- [ ] Add native Q4_K streaming/WebGPU only if benchmarks justify it over conversion.

## Done when

- A pasted Hugging Face GGUF URL can be inspected before weight download and gets a truthful compatibility verdict.
- A Qwen-family Q4_K_M model runs end-to-end through the conversion path and has a golden test.
- At least one Llama-family and one Gemma-family model run on one device and across a two-device room.
- The architecture adapter contract owns prompt/tokenizer/runtime differences so adding a compatible model no longer requires editing the generation loop.
- Native Q4_K GPU support is either shipped or has benchmark evidence showing conversion remains the better trade-off.