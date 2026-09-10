# 28 · Model adapters, K-quants and Hugging Face preflight

**Phase:** now · **Status:** in progress

## Why

FIELD STATION can currently run a narrow set of Qwen-family GGUFs well, but "GGUF" is a container rather than an architecture guarantee. Broader model support has three separate boundaries: tensor/quantisation codecs, model architecture, and tokenizer/chat behaviour. Keeping those boundaries explicit lets us add Llama, Gemma, Phi, Mistral and future models without scattering model-name conditionals through the room runtime.

This is also part of the experiment: a room should be able to inspect a model before committing to a multi-gigabyte download and say what this collection of devices can actually run.

## Design

### A. Capability preflight

Add a no-weight-download inspector over the GGUF header. It reports:

- `general.architecture` and the FIELD STATION adapter that would execute it;
- layer count and tensor count;
- every GGML tensor type present;
- whether each type is native, converted, codec-ready, or unsupported;
- estimated tensor bytes;
- one explicit verdict: supported, supported-with-conversion, integration-needed, architecture-needed, or unsupported.

The room UI can later expose this behind **Other model… / Load from Hugging Face**.

### B. K-quant compatibility path

Support common mixed K-quant GGUFs in two stages:

1. **correctness path:** decode Q4_K/Q5_K/Q6_K and stream-requantise unsupported GPU formats to the existing Q8 representation;
2. **performance path:** add native WebGPU matvec kernels for the formats that materially reduce room memory or load time, starting with Q4_K.

`Q4_K_M` is treated as a mixed tensor recipe, not a new tensor encoding. Compatibility is decided tensor by tensor.

### C. Architecture adapters

Extract model-specific behaviour behind a small adapter contract rather than adding branches to `room.js`:

- detection / metadata keys;
- tensor name mapping;
- layer/state plan;
- attention/RoPE/activation quirks;
- tokenizer and chat template;
- stop tokens;
- memory estimation.

Target order:

1. existing Qwen3 dense and Qwen3.5/3.8 hybrid expressed through the adapter boundary;
2. Llama dense as the first proof that the boundary is genuinely reusable;
3. Gemma dense/text path (Gemma 2/3 first; Gemma 4 separately where its architecture needs it);
4. Phi and Mistral-family variants;
5. MoE expert placement as a later distributed-compute experiment (links to item 10).

### D. Reference verification

Every architecture × quantisation combination needs deterministic reference evidence:

- first-token/top-logit comparison against llama.cpp or another trusted implementation;
- deterministic greedy output on one device;
- one-device output agrees with two- and three-device splits within the documented tolerance;
- benchmark row records load time, peak memory and tok/s.

## First implementation slice

- [x] Add standalone Q4_K block decoder and byte accounting from the ggml layout.
- [x] Add a model capability registry / GGUF header inspector.
- [x] Add no-GPU unit coverage for Q4_K layout and architecture-vs-quant verdicts.
- [x] Wire Q4_K into the FIELD STATION distributable `engine/gguf.js` at build time, preserving the upstream source file while eligible 2D tensors take the existing requant-to-Q8 path.
- [x] Add a small Qwen3 0.6B `Q4_K_M` model to the room as an explicitly experimental field test.
- [ ] Validate that real Qwen3 `Q4_K_M` GGUF end-to-end against llama.cpp and add a golden.
- [ ] Surface the preflight report in the room model picker.

## Done when

- A pasted Hugging Face GGUF URL can be inspected before weight download and gets a truthful compatibility verdict.
- A Qwen-family Q4_K_M model runs end-to-end through the conversion path and has a golden test.
- At least one Llama-family and one Gemma-family model run on one device and across a two-device room.
- The architecture adapter contract is documented and adding a compatible model no longer requires editing the generation loop.
- Native Q4_K GPU support is either shipped or has benchmark evidence showing conversion remains the better trade-off.
