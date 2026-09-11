# 28 · Model adapters, K-quants and Hugging Face preflight

**Phase:** now · **Status:** in progress

## Why

FIELD STATION can already run Qwen and experimental Llama-family GGUFs, but "GGUF" is a container rather than an architecture guarantee. Broader model support has separate boundaries: tensor/quantisation codecs, model architecture, rotary/attention behaviour, and tokenizer/chat behaviour.

## Capability preflight

**Inspect another GGUF…** reads only the GGUF header/index using HTTP Range requests and reports architecture, layer/tensor count, tensor formats, runtime-memory estimate and an explicit compatibility verdict. Supported Qwen3 and Llama GGUFs can be selected from the inspected URL without a server-side model registry.

Llama 3.1/3.2 `llama3` scaled-RoPE files are now recognised when they contain llama.cpp's exact `rope_freqs.weight` tensor. Other/unrecognised scaling schemes still fail closed.

## K-quant compatibility

Common mixed K-quants use a correctness-first path: decode Q4_K/Q5_K/Q6_K and requantise to the existing Q8 GPU representation. `Q4_K_M` is treated as a mixed tensor recipe. Native Q4_K remains an optimisation only if field benchmarks justify it.

## Architecture adapters

`engine/model-adapters.js` owns model-family detection, runtime config, chat format, stop tokens and RoPE behaviour.

Current experimental Llama path:

- original Llama 3: adjacent/interleaved rotary pairs matching llama.cpp's GGUF Q/K permutation;
- Llama 3.1/3.2: same layout plus exact per-frequency factors loaded from `rope_freqs.weight` on every shard;
- original Llama/Qwen use identity frequency factors, keeping their existing behaviour;
- unknown RoPE scaling types fail closed.

Built-in experimental targets, deliberately using native Q4_0 so architecture work is not confounded with K-quant conversion:

1. **Llama 3.2 1B Instruct Q4_0** — cheap scaled-RoPE verification target;
2. **Llama 3.2 3B Instruct Q4_0** — useful small collective model;
3. **Meta-Llama-3 8B Instruct Q4_0** — first proven-to-load non-Qwen architecture;
4. **Llama 3.1 8B** — next once the 3.2 scaled-RoPE reference run passes;
5. **Llama 3 70B** — later collective-compute experiment requiring embedding/head sharding as well as layer distribution.

After Llama: Gemma 2/3 text, then Phi/Mistral families; MoE/expert placement remains later research.

## Room-safe custom model descriptors

Custom models remain ephemeral room state. Only a sanitised model label, HTTPS GGUF URL, kind marker and memory estimate cross the authenticated room control channel. Arbitrary remote configuration is not accepted.

## Reference verification

Every architecture × quantisation combination needs deterministic reference evidence:

- first-token/top-logit comparison against llama.cpp;
- deterministic greedy output on one FIELD STATION device;
- equivalent output across two- and three-device splits;
- load time, memory, TTFT and tok/s recorded in `docs/bench-log.md`.

A green build or plausible text is not sufficient evidence.

## Implementation state

- [x] Q4_K codec/accounting and Q4_K/Q5_K/Q6_K → Q8 compatibility path.
- [x] GGUF capability registry and browser preflight.
- [x] Qwen3 self-describing dense GGUF runtime and adapter-owned chat/stops.
- [x] Sanitised custom GGUF propagation through rooms.
- [x] Original Llama 3 dense adapter, Llama chat framing and single-conversation BOS.
- [x] Interleaved RoPE matching canonical Llama GGUF Q/K permutation.
- [x] Load and apply exact `rope_freqs.weight` factors for `llama3` scaled RoPE on every shard.
- [x] Add experimental Llama 3.2 1B and 3B Q4_0 catalogue targets.
- [x] Add Vercel build assertions covering scaled RoPE, catalogue sources and unchanged original-Llama behaviour.
- [ ] Run the real Llama 3.2 1B Q4_0 and compare deterministic output/top logits with llama.cpp.
- [ ] Repeat reference check with Llama 3.2 3B Q4_0.
- [ ] Validate one/two/three-device equivalence for Llama 3.2 and Llama 3 8B.
- [ ] Validate real Qwen3 Q4_K_M against llama.cpp and add golden.
- [ ] Add Llama 3.1 8B after scaled-RoPE evidence lands.
- [ ] Explore embedding/head sharding required for a 70B room.
- [ ] Add native Q4_K only if benchmarks justify it.

## Done when

- inspected model URLs get truthful architecture/quantisation/feature verdicts;
- Qwen Q4_K_M has deterministic reference evidence;
- original Llama 3 and scaled-RoPE Llama 3.x have deterministic reference evidence and equivalent multi-device runs;
- a Gemma-family model runs across at least two devices;
- adding a compatible dense model no longer requires changing the generation loop.
