# 28 · Model adapters, K-quants and Hugging Face preflight

**Phase:** now · **Status:** in progress

## Why

FIELD STATION can already run Qwen and experimental Llama-family GGUFs, but "GGUF" is a container rather than an architecture guarantee. Broader model support has separate boundaries: tensor/quantisation codecs, model architecture, rotary/attention behaviour, tokenizer/chat behaviour, and — at larger scales — whether individual global tensors still fit a browser GPU binding.

## Capability preflight

**Inspect another GGUF…** reads only the GGUF header/index using HTTP Range requests and reports architecture, layer/tensor count, tensor formats, runtime-memory estimate and an explicit compatibility verdict. Supported Qwen3 and Llama GGUFs can be selected from the inspected URL without a server-side model registry.

Llama 3.1/3.2 `llama3` scaled-RoPE files are recognised when they contain llama.cpp's exact `rope_freqs.weight` tensor. Other/unrecognised scaling schemes still fail closed.

## K-quant compatibility

Common mixed K-quants use a correctness-first path: decode Q4_K/Q5_K/Q6_K and requantise to the existing Q8 GPU representation. `Q4_K_M` is treated as a mixed tensor recipe. Native Q4_K remains an optimisation only if field benchmarks justify it.

For the first 70B experiment we deliberately use **Q4_0**, not Q4_K_M. Native Q4_0 streams directly into the existing GPU representation; converting a 70B K-quant to Q8 would erase much of the memory advantage and confound the collective-compute experiment with quantisation work.

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
4. **Meta-Llama-3 70B Instruct Q4_0** — ~40 GB model / ~41.5 GB room target; first deliberately room-scale Llama experiment;
5. **Llama 3.1 8B** — next scaled-RoPE 8B reference once the 3.2 evidence is complete.

After Llama: Gemma 2/3 text, then Phi/Mistral families; MoE/expert placement remains later research.

## 70B collective-compute path

The first 70B route no longer assumes that embedding/head sharding is required before testing. Original Llama 3 70B has an untied output head. On a capable host, FIELD STATION can:

- keep the quantised token embedding CPU-side for per-token row lookup;
- stream the separate `output.weight` directly to GPU;
- avoid the previous temporary duplicate GPU upload of the embedding;
- select the room host only from devices whose WebGPU storage-binding limit can hold the large output-head quant buffer;
- distribute the 80 transformer layers according to pledged capacity;
- include per-layer K/V cache memory in the room planner;
- skip Cache API cloning for giant embedding/head range responses, avoiding another hundreds-of-megabytes browser copy.

The built-in 70B target currently asks for about **41.5 GB pledged across the room** and a host with at least **0.55 GB per WebGPU storage binding**. The current room context is still 2,048 tokens.

This is an engineering-ready experiment, **not yet evidence that 70B works**. The remaining high-risk step is the real browser load. In particular, the host still has to fetch and repack the compact quantised embedding CPU-side. If that transient allocation is the next failure, the next intervention is a streaming CPU embedding repack (or GPU embedding lookup), not full transformer redesign.

Full embedding/head sharding remains useful later for lower-limit hosts and broader hardware portability; it is no longer a prerequisite for the first 70B field run.

## Room-safe custom model descriptors

Custom models remain ephemeral room state. Only a sanitised model label, HTTPS GGUF URL, kind marker and memory estimate cross the authenticated room control channel. Arbitrary remote configuration is not accepted.

## Reference verification

Every architecture × quantisation combination needs deterministic reference evidence:

- first-token/top-logit comparison against llama.cpp;
- deterministic greedy output on one FIELD STATION device where the model can fit;
- equivalent output across multi-device splits;
- load time, memory, TTFT and tok/s recorded in `docs/bench-log.md` / experiment notes.

For 70B, a solo FIELD STATION run is not required as the first milestone because the point of the experiment is precisely that ordinary individual devices cannot hold it. The reference remains the exact same GGUF in llama.cpp on hardware capable of running it; FIELD STATION evidence starts with a multi-device room.

A green build, a successful download, or plausible text is not sufficient evidence.

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
- [x] Add Meta-Llama-3 70B Instruct Q4_0 as a built-in collective target.
- [x] Remove duplicate GPU embedding upload for untied Llama output heads.
- [x] Make dense room planning include per-layer K/V cache memory.
- [x] Record per-device storage-binding limits and require a suitable 70B host.
- [x] Avoid caching/cloning giant model tensor ranges in browser memory.
- [x] Add build-gate assertions for the 70B architecture, catalogue target and planner safeguards.
- [ ] Run the real Llama 3.2 1B Q4_0 and compare deterministic output/top logits with llama.cpp.
- [ ] Repeat reference check with Llama 3.2 3B Q4_0.
- [ ] Validate one/two/three-device equivalence for Llama 3.2 and Llama 3 8B.
- [ ] Attempt first Llama 3 70B room load with >=41.5 GB pledged and record the exact failure/success boundary.
- [ ] Compare first 70B generated tokens against the exact QuantFactory Q4_0 llama.cpp reference.
- [ ] If host embedding parsing is the blocker, implement streaming CPU embedding repack / lookup.
- [ ] Explore global tensor sharding for hosts whose individual storage-binding limit cannot hold the output head.
- [ ] Validate real Qwen3 Q4_K_M against llama.cpp and add golden.
- [ ] Add Llama 3.1 8B after scaled-RoPE evidence lands.
- [ ] Add native Q4_K only if benchmarks justify it.

## Done when

- inspected model URLs get truthful architecture/quantisation/feature verdicts;
- Qwen Q4_K_M has deterministic reference evidence;
- original Llama 3 and scaled-RoPE Llama 3.x have deterministic reference evidence and equivalent multi-device runs;
- Llama 3 70B has at least one reproducible multi-device FIELD STATION run or a precisely documented browser/runtime boundary that explains what remains;
- a Gemma-family model runs across at least two devices;
- adding a compatible dense model no longer requires changing the generation loop.
