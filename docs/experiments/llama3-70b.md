# FIELD STATION experiment · Llama 3 70B collective room

**Status:** implementation ready / field run pending  
**Question:** can several ordinary browser devices collectively run an open 70B language model that none of them could comfortably hold alone?

This is a FIELD STATION experiment, not a supported-model claim. A green build means the runtime knows how to plan and attempt the model. It does not mean a real multi-device room has successfully loaded or produced reference-equivalent output yet.

## Exact target

- Model: **Meta-Llama-3-70B-Instruct**
- GGUF: **Q4_0**
- File: `Meta-Llama-3-70B-Instruct.Q4_0.gguf`
- Source: `https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q4_0.gguf`
- Approximate GGUF size: ~40 GB
- FIELD STATION room requirement: **~41.5 GB pledged**
- Architecture: original Llama 3 dense transformer, 80 layers, 8192 hidden size, 64 attention heads, 8 KV heads, 128 head dimension
- RoPE: original Llama 3 / unscaled, `theta = 500000`
- FIELD STATION context for this experiment: 2,048 tokens

Use this exact GGUF for the llama.cpp reference too. Do not compare a QuantFactory FIELD STATION run against a separately converted 70B GGUF and call numerical differences a runtime failure.

## Why this target

Original Llama 3 70B is deliberately boring architecturally. FIELD STATION already has a working original-Llama-3 adapter at 8B: Llama chat framing, canonical GGUF Q/K permutation, adjacent/interleaved rotary pairs, GQA and Q4_0 streaming are all on the existing path.

That makes 70B primarily a **distributed-systems and browser-memory experiment** rather than a simultaneous model-family experiment.

Q4_0 is deliberate. FIELD STATION can convert some K-quants to Q8 for compatibility, but doing that to a 70B model would increase runtime memory and muddy the question we are trying to answer.

## What changed to make a first attempt reasonable

The first 70B path does not require full embedding/head sharding on every machine.

- The quantised token embedding remains CPU-side on the host for row lookup.
- Llama 3's separate `output.weight` streams directly to GPU for logits.
- FIELD STATION no longer briefly uploads the embedding as a duplicate GPU head before replacing it with `output.weight`.
- Devices advertise their WebGPU storage-binding limit. The 70B model will only choose a host whose single binding is large enough for the output-head quant buffer.
- The room planner includes the K/V cache cost for every assigned transformer layer, not only model-file bytes.
- Very large range responses are not cloned into Cache API storage while they are being consumed; this avoids an extra hundreds-of-megabytes browser allocation during embedding/head loading.

The catalogue currently requires a host with at least **0.55 GB per WebGPU storage binding**. This is a conservative gate for the Q4 output head, not the total memory requirement of the host.

## First-run topology

This experiment should begin with **multiple desktop/laptop browsers**. Do not make a phone carry the host/global tensors for the first run.

The room should show at least **41.5 GB pledged** before starting. More headroom is preferable because browser/GPU allocation behaviour is not perfectly predicted by model-file size.

Record for every device:

| Device | Browser / OS | GPU | Pledged GB | max buffer | max storage binding | Assigned layers |
|---|---|---|---:|---:|---:|---|
| host | | | | | | |
| peer 1 | | | | | | |
| peer 2 | | | | | | |
| … | | | | | | |

Also record whether devices are on the same LAN or crossing the internet.

## Milestones

Treat these as separate results:

1. **Header/index succeeds** — the 70B GGUF can be ranged and parsed in-browser.
2. **Layer plan succeeds** — 80 layers are dealt and no device is assigned beyond its pledge.
3. **Weights load** — every device reaches ready without browser reload, OOM or invalid GPU buffers.
4. **Cluster online** — all 80 layers are connected in the inference chain.
5. **First prefill completes** — a short prompt traverses the whole room without NaN or GPU validation error.
6. **First token is coherent** — useful, but not correctness evidence on its own.
7. **Reference check passes** — first token / top logits and greedy sequence agree with the exact GGUF in llama.cpp within the documented numerical tolerance.
8. **Repeatability** — a fresh room can reproduce the load and answer.

A failure at any one milestone is useful experimental evidence. Record the first failure rather than retrying until the original condition is lost.

## Initial prompts

Keep the first prompts short so loading and distributed decode are what we are testing, not context length.

**P1**  
`What is the capital of Wales? Answer in one sentence.`

**P2**  
`A shop has 17 apples and sells 6. How many remain? Explain in one short sentence.`

**P3**  
`Give exactly three colours, separated by commas, and nothing else.`

After deterministic reference evidence exists, use longer/contextual prompts and Sources.

## Reference run

Run the exact QuantFactory Q4_0 in llama.cpp with deterministic / greedy sampling. Record:

- llama.cpp commit/version;
- exact GGUF source and, where practical, checksum;
- prompt and chat template;
- first-token id;
- top logits or top candidates where available;
- complete greedy output token ids;
- hardware used for the reference.

FIELD STATION does not need to run 70B solo on an ordinary browser as a prerequisite. The point of the experiment is that the room makes an otherwise impractical model possible. The external llama.cpp run is the numerical reference.

## Evidence to record from FIELD STATION

For every attempt capture:

- room device roster and pledges;
- actual layer split;
- GPU limits for each device;
- model/index read time;
- per-device download/load time;
- whether the host survives `token_embd.weight` parsing/repack;
- cluster-online time;
- prompt token count;
- prefill time;
- time to first token;
- decode tok/s;
- output text and token ids if available;
- first GPU validation error, OOM, device-lost event or browser reload breadcrumb;
- network RTT/bandwidth if a peer appears to dominate latency;
- qualitative heat/battery observations only when useful.

## Expected next failure boundary

The architecture and individual transformer matrices are no longer the obvious blockers. The most likely new boundary is the **host-side embedding load**.

The 70B quantised embedding is compact compared with f32, but the current host still fetches that whole tensor and repacks it into CPU Q4 arrays for row lookup. During repack there is a transient period where the network bytes and compact arrays coexist. We have removed the extra Cache API clone and the duplicate GPU upload, but browser process memory may still be the first thing that fails.

If that happens, the next change is narrow and testable:

1. stream the embedding range directly into the CPU Q4 `qs` + scale arrays rather than `arrayBuffer()` then repack; or
2. implement GPU embedding-row lookup so the embedding can stream directly to GPU and avoid a full CPU copy.

Do **not** jump to global tensor sharding unless the actual evidence shows the host's individual storage-binding limit is the blocker. Full embedding/head sharding remains a useful portability path for lower-limit devices, but it is not required for the first room on a capable host.

## Success criterion

The strongest first result is:

> A room of ordinary WebGPU devices, none individually provisioned to hold the full model, loads all 80 layers of the exact Llama 3 70B Instruct Q4_0 GGUF, produces a deterministic answer, and agrees with the exact-file llama.cpp reference.

Performance can be poor and the experiment can still succeed. The primary question is whether collective browser compute can cross the model-size boundary correctly and reproducibly.
