# Experiment · Llama 3 8B across ordinary devices

**Status:** implementation ready for field verification  
**Target:** `Meta-Llama-3-8B-Instruct-Q4_0.gguf`  
**Question:** can several ordinary browser devices collectively run a useful 8B model while preserving the same answer as a single reference runtime?

This is the first FIELD STATION experiment whose main purpose is **cross-architecture evidence**, not another Qwen benchmark. The target deliberately uses Q4_0 because FIELD STATION already streams that tensor format directly to WebGPU. That keeps the experiment focused on the Llama adapter, chat format, GGUF Q/K layout and distributed execution.

## What counts as success

Do not count plausible text as a pass. A successful run needs all three layers of evidence:

1. **Reference:** the exact GGUF produces a deterministic answer in llama.cpp.
2. **Single FIELD STATION device:** the same prompt reaches the same greedy token sequence, or a documented numerically equivalent sequence where tiny floating-point differences change a later tie/near-tie.
3. **Collective FIELD STATION room:** two- and three-device splits agree with the single-device FIELD STATION run within the same tolerance and complete without transport/GPU errors.

Until those three checks pass, the catalogue label stays `experimental`.

## Target file

Built-in catalogue key: `llama3-8b-q4`

```text
https://huggingface.co/tensorblock/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf
```

Expected characteristics:

- architecture: `llama`
- 32 transformer blocks
- hidden size: 4096
- 32 attention heads / 8 KV heads
- head dimension: 128
- FFN size: 14336
- RoPE base: 500000
- no scaled-RoPE feature
- Llama 3 role-header chat format
- Q4_0 native streaming path

If preflight reports a non-`none` `llama.rope.scaling.type`, stop: that is a different Llama runtime path and FIELD STATION should reject it.

## Reference prompts

Keep these short so the first experiment measures architecture correctness rather than long-context behaviour.

```text
P1 factual: What is the capital of Wales? Answer in one sentence.
P2 reasoning: A shop has 17 apples and sells 6. How many remain? Explain in one short sentence.
P3 format: Give exactly three colours, separated by commas, and nothing else.
P4 continuation: Turn 1: My code word is lighthouse. Reply only OK. Turn 2: What was my code word?
```

Use greedy decoding / temperature 0 for reference and FIELD STATION comparisons. Do not change the prompt wording between runtimes.

## Run matrix

| Run | Devices | Purpose |
|---|---:|---|
| R0 | llama.cpp reference | authoritative token/output baseline |
| R1 | 1 | isolate FIELD STATION Llama runtime from networking |
| R2 | 2 | prove one pipeline boundary preserves the result |
| R3 | 3 | prove multiple pipeline boundaries preserve the result |

For R2/R3, record the layer ranges assigned to each device. Repeat P1 after swapping which physical device hosts the room if practical; this catches accidental host-only assumptions.

## Record for every run

Capture:

- exact GGUF URL / revision if pinned;
- browser + OS for each FIELD STATION participant;
- GPU description exposed by WebGPU;
- pledged memory and assigned layer range;
- header/preflight verdict;
- model-load time;
- time to first token;
- decode tokens/second;
- final text and, once token tracing is wired, generated token IDs;
- any first GPU validation error;
- peer RTT / bandwidth where available;
- whether weights were cold-download or cache-hit;
- battery/heat observation for portable devices.

The evidence belongs in `docs/bench-log.md` with a link back to this protocol.

## Llama-specific checks

### Chat framing

The whole conversation gets one `<|begin_of_text|>` token. Each message uses Llama 3 start/end role headers and `<|eot_id|>` terminates completed messages. Generation stops on Llama end-of-turn/end-of-text tokens.

### Rotary layout

llama.cpp's GGUF conversion permutes Llama Q/K projection rows. FIELD STATION therefore uses adjacent/interleaved rotary pairs for this adapter rather than Qwen's half-split pairs. The GGUF weights themselves stay untouched, preserving direct Q4_0 streaming.

This is the highest-risk correctness point in the first Llama run. If R1 diverges from llama.cpp immediately, inspect Q/K + RoPE first rather than tuning sampling or chat text.

### Scaled RoPE

Original Llama 3 is the deliberate first target. Llama 3.1/3.2-style scaled-RoPE metadata currently fails closed. Do not broaden the catalogue until scaled frequencies have their own numerical reference test.

## Failure interpretation

- **Preflight fails:** metadata/tensor/host compatibility problem; do not download weights.
- **R1 differs immediately from llama.cpp:** local runtime/model-adapter bug, most likely tensor mapping, RoPE layout, tokenizer or chat framing.
- **R1 matches but R2/R3 differ:** distributed transport/sharding problem rather than Llama architecture support.
- **Outputs match but performance is poor:** correctness passes; measure placement, bandwidth, compilation and streaming separately before optimising.
- **One peer dies during load/generation:** record it as an experimental result; do not hide it by immediately retrying. Recovery is a separate roadmap item.

## Promotion gate

Promote `Llama 3 8B · Q4 · experimental` to a supported model only after:

- reference output is captured for all four prompts;
- one-device FIELD STATION agrees;
- two- and three-device runs agree;
- no unexplained WebGPU validation errors occur;
- the result and device matrix are recorded in `docs/bench-log.md`.

Only after that should we consider a better-quality Q4_K_M Llama target. That second experiment can then isolate the K-quant conversion/native-Q4_K question rather than mixing it into the architecture proof.
