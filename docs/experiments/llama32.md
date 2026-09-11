# Llama 3.2 · scaled-RoPE field experiment

**Status:** experimental · not verified

This experiment extends the working original-Llama-3 path to Llama 3.2 1B/3B Instruct Q4_0. The architecture remains dense RMSNorm/GQA/SwiGLU, but Llama 3.2 uses `llama3` RoPE scaling. FIELD STATION consumes the exact `rope_freqs.weight` factors stored in the llama.cpp-produced GGUF rather than independently reproducing the converter constants.

## Targets

- `Llama 3.2 1B Instruct · Q4_0` — QuantFactory GGUF
- `Llama 3.2 3B Instruct · Q4_0` — QuantFactory GGUF

The targets remain experimental until deterministic llama.cpp comparison and one/two/three-device equivalence are recorded.

## Field run 01 · 3B / AMD GCN4 / solo

**Date:** 2026-09-11

Observed on one browser device reporting `amd gcn-4`, with 6 GB pledged to the room. The 3B Q4 model loaded and generation started at roughly 4.3 tok/s, but the generated text was severely corrupted. The room emitted repeated WebGPU follow-on errors including `Invalid BindGroup` and `Invalid CommandBuffer` while generation continued.

### Interpretation

This is a failed run, not weak model quality. Once a WebGPU validation error invalidates a resource or command buffer, subsequent sampled tokens cannot be treated as model output. The repeated bind-group/command-buffer messages are secondary failures; the first validation error is the useful diagnostic.

The first scaled-RoPE implementation introduced a third RoPE bind-group resource as a small runtime-sized read-only storage buffer. That binding is the main new GPU resource relative to the already-working original-Llama-3 path and is therefore the first portability suspect on this device class.

### Changes after run 01

1. Move the 64 RoPE frequency factors to a fixed 256-byte uniform buffer (`array<vec4<f32>, 16>`) rather than a runtime-sized storage buffer.
2. Identity-pad unused entries with `1.0`, preserving original Llama 3 and Qwen behaviour.
3. Validate the factor count against `head_dim / 2` before building pipelines.
4. Treat the first uncaptured WebGPU validation error as fatal for the current model instance. Stop generation at the next pipeline boundary rather than continuing to sample corrupt output.
5. Record the first error plus `maxBufferSize`, `maxStorageBufferBindingSize`, storage-buffer-stage limit, GPU label and model in FIELD STATION diagnostics; suppress secondary error spam.

## Retest gate

Run 02 should repeat the same solo 3B prompt on the same AMD GCN4 device after the uniform-buffer change. A pass requires:

- no WebGPU validation error;
- coherent deterministic output;
- then comparison with the exact GGUF in llama.cpp.

If the new run still fails, Diagnostics should now preserve the first validation error and relevant device limits. That error becomes the next debugging target; do not infer model correctness from generation continuing.

After 3B is clean on the original device, run the same scaled-RoPE path with 1B as a cheaper reference target, then proceed to split-device equivalence.
