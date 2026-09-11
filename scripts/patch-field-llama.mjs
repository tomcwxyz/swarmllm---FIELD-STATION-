import { readFile, writeFile } from "node:fs/promises";

function replaceOnce(source, marker, replacement, label) {
  const first = source.indexOf(marker);
  if (first < 0) throw new Error(`FIELD STATION Llama patch: missing ${label} marker`);
  if (source.indexOf(marker, first + marker.length) >= 0) throw new Error(`FIELD STATION Llama patch: ${label} marker is not unique`);
  return source.slice(0, first) + replacement + source.slice(first + marker.length);
}

/**
 * Patch the copied FIELD STATION DenseEngine to understand Llama GGUF RoPE.
 *
 * Qwen's GGUF path produces the existing half-split layout: (i, i + half).
 * llama.cpp permutes Llama Q/K projection rows during GGUF conversion, so each
 * rotary pair is adjacent instead: (2i, 2i + 1).
 *
 * Llama 3.1/3.2 GGUFs additionally carry `rope_freqs.weight`: one exact
 * frequency-factor value per rotary pair. Keep these tiny factors in a fixed
 * 256-byte uniform buffer instead of a runtime-sized storage buffer. Besides
 * being cheaper, this avoids a WebGPU validation/driver edge seen on AMD GCN
 * devices when the additional storage binding is used by the RoPE pass.
 */
export async function patchFieldLlamaRuntime(densePath, wgslPath) {
  let dense = await readFile(densePath, "utf8");
  dense = replaceOnce(
    dense,
    "    cf[8] = cfg.rms_norm_eps; cf[9] = cfg.rope_theta; cu[10] = qDim;",
    "    cf[8] = cfg.rms_norm_eps; cf[9] = cfg.rope_theta; cu[10] = qDim; cu[11] = cfg.rope_interleaved ? 1 : 0;",
    "DenseEngine config uniform",
  );
  dense = replaceOnce(
    dense,
    '      rope: ["rw", "u"], attn_scores: ["ro", "ro", "rw"], attn_softmax: ["rw"],',
    '      rope: ["rw", "u", "u"], attn_scores: ["ro", "ro", "rw"], attn_softmax: ["rw"],',
    "RoPE bind-group layout",
  );
  dense = replaceOnce(
    dense,
    "    this.nKVBuf = this._buf(new Uint32Array([nKV]), GPUBufferUsage.UNIFORM);",
    "    this.nKVBuf = this._buf(new Uint32Array([nKV]), GPUBufferUsage.UNIFORM);\n    // FIELD STATION: fixed uniform storage for up to 64 rotary-pair factors\n    // (head_dim <= 128 for the currently supported Llama 3.x family). Pad\n    // unused values with 1.0 so original Llama 3 and Qwen remain identical.\n    const ropePairs = headDim / 2;\n    if (ropePairs > 64) throw new Error(`RoPE head dimension ${headDim} exceeds FIELD STATION factor capacity`);\n    const ropeFactors = new Float32Array(64).fill(1);\n    if (W.ropeFreqs?.data instanceof Float32Array) {\n      if (W.ropeFreqs.data.length !== ropePairs) throw new Error(`rope_freqs.weight has ${W.ropeFreqs.data.length} values; expected ${ropePairs}`);\n      ropeFactors.set(W.ropeFreqs.data);\n    }\n    this.ropeFreqBuf = this._buf(ropeFactors, GPUBufferUsage.UNIFORM);",
    "RoPE frequency-factor buffer",
  );
  dense = replaceOnce(
    dense,
    "    this.bgRopeQ = this._bg(this.pipes.rope, 1, [this.q, this.nHBuf]);\n    this.bgRopeK = this._bg(this.pipes.rope, 1, [this.k, this.nKVBuf]);",
    "    this.bgRopeQ = this._bg(this.pipes.rope, 1, [this.q, this.nHBuf, this.ropeFreqBuf]);\n    this.bgRopeK = this._bg(this.pipes.rope, 1, [this.k, this.nKVBuf, this.ropeFreqBuf]);",
    "single-token RoPE bind groups",
  );
  dense = replaceOnce(
    dense,
    "          ropeQ: this._bg2res(this.pipes.rope, [slice(B.q, c), { buffer: this.nHBuf }]),\n          ropeK: this._bg2res(this.pipes.rope, [slice(B.k, c), { buffer: this.nKVBuf }]),",
    "          ropeQ: this._bg2res(this.pipes.rope, [slice(B.q, c), { buffer: this.nHBuf }, { buffer: this.ropeFreqBuf }]),\n          ropeK: this._bg2res(this.pipes.rope, [slice(B.k, c), { buffer: this.nKVBuf }, { buffer: this.ropeFreqBuf }]),",
    "batched-prefill RoPE bind groups",
  );
  await writeFile(densePath, dense);

  let wgsl = await readFile(wgslPath, "utf8");
  wgsl = replaceOnce(
    wgsl,
    "  eps: f32, theta: f32, qDim: u32,\n};",
    "  eps: f32, theta: f32, qDim: u32, ropeInterleaved: u32,\n};",
    "WGSL Config layout",
  );
  wgsl = replaceOnce(
    wgsl,
    "@group(1) @binding(1) var<uniform> rp_nheads: u32;",
    "@group(1) @binding(1) var<uniform> rp_nheads: u32;\nstruct RopeFactors { values: array<vec4<f32>, 16>, };\n@group(1) @binding(2) var<uniform> rp_freq_factor: RopeFactors;",
    "RoPE frequency-factor binding",
  );
  wgsl = replaceOnce(
    wgsl,
    `  let off = h * cfg.headDim;\n  let freq = pow(cfg.theta, -f32(2u * i) / f32(cfg.headDim));\n  let ang = f32(frame.pos) * freq;\n  let c = cos(ang); let s = sin(ang);\n  let a = rp_v[off + i]; let b = rp_v[off + i + half];\n  rp_v[off + i] = a * c - b * s;\n  rp_v[off + i + half] = b * c + a * s;`,
    `  let off = h * cfg.headDim;\n  // Canonical Qwen vectors pair the two halves. Canonical Llama GGUFs have\n  // Q/K rows permuted by llama.cpp so rotary pairs are adjacent.\n  let pairA = select(i, 2u * i, cfg.ropeInterleaved != 0u);\n  let pairB = select(i + half, 2u * i + 1u, cfg.ropeInterleaved != 0u);\n  let baseFreq = pow(cfg.theta, -f32(2u * i) / f32(cfg.headDim));\n  // llama.cpp applies GGUF rope_freqs as theta_base / freq_factor. Factors\n  // are packed four-at-a-time in a fixed uniform buffer for WebGPU portability.\n  let freqFactor = max(rp_freq_factor.values[i / 4u][i % 4u], 1.0e-12);\n  let freq = baseFreq / freqFactor;\n  let ang = f32(frame.pos) * freq;\n  let c = cos(ang); let s = sin(ang);\n  let a = rp_v[off + pairA]; let b = rp_v[off + pairB];\n  rp_v[off + pairA] = a * c - b * s;\n  rp_v[off + pairB] = b * c + a * s;`,
    "RoPE pair layout + scaling",
  );
  await writeFile(wgslPath, wgsl);
}
