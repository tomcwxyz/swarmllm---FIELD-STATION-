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
 * frequency-factor value per rotary pair. The loader makes that tiny tensor
 * available on every shard; the shader divides the base frequency by it, matching
 * llama.cpp's `theta_base / freq_factor` path without reimplementing converter constants.
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
    '      rope: ["rw", "u", "ro"], attn_scores: ["ro", "ro", "rw"], attn_softmax: ["rw"],',
    "RoPE bind-group layout",
  );
  dense = replaceOnce(
    dense,
    "    this.nKVBuf = this._buf(new Uint32Array([nKV]), GPUBufferUsage.UNIFORM);",
    "    this.nKVBuf = this._buf(new Uint32Array([nKV]), GPUBufferUsage.UNIFORM);\n    // Default factors are all 1.0. Scaled Llama GGUFs provide an exact\n    // rope_freqs.weight vector through W.ropeFreqs.\n    const ropePairs = headDim / 2;\n    const ropeFactors = W.ropeFreqs?.data instanceof Float32Array && W.ropeFreqs.data.length === ropePairs\n      ? W.ropeFreqs.data\n      : new Float32Array(ropePairs).fill(1);\n    this.ropeFreqBuf = this._buf(ropeFactors, GPUBufferUsage.STORAGE);",
    "RoPE frequency-factor buffer",
  );
  dense = replaceOnce(
    dense,
    "    this.bgRopeQ = this._bg(this.pipes.rope, 1, [this.q, this.nHBuf]);\n    this.bgRopeK = this._bg(this.pipes.rope, 1, [this.k, this.nKVBuf]);",
    "    this.bgRopeQ = this._bg(this.pipes.rope, 1, [this.q, this.nHBuf, this.ropeFreqBuf]);\n    this.bgRopeK = this._bg(this.pipes.rope, 1, [this.k, this.nKVBuf, this.ropeFreqBuf]);",
    "RoPE bind groups",
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
    "@group(1) @binding(1) var<uniform> rp_nheads: u32;\n@group(1) @binding(2) var<storage, read> rp_freq_factor: array<f32>;",
    "RoPE frequency-factor binding",
  );
  wgsl = replaceOnce(
    wgsl,
    `  let off = h * cfg.headDim;\n  let freq = pow(cfg.theta, -f32(2u * i) / f32(cfg.headDim));\n  let ang = f32(frame.pos) * freq;\n  let c = cos(ang); let s = sin(ang);\n  let a = rp_v[off + i]; let b = rp_v[off + i + half];\n  rp_v[off + i] = a * c - b * s;\n  rp_v[off + i + half] = b * c + a * s;`,
    `  let off = h * cfg.headDim;\n  // Canonical Qwen vectors pair the two halves. Canonical Llama GGUFs have\n  // Q/K rows permuted by llama.cpp so rotary pairs are adjacent.\n  let pairA = select(i, 2u * i, cfg.ropeInterleaved != 0u);\n  let pairB = select(i + half, 2u * i + 1u, cfg.ropeInterleaved != 0u);\n  let baseFreq = pow(cfg.theta, -f32(2u * i) / f32(cfg.headDim));\n  // llama.cpp applies GGUF rope_freqs as theta_base / freq_factor. A vector\n  // of ones preserves original Llama 3 and Qwen behaviour exactly.\n  let freqFactor = max(rp_freq_factor[i], 1.0e-12);\n  let freq = baseFreq / freqFactor;\n  let ang = f32(frame.pos) * freq;\n  let c = cos(ang); let s = sin(ang);\n  let a = rp_v[off + pairA]; let b = rp_v[off + pairB];\n  rp_v[off + pairA] = a * c - b * s;\n  rp_v[off + pairB] = b * c + a * s;`,
    "RoPE pair layout + scaling",
  );
  await writeFile(wgslPath, wgsl);
}
