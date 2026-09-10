import { readFile, writeFile } from "node:fs/promises";

function replaceOnce(source, marker, replacement, label) {
  const first = source.indexOf(marker);
  if (first < 0) throw new Error(`FIELD STATION Llama patch: missing ${label} marker`);
  if (source.indexOf(marker, first + marker.length) >= 0) throw new Error(`FIELD STATION Llama patch: ${label} marker is not unique`);
  return source.slice(0, first) + replacement + source.slice(first + marker.length);
}

/**
 * Patch the copied FIELD STATION DenseEngine to understand two RoPE layouts.
 *
 * Qwen's GGUF path produces the existing half-split layout: (i, i + half).
 * llama.cpp permutes Llama Q/K projection rows during GGUF conversion, so each
 * rotary pair is adjacent instead: (2i, 2i + 1). Keeping the GGUF weights in
 * their canonical layout preserves streaming and avoids a large load-time copy.
 */
export async function patchFieldLlamaRuntime(densePath, wgslPath) {
  let dense = await readFile(densePath, "utf8");
  dense = replaceOnce(
    dense,
    "    cf[8] = cfg.rms_norm_eps; cf[9] = cfg.rope_theta; cu[10] = qDim;",
    "    cf[8] = cfg.rms_norm_eps; cf[9] = cfg.rope_theta; cu[10] = qDim; cu[11] = cfg.rope_interleaved ? 1 : 0;",
    "DenseEngine config uniform",
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
    `  let off = h * cfg.headDim;\n  let freq = pow(cfg.theta, -f32(2u * i) / f32(cfg.headDim));\n  let ang = f32(frame.pos) * freq;\n  let c = cos(ang); let s = sin(ang);\n  let a = rp_v[off + i]; let b = rp_v[off + i + half];\n  rp_v[off + i] = a * c - b * s;\n  rp_v[off + i + half] = b * c + a * s;`,
    `  let off = h * cfg.headDim;\n  // Canonical Qwen vectors pair the two halves. Canonical Llama GGUFs have\n  // Q/K rows permuted by llama.cpp so rotary pairs are adjacent.\n  let pairA = select(i, 2u * i, cfg.ropeInterleaved != 0u);\n  let pairB = select(i + half, 2u * i + 1u, cfg.ropeInterleaved != 0u);\n  let freq = pow(cfg.theta, -f32(2u * i) / f32(cfg.headDim));\n  let ang = f32(frame.pos) * freq;\n  let c = cos(ang); let s = sin(ang);\n  let a = rp_v[off + pairA]; let b = rp_v[off + pairB];\n  rp_v[off + pairA] = a * c - b * s;\n  rp_v[off + pairB] = b * c + a * s;`,
    "RoPE pair layout",
  );
  await writeFile(wgslPath, wgsl);
}
