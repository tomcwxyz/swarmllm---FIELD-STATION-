import { readFile, writeFile } from "node:fs/promises";

function replaceOnce(source, marker, replacement, label) {
  const first = source.indexOf(marker);
  if (first < 0) throw new Error(`FIELD STATION model patch: missing ${label} marker`);
  if (source.indexOf(marker, first + marker.length) >= 0) throw new Error(`FIELD STATION model patch: ${label} marker is not unique`);
  return source.slice(0, first) + replacement + source.slice(first + marker.length);
}

export async function patchFieldModels(ggufPath) {
  let source = await readFile(ggufPath, "utf8");

  source = `import { GGML_Q4_K, q4KTypeBytes, dequantQ4K } from "./q4k.js";\n\n${source}`;

  source = replaceOnce(
    source,
    "    case GGML_Q8_0: return (n / 32) * 34;\n    case GGML_Q5_K: return (n / QK_K) * 176;",
    "    case GGML_Q8_0: return (n / 32) * 34;\n    case GGML_Q4_K: return q4KTypeBytes(n);\n    case GGML_Q5_K: return (n / QK_K) * 176;",
    "GGML byte-size",
  );

  source = replaceOnce(
    source,
    "  if (T === GGML_Q5_K) {",
    "  if (T === GGML_Q4_K) return dequantQ4K(bytes, n);\n  if (T === GGML_Q5_K) {",
    "Q4_K dequant",
  );

  // Llama 3.1/3.2 GGUFs produced by llama.cpp carry the exact per-frequency
  // scaling factors as a tiny global tensor. Every device needs it because every
  // shard applies RoPE to its own Q/K activations.
  source = replaceOnce(
    source,
    'export const GGML_OUTPUT = "output.weight"; // absent when embeddings are tied',
    'export const GGML_OUTPUT = "output.weight"; // absent when embeddings are tied\nexport const GGML_ROPE_FREQS = "rope_freqs.weight";',
    "global RoPE tensor name",
  );

  source = replaceOnce(
    source,
    "  const out = { layers };\n  if (hasEmbed || hasHead) out.embed = await entry(GGML_EMBED);",
    "  const out = { layers };\n  // Optional exact frequency factors for scaled-RoPE Llama GGUFs. This is tiny\n  // (head_dim / 2 floats) and is intentionally fetched by every shard.\n  out.ropeFreqs = await entry(GGML_ROPE_FREQS, true);\n  if (hasEmbed || hasHead) out.embed = await entry(GGML_EMBED);",
    "dense GGUF optional RoPE factors",
  );

  source = replaceOnce(
    source,
    "  for (let i = lo; i < hi; i++) Object.values(ggmlLayerNames(i)).forEach(add);\n  if (hasEmbed || hasHead) add(GGML_EMBED);",
    "  for (let i = lo; i < hi; i++) Object.values(ggmlLayerNames(i)).forEach(add);\n  add(GGML_ROPE_FREQS);\n  if (hasEmbed || hasHead) add(GGML_EMBED);",
    "dense GGUF RoPE bytes",
  );

  await writeFile(ggufPath, source);
}
