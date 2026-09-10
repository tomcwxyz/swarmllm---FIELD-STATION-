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

  await writeFile(ggufPath, source);
}
