// Model capability registry and GGUF preflight inspection.
//
// This intentionally describes what FIELD STATION can execute separately from
// what a GGUF file happens to contain. It is the first boundary for future
// Hugging Face "inspect before download" model selection.

import {
  GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4_0, GGML_Q4_1, GGML_Q5_K, GGML_Q6_K,
  ggmlTypeBytes,
} from "./gguf.js";
import { GGML_Q4_K, q4KTypeBytes } from "./q4k.js";

export const QUANT_CAPABILITIES = Object.freeze({
  [GGML_F32]: { name: "F32", status: "supported", path: "float" },
  [GGML_F16]: { name: "F16", status: "supported", path: "float" },
  [GGML_Q4_0]: { name: "Q4_0", status: "supported", path: "native-gpu" },
  [GGML_Q4_1]: { name: "Q4_1", status: "supported-with-conversion", path: "requant-q8" },
  [GGML_Q8_0]: { name: "Q8_0", status: "supported", path: "native-gpu" },
  [GGML_Q4_K]: { name: "Q4_K", status: "supported-with-conversion", path: "field-build-requant-q8" },
  [GGML_Q5_K]: { name: "Q5_K", status: "supported-with-conversion", path: "requant-q8" },
  [GGML_Q6_K]: { name: "Q6_K", status: "supported-with-conversion", path: "requant-q8" },
});

export const ARCHITECTURE_CAPABILITIES = Object.freeze({
  qwen3: { adapter: "dense", status: "supported" },
  qwen35: { adapter: "qwen35", status: "supported" },
  llama: { adapter: "llama", status: "planned" },
  gemma: { adapter: "gemma", status: "planned" },
  gemma2: { adapter: "gemma", status: "planned" },
  gemma3: { adapter: "gemma", status: "planned" },
  gemma4: { adapter: "gemma", status: "research" },
  phi3: { adapter: "phi", status: "planned" },
  phi4: { adapter: "phi", status: "planned" },
  mistral: { adapter: "mistral", status: "planned" },
});

export function ggmlTypeName(type) {
  return QUANT_CAPABILITIES[type]?.name || `GGML_TYPE_${type}`;
}

export function tensorTypeBytes(type, nElems) {
  if (type === GGML_Q4_K) return q4KTypeBytes(nElems);
  return ggmlTypeBytes(type, nElems);
}

function architectureFromMeta(meta = {}) {
  const raw = String(meta["general.architecture"] || "unknown").trim().toLowerCase();
  // GGUF producers are not perfectly consistent about separators.
  return raw.replaceAll("-", "").replaceAll("_", "");
}

function layerCount(meta, architecture, tensors) {
  const exact = meta[`${architecture}.block_count`];
  if (Number.isFinite(exact)) return exact;
  const keys = Object.keys(meta).filter((k) => k.endsWith(".block_count"));
  if (keys.length === 1 && Number.isFinite(meta[keys[0]])) return meta[keys[0]];
  let max = -1;
  for (const name of Object.keys(tensors || {})) {
    const match = /^blk\.(\d+)\./.exec(name);
    if (match) max = Math.max(max, Number(match[1]));
  }
  return max + 1 || null;
}

/**
 * Inspect a parsed GGUF header without fetching tensor bodies.
 * Returns an explicit architecture/quantisation verdict suitable for UI.
 */
export function inspectGGUFCompatibility(G) {
  const meta = G?.meta || {};
  const tensors = G?.tensors || {};
  const architecture = architectureFromMeta(meta);
  const arch = ARCHITECTURE_CAPABILITIES[architecture] || { adapter: null, status: "unsupported" };
  const counts = new Map();
  let totalBytes = 0;
  const unsupportedTypes = new Set();
  const conversionTypes = new Set();
  const codecReadyTypes = new Set();

  for (const tensor of Object.values(tensors)) {
    const type = tensor.ggmlType;
    counts.set(type, (counts.get(type) || 0) + 1);
    const cap = QUANT_CAPABILITIES[type];
    if (!cap) unsupportedTypes.add(type);
    else if (cap.status === "supported-with-conversion") conversionTypes.add(type);
    else if (cap.status === "codec-ready") codecReadyTypes.add(type);
    const bytes = tensorTypeBytes(type, tensor.nElems);
    if (bytes >= 0) totalBytes += bytes;
  }

  let status = "unsupported";
  if (arch.status === "supported" && unsupportedTypes.size === 0 && codecReadyTypes.size === 0) {
    status = conversionTypes.size ? "supported-with-conversion" : "supported";
  } else if (arch.status === "supported" && unsupportedTypes.size === 0 && codecReadyTypes.size > 0) {
    status = "integration-needed";
  } else if (arch.status === "planned" || arch.status === "research") {
    status = "architecture-needed";
  }

  const tensorTypes = [...counts.entries()]
    .map(([type, count]) => ({
      type,
      name: ggmlTypeName(type),
      count,
      status: QUANT_CAPABILITIES[type]?.status || "unsupported",
      path: QUANT_CAPABILITIES[type]?.path || null,
    }))
    .sort((a, b) => a.type - b.type);

  const reasons = [];
  if (arch.status !== "supported") reasons.push(`architecture ${architecture} is ${arch.status}`);
  if (unsupportedTypes.size) reasons.push(`unsupported tensor types: ${[...unsupportedTypes].map(ggmlTypeName).join(", ")}`);
  if (codecReadyTypes.size) reasons.push(`codec ready, loader integration pending: ${[...codecReadyTypes].map(ggmlTypeName).join(", ")}`);
  if (conversionTypes.size) reasons.push(`converted to Q8 at load: ${[...conversionTypes].map(ggmlTypeName).join(", ")}`);

  return {
    architecture,
    architectureStatus: arch.status,
    adapter: arch.adapter,
    status,
    layerCount: layerCount(meta, architecture, tensors),
    tensorCount: Object.keys(tensors).length,
    tensorTypes,
    totalTensorBytes: totalBytes,
    reasons,
  };
}
