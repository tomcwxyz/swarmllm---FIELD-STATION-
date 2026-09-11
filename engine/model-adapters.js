// Architecture adapters for GGUF-backed models.
//
// Keep model-family knowledge out of the room orchestration. An adapter answers:
// what architecture is this, can FIELD STATION execute it, what runtime config
// can be derived from the GGUF metadata/tensor index, and how a conversation is
// represented to the model.

import { GGML_EMBED, ggmlLayerNames } from "./gguf.js";

const GGML_ROPE_FREQS = "rope_freqs.weight";

export function normaliseGGUFArchitecture(value) {
  return String(value || "unknown").trim().toLowerCase().replaceAll("-", "").replaceAll("_", "");
}

function finitePositive(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function required(value, label) {
  const n = finitePositive(value);
  if (n === null) throw new Error(`GGUF is missing ${label}`);
  return n;
}

function tensorDim(G, name, axis) {
  const shape = G?.tensors?.[name]?.shape;
  return Array.isArray(shape) && shape.length > axis ? finitePositive(shape[axis]) : null;
}

function llamaRopeScalingType(meta) {
  const value = meta?.["llama.rope.scaling.type"];
  if (value === undefined || value === null) return null;
  const type = String(value).trim().toLowerCase();
  return type && type !== "none" ? type : null;
}

function validateLlamaRope(G) {
  const type = llamaRopeScalingType(G?.meta || {});
  if (!type) return null;
  if (type !== "llama3") throw new Error(`Llama RoPE scaling ${type} is not implemented by FIELD STATION yet`);
  const factors = G?.tensors?.[GGML_ROPE_FREQS];
  if (!factors) throw new Error("Llama3 scaled RoPE GGUF is missing rope_freqs.weight");
  if (!Array.isArray(factors.shape) || factors.shape.length !== 1)
    throw new Error("Llama3 rope_freqs.weight must be a 1D frequency-factor tensor");
  return type;
}

/**
 * Build the DenseEngine config from a dense GGUF header.
 *
 * Qwen3 and Llama 3 share the DenseEngine's RMSNorm + GQA + SwiGLU execution
 * shape. Llama GGUF conversion permutes Q/K rows for interleaved RoPE, while
 * Llama 3.1/3.2 can additionally carry exact per-frequency scaling factors.
 */
export function denseConfigFromGGUF(G) {
  const meta = G?.meta || {};
  const architecture = normaliseGGUFArchitecture(meta["general.architecture"]);
  const adapter = MODEL_ADAPTERS[architecture];
  if (!adapter || adapter.engine !== "dense" || adapter.status !== "supported") {
    throw new Error(`GGUF architecture ${architecture} does not have a FIELD STATION dense adapter`);
  }
  const ropeScaling = architecture === "llama" ? validateLlamaRope(G) : null;

  const prefix = adapter.metaPrefix || architecture;
  const names = ggmlLayerNames(0);
  const hidden = finitePositive(meta[`${prefix}.embedding_length`])
    || tensorDim(G, GGML_EMBED, 1)
    || tensorDim(G, names.q, 1);
  const heads = finitePositive(meta[`${prefix}.attention.head_count`]);
  const kvHeads = finitePositive(meta[`${prefix}.attention.head_count_kv`]) || heads;
  const qRows = tensorDim(G, names.q, 0);
  const kRows = tensorDim(G, names.k, 0);
  const headDim = finitePositive(meta[`${prefix}.attention.key_length`])
    || finitePositive(meta[`${prefix}.rope.dimension_count`])
    || (kRows && kvHeads ? kRows / kvHeads : null)
    || (qRows && heads ? qRows / heads : null)
    || (hidden && heads ? hidden / heads : null);
  const intermediate = finitePositive(meta[`${prefix}.feed_forward_length`])
    || tensorDim(G, names.up, 0)
    || tensorDim(G, names.gate, 0);
  const vocab = finitePositive(meta[`${prefix}.vocab_size`])
    || tensorDim(G, GGML_EMBED, 0);
  const rms = finitePositive(meta[`${prefix}.attention.layer_norm_rms_epsilon`]);
  const rope = finitePositive(meta[`${prefix}.rope.freq_base`]) || adapter.defaultRopeTheta;

  const cfg = {
    model_type: architecture,
    hidden_size: required(hidden, `${prefix}.embedding_length / embedding tensor width`),
    num_attention_heads: required(heads, `${prefix}.attention.head_count`),
    num_key_value_heads: required(kvHeads, `${prefix}.attention.head_count_kv`),
    head_dim: required(headDim, `${prefix}.attention.key_length / rope.dimension_count / attention tensor shape`),
    intermediate_size: required(intermediate, `${prefix}.feed_forward_length / FFN tensor shape`),
    num_hidden_layers: required(meta[`${prefix}.block_count`], `${prefix}.block_count`),
    vocab_size: required(vocab, `${prefix}.vocab_size / embedding tensor rows`),
    rms_norm_eps: required(rms, `${prefix}.attention.layer_norm_rms_epsilon`),
    rope_theta: required(rope, `${prefix}.rope.freq_base`),
    rope_interleaved: adapter.ropeLayout === "interleaved",
    rope_scaling_type: ropeScaling,
  };

  const context = finitePositive(meta[`${prefix}.context_length`]);
  if (context) cfg.max_position_embeddings = context;
  return cfg;
}

function tokenId(vocab, name, requiredToken = true) {
  const id = vocab?.[name];
  if (Number.isInteger(id)) return id;
  if (requiredToken) throw new Error(`chat special token ${name} is missing`);
  return null;
}

export function chatRuntimeForStyle(style, tok) {
  if (!tok?.vocab || typeof tok.encode !== "function") throw new Error("model tokenizer is not ready");
  const vocab = tok.vocab;

  if (style === "chatml-qwen") {
    const imStart = tokenId(vocab, "<|im_start|>");
    const imEnd = tokenId(vocab, "<|im_end|>");
    const eot = tokenId(vocab, "<|endoftext|>", false);
    const think = tokenId(vocab, "<think>", false);
    const thinkEnd = tokenId(vocab, "</think>", false);
    const stopTokens = new Set([imEnd, eot].filter(Number.isInteger));
    return Object.freeze({
      id: "chatml-qwen",
      conversationPrefix() { return []; },
      encodeMessage(role, content) { return [imStart, ...tok.encode(`${role}\n${content ?? ""}`), imEnd, ...tok.encode("\n")]; },
      assistantPrefix() {
        const ids = [imStart, ...tok.encode("assistant\n")];
        if (Number.isInteger(think) && Number.isInteger(thinkEnd)) ids.push(think, ...tok.encode("\n\n"), thinkEnd, ...tok.encode("\n\n"));
        return ids;
      },
      stopTokens,
      isStop(token) { return stopTokens.has(token); },
    });
  }

  if (style === "llama3-header") {
    const bos = tokenId(vocab, "<|begin_of_text|>");
    const headerStart = tokenId(vocab, "<|start_header_id|>");
    const headerEnd = tokenId(vocab, "<|end_header_id|>");
    const eot = tokenId(vocab, "<|eot_id|>");
    const eos = tokenId(vocab, "<|end_of_text|>", false);
    const eom = tokenId(vocab, "<|eom_id|>", false);
    const stopTokens = new Set([eot, eos, eom].filter(Number.isInteger));
    const header = (role) => [headerStart, ...tok.encode(String(role || "user")), headerEnd, ...tok.encode("\n\n")];
    return Object.freeze({
      id: "llama3-header",
      conversationPrefix() { return [bos]; },
      encodeMessage(role, content) { return [...header(role), ...tok.encode(String(content ?? "")), eot]; },
      assistantPrefix() { return header("assistant"); },
      stopTokens,
      isStop(token) { return stopTokens.has(token); },
    });
  }

  throw new Error(`FIELD STATION chat style ${style || "unknown"} is not implemented`);
}

export function chatRuntimeFromGGUF(G, tok) {
  const adapter = resolveGGUFAdapter(G);
  if (!adapter) throw new Error(`GGUF architecture ${normaliseGGUFArchitecture(G?.meta?.["general.architecture"])} has no FIELD STATION adapter`);
  return chatRuntimeForStyle(adapter.promptStyle, tok);
}

export const MODEL_ADAPTERS = Object.freeze({
  qwen3: Object.freeze({ id: "qwen3-dense", architecture: "qwen3", metaPrefix: "qwen3", engine: "dense", status: "supported", promptStyle: "chatml-qwen", ropeLayout: "half", defaultRopeTheta: 1_000_000, configFromGGUF: denseConfigFromGGUF }),
  qwen35: Object.freeze({ id: "qwen35-hybrid", architecture: "qwen35", metaPrefix: "qwen35", engine: "qwen35", status: "supported", promptStyle: "chatml-qwen" }),
  llama: Object.freeze({ id: "llama-dense", architecture: "llama", metaPrefix: "llama", engine: "dense", status: "supported", promptStyle: "llama3-header", ropeLayout: "interleaved", defaultRopeTheta: 500_000, configFromGGUF: denseConfigFromGGUF }),
  gemma: Object.freeze({ id: "gemma-dense", architecture: "gemma", engine: "dense", status: "planned", promptStyle: "model-template" }),
  gemma2: Object.freeze({ id: "gemma2-dense", architecture: "gemma2", engine: "dense", status: "planned", promptStyle: "model-template" }),
  gemma3: Object.freeze({ id: "gemma3-dense", architecture: "gemma3", engine: "dense", status: "planned", promptStyle: "model-template" }),
  gemma4: Object.freeze({ id: "gemma4", architecture: "gemma4", engine: null, status: "research", promptStyle: "model-template" }),
  phi3: Object.freeze({ id: "phi3-dense", architecture: "phi3", engine: "dense", status: "planned", promptStyle: "model-template" }),
  phi4: Object.freeze({ id: "phi4-dense", architecture: "phi4", engine: "dense", status: "planned", promptStyle: "model-template" }),
  mistral: Object.freeze({ id: "mistral-dense", architecture: "mistral", engine: "dense", status: "planned", promptStyle: "model-template" }),
});

export function resolveGGUFAdapter(G) {
  const architecture = normaliseGGUFArchitecture(G?.meta?.["general.architecture"]);
  return MODEL_ADAPTERS[architecture] || null;
}
