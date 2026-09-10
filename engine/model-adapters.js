// Architecture adapters for GGUF-backed models.
//
// Keep model-family knowledge out of the room orchestration. An adapter answers:
// what architecture is this, can FIELD STATION execute it, what runtime config
// can be derived from the GGUF metadata/tensor index, and how a conversation is
// represented to the model.

import { GGML_EMBED, ggmlLayerNames } from "./gguf.js";

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

/**
 * Build the DenseEngine config from a dense GGUF header.
 *
 * Qwen3 is the first implemented adapter. Shape fallbacks are deliberate: some
 * converters omit optional metadata such as attention.key_length even though the
 * exact dimensions are already present in the tensor index.
 */
export function denseConfigFromGGUF(G) {
  const meta = G?.meta || {};
  const architecture = normaliseGGUFArchitecture(meta["general.architecture"]);
  const adapter = MODEL_ADAPTERS[architecture];
  if (!adapter || adapter.engine !== "dense" || adapter.status !== "supported") {
    throw new Error(`GGUF architecture ${architecture} does not have a FIELD STATION dense adapter`);
  }

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
    head_dim: required(headDim, `${prefix}.attention.key_length / attention tensor shape`),
    intermediate_size: required(intermediate, `${prefix}.feed_forward_length / FFN tensor shape`),
    num_hidden_layers: required(meta[`${prefix}.block_count`], `${prefix}.block_count`),
    vocab_size: required(vocab, `${prefix}.vocab_size / embedding tensor rows`),
    rms_norm_eps: required(rms, `${prefix}.attention.layer_norm_rms_epsilon`),
    rope_theta: required(rope, `${prefix}.rope.freq_base`),
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

/**
 * Runtime conversation formatter used by field-station/conversation.js and the
 * generation stop condition. It deliberately exposes token-id operations rather
 * than templates/Jinja so the distributed loop stays deterministic and small.
 */
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
      encodeMessage(role, content) {
        return [imStart, ...tok.encode(`${role}\n${content ?? ""}`), imEnd, ...tok.encode("\n")];
      },
      assistantPrefix() {
        const ids = [imStart, ...tok.encode("assistant\n")];
        // Qwen3 thinking models default to a visible reasoning block. FIELD STATION has
        // always pre-closed it so the room displays the answer rather than model scratchwork.
        if (Number.isInteger(think) && Number.isInteger(thinkEnd))
          ids.push(think, ...tok.encode("\n\n"), thinkEnd, ...tok.encode("\n\n"));
        return ids;
      },
      stopTokens,
      isStop(token) { return stopTokens.has(token); },
    });
  }

  throw new Error(`FIELD STATION chat style ${style || "unknown"} is not implemented`);
}

/** Resolve the GGUF architecture then construct its conversation formatter. */
export function chatRuntimeFromGGUF(G, tok) {
  const adapter = resolveGGUFAdapter(G);
  if (!adapter) throw new Error(`GGUF architecture ${normaliseGGUFArchitecture(G?.meta?.["general.architecture"])} has no FIELD STATION adapter`);
  return chatRuntimeForStyle(adapter.promptStyle, tok);
}

/**
 * Small, explicit adapter contract. Planned entries are visible so capability
 * preflight can distinguish "architecture work" from an unknown model family.
 */
export const MODEL_ADAPTERS = Object.freeze({
  qwen3: Object.freeze({
    id: "qwen3-dense",
    architecture: "qwen3",
    metaPrefix: "qwen3",
    engine: "dense",
    status: "supported",
    promptStyle: "chatml-qwen",
    defaultRopeTheta: 1_000_000,
    configFromGGUF: denseConfigFromGGUF,
  }),
  qwen35: Object.freeze({
    id: "qwen35-hybrid",
    architecture: "qwen35",
    metaPrefix: "qwen35",
    engine: "qwen35",
    status: "supported",
    promptStyle: "chatml-qwen",
  }),
  llama: Object.freeze({ id: "llama-dense", architecture: "llama", metaPrefix: "llama", engine: "dense", status: "planned", promptStyle: "llama3-header" }),
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
