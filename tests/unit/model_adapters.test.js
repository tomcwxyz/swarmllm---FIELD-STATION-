import { assert, assertEquals, assertThrows } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { chatRuntimeFromGGUF, chatRuntimeForStyle, denseConfigFromGGUF, resolveGGUFAdapter } from "../../engine/model-adapters.js";

function qwenHeader(overrides = {}) {
  return {
    meta: {
      "general.architecture": "qwen3", "qwen3.context_length": 40960,
      "qwen3.embedding_length": 1024, "qwen3.block_count": 28, "qwen3.feed_forward_length": 3072,
      "qwen3.attention.head_count": 16, "qwen3.attention.head_count_kv": 8,
      "qwen3.attention.layer_norm_rms_epsilon": 1e-6, "qwen3.rope.freq_base": 1_000_000,
      ...overrides,
    },
    tensors: {
      "token_embd.weight": { shape: [151936, 1024] }, "blk.0.attn_q.weight": { shape: [2048, 1024] },
      "blk.0.attn_k.weight": { shape: [1024, 1024] }, "blk.0.ffn_up.weight": { shape: [3072, 1024] },
      "blk.0.ffn_gate.weight": { shape: [3072, 1024] },
    },
  };
}

function llama3Header(overrides = {}, tensorOverrides = {}) {
  return {
    meta: {
      "general.architecture": "llama", "general.name": "Meta-Llama-3-8B-Instruct",
      "llama.context_length": 8192, "llama.embedding_length": 4096, "llama.block_count": 32,
      "llama.feed_forward_length": 14336, "llama.attention.head_count": 32,
      "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128,
      "llama.attention.layer_norm_rms_epsilon": 1e-5, "llama.rope.freq_base": 500_000,
      "llama.vocab_size": 128256, ...overrides,
    },
    tensors: {
      "token_embd.weight": { shape: [128256, 4096] }, "blk.0.attn_q.weight": { shape: [4096, 4096] },
      "blk.0.attn_k.weight": { shape: [1024, 4096] }, "blk.0.ffn_up.weight": { shape: [14336, 4096] },
      "blk.0.ffn_gate.weight": { shape: [14336, 4096] }, ...tensorOverrides,
    },
  };
}

function fakeTokenizer(vocab = {}) {
  return { vocab: { "<|im_start|>": 900, "<|im_end|>": 901, "<|endoftext|>": 902, "<think>": 903, "</think>": 904, ...vocab }, encode(text) { return [...text].map((ch) => ch.codePointAt(0)); } };
}
function llamaTokenizer() {
  return fakeTokenizer({ "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001, "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009 });
}

Deno.test("Qwen3 GGUF adapter reconstructs DenseEngine config from one header", () => {
  const cfg = denseConfigFromGGUF(qwenHeader());
  assertEquals(cfg.model_type, "qwen3"); assertEquals(cfg.hidden_size, 1024); assertEquals(cfg.num_hidden_layers, 28);
  assertEquals(cfg.num_attention_heads, 16); assertEquals(cfg.num_key_value_heads, 8); assertEquals(cfg.head_dim, 128);
  assertEquals(cfg.intermediate_size, 3072); assertEquals(cfg.vocab_size, 151936); assertEquals(cfg.rms_norm_eps, 1e-6);
  assertEquals(cfg.rope_theta, 1_000_000); assertEquals(cfg.rope_interleaved, false); assertEquals(cfg.max_position_embeddings, 40960);
});

Deno.test("adapter owns Qwen ChatML message envelope and stop tokens", () => {
  const tok = fakeTokenizer(); const chat = chatRuntimeForStyle("chatml-qwen", tok);
  assertEquals(chat.encodeMessage("user", "hi"), [900, ...tok.encode("user\nhi"), 901, 10]);
  assert(chat.isStop(901)); assert(chat.isStop(902)); assertEquals(chatRuntimeFromGGUF(qwenHeader(), tok).id, "chatml-qwen");
});

Deno.test("Llama 3 GGUF adapter reconstructs DenseEngine config", () => {
  const G = llama3Header(); const adapter = resolveGGUFAdapter(G); assertEquals(adapter?.id, "llama-dense");
  const cfg = denseConfigFromGGUF(G);
  assertEquals(cfg.hidden_size, 4096); assertEquals(cfg.num_hidden_layers, 32); assertEquals(cfg.head_dim, 128);
  assertEquals(cfg.rope_theta, 500_000); assertEquals(cfg.rope_interleaved, true); assertEquals(cfg.rope_scaling_type, null);
});

Deno.test("Llama 3 chat adapter emits one-conversation BOS, headers and terminators", () => {
  const tok = llamaTokenizer(); const chat = chatRuntimeFromGGUF(llama3Header(), tok);
  assertEquals(chat.conversationPrefix(), [128000]);
  assertEquals(chat.encodeMessage("user", "hi"), [128006, ...tok.encode("user"), 128007, 10, 10, ...tok.encode("hi"), 128009]);
  assert(chat.isStop(128009)); assert(chat.isStop(128001));
});

Deno.test("Llama 3.1/3.2 scaled RoPE accepts exact GGUF frequency factors", () => {
  const G = llama3Header({ "llama.context_length": 131072, "llama.rope.scaling.type": "llama3", "llama.rope.scaling.factor": 32 }, {
    "rope_freqs.weight": { shape: [64], ggmlType: 0, nElems: 64, byteLength: 256 },
  });
  const cfg = denseConfigFromGGUF(G);
  assertEquals(cfg.rope_scaling_type, "llama3"); assertEquals(cfg.max_position_embeddings, 131072); assertEquals(cfg.rope_interleaved, true);
});

Deno.test("scaled Llama fails closed when exact factor tensor is missing", () => {
  assertThrows(() => denseConfigFromGGUF(llama3Header({ "llama.rope.scaling.type": "llama3" })), Error, "missing rope_freqs.weight");
});

Deno.test("unknown Llama RoPE schemes still fail closed", () => {
  assertThrows(() => denseConfigFromGGUF(llama3Header({ "llama.rope.scaling.type": "yarn" })), Error, "RoPE scaling yarn is not implemented");
});

Deno.test("planned architectures do not silently enter the dense engine", () => {
  const G = { meta: { "general.architecture": "gemma3" }, tensors: {} };
  assertEquals(resolveGGUFAdapter(G)?.status, "planned");
  assertThrows(() => denseConfigFromGGUF(G), Error, "does not have a FIELD STATION dense adapter");
});
