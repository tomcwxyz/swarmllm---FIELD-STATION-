import { assert, assertEquals, assertThrows } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { chatRuntimeFromGGUF, chatRuntimeForStyle, denseConfigFromGGUF, resolveGGUFAdapter } from "../../engine/model-adapters.js";

function qwenHeader(overrides = {}) {
  return {
    meta: {
      "general.architecture": "qwen3",
      "qwen3.context_length": 40960,
      "qwen3.embedding_length": 1024,
      "qwen3.block_count": 28,
      "qwen3.feed_forward_length": 3072,
      "qwen3.attention.head_count": 16,
      "qwen3.attention.head_count_kv": 8,
      "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
      "qwen3.rope.freq_base": 1_000_000,
      ...overrides,
    },
    tensors: {
      "token_embd.weight": { shape: [151936, 1024] },
      "blk.0.attn_q.weight": { shape: [2048, 1024] },
      "blk.0.attn_k.weight": { shape: [1024, 1024] },
      "blk.0.ffn_up.weight": { shape: [3072, 1024] },
      "blk.0.ffn_gate.weight": { shape: [3072, 1024] },
    },
  };
}

function fakeTokenizer() {
  return {
    vocab: {
      "<|im_start|>": 900,
      "<|im_end|>": 901,
      "<|endoftext|>": 902,
      "<think>": 903,
      "</think>": 904,
    },
    encode(text) { return [...text].map((ch) => ch.codePointAt(0)); },
  };
}

Deno.test("Qwen3 GGUF adapter reconstructs DenseEngine config from one header", () => {
  const G = qwenHeader();
  const adapter = resolveGGUFAdapter(G);
  assertEquals(adapter?.id, "qwen3-dense");

  const cfg = denseConfigFromGGUF(G);
  assertEquals(cfg.model_type, "qwen3");
  assertEquals(cfg.hidden_size, 1024);
  assertEquals(cfg.num_hidden_layers, 28);
  assertEquals(cfg.num_attention_heads, 16);
  assertEquals(cfg.num_key_value_heads, 8);
  assertEquals(cfg.head_dim, 128);
  assertEquals(cfg.intermediate_size, 3072);
  assertEquals(cfg.vocab_size, 151936);
  assertEquals(cfg.rms_norm_eps, 1e-6);
  assertEquals(cfg.rope_theta, 1_000_000);
  assertEquals(cfg.max_position_embeddings, 40960);
});

Deno.test("Qwen3 adapter falls back to tensor shapes for optional dimensions", () => {
  const G = qwenHeader({
    "qwen3.embedding_length": undefined,
    "qwen3.feed_forward_length": undefined,
    "qwen3.attention.key_length": undefined,
    "qwen3.vocab_size": undefined,
  });
  const cfg = denseConfigFromGGUF(G);
  assertEquals(cfg.hidden_size, 1024);
  assertEquals(cfg.head_dim, 128);
  assertEquals(cfg.intermediate_size, 3072);
  assertEquals(cfg.vocab_size, 151936);
});

Deno.test("adapter owns Qwen ChatML message envelope and stop tokens", () => {
  const tok = fakeTokenizer();
  const chat = chatRuntimeForStyle("chatml-qwen", tok);
  assertEquals(chat.id, "chatml-qwen");
  assertEquals(chat.encodeMessage("user", "hi"), [900, ...tok.encode("user\nhi"), 901, 10]);
  assertEquals(chat.assistantPrefix(), [900, ...tok.encode("assistant\n"), 903, 10, 10, 904, 10, 10]);
  assert(chat.isStop(901));
  assert(chat.isStop(902));
  assert(!chat.isStop(42));
  assertEquals(chatRuntimeFromGGUF(qwenHeader(), tok).id, "chatml-qwen");
});

Deno.test("planned architectures do not silently enter the dense engine", () => {
  const G = { meta: { "general.architecture": "llama" }, tensors: {} };
  assertEquals(resolveGGUFAdapter(G)?.status, "planned");
  assertThrows(() => denseConfigFromGGUF(G), Error, "does not have a FIELD STATION dense adapter");
  assertThrows(() => chatRuntimeFromGGUF(G, fakeTokenizer()), Error, "chat style llama3-header is not implemented");
});
