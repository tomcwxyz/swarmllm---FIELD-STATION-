import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

function llamaHeader(overrides = {}) {
  return {
    meta: {
      "general.architecture": "llama",
      "general.name": "Meta-Llama-3-8B-Instruct",
      "llama.context_length": 8192,
      "llama.embedding_length": 4096,
      "llama.block_count": 32,
      "llama.feed_forward_length": 14336,
      "llama.attention.head_count": 32,
      "llama.attention.head_count_kv": 8,
      "llama.attention.key_length": 128,
      "llama.attention.layer_norm_rms_epsilon": 1e-5,
      "llama.rope.freq_base": 500_000,
      "llama.vocab_size": 128256,
      ...overrides,
    },
    tensors: {
      "token_embd.weight": { shape: [128256, 4096] },
      "blk.0.attn_q.weight": { shape: [4096, 4096] },
      "blk.0.attn_k.weight": { shape: [1024, 4096] },
      "blk.0.ffn_up.weight": { shape: [14336, 4096] },
      "blk.0.ffn_gate.weight": { shape: [14336, 4096] },
    },
  };
}

function llamaTokenizer() {
  return {
    vocab: {
      "<|begin_of_text|>": 128000,
      "<|end_of_text|>": 128001,
      "<|start_header_id|>": 128006,
      "<|end_header_id|>": 128007,
      "<|eot_id|>": 128009,
    },
    encode(text) { return [...String(text)].map((ch) => ch.codePointAt(0)); },
  };
}

export async function verifyFieldBuild(dist) {
  const suffix = `?field-verify=${Date.now()}`;
  const adapters = await import(pathToFileURL(join(dist, "engine", "model-adapters.js")).href + suffix);
  const conversation = await import(pathToFileURL(join(dist, "field-station", "conversation.js")).href + suffix);
  const catalogue = await import(pathToFileURL(join(dist, "room", "models.js")).href + suffix);

  const cfg = adapters.denseConfigFromGGUF(llamaHeader());
  assert.equal(cfg.model_type, "llama");
  assert.equal(cfg.hidden_size, 4096);
  assert.equal(cfg.num_attention_heads, 32);
  assert.equal(cfg.num_key_value_heads, 8);
  assert.equal(cfg.head_dim, 128);
  assert.equal(cfg.rope_theta, 500_000);
  assert.equal(cfg.rope_interleaved, true, "Llama GGUF must select interleaved rotary pairs");

  assert.throws(
    () => adapters.denseConfigFromGGUF(llamaHeader({ "llama.rope.scaling.type": "llama3" })),
    /RoPE scaling llama3 is not implemented/,
    "scaled-RoPE Llama files must fail closed",
  );

  const tok = llamaTokenizer();
  const chat = adapters.chatRuntimeFromGGUF(llamaHeader(), tok);
  const prompt = conversation.buildConversationPrompt({
    tok,
    vocab: tok.vocab,
    chat,
    history: [{ user: "remember lighthouse", assistant: "OK" }],
    currentText: "what was it?",
    maxSeq: 1024,
  });
  assert.equal(prompt.ids[0], 128000, "Llama conversation must start with BOS");
  assert.equal(prompt.ids.filter((id) => id === 128000).length, 1, "Llama BOS must appear once per conversation");
  assert.ok(prompt.ids.includes(128009), "Llama completed messages must contain EOT");

  const model = catalogue.MODELS["llama3-8b-q4"];
  assert.ok(model, "Llama 3 experiment model must be in the built catalogue");
  assert.equal(model.kind, "gguf");
  assert.match(model.gguf, /Meta-Llama-3-8B-Instruct-Q4_0\.gguf$/);

  const denseSource = await readFile(join(dist, "engine", "dense.js"), "utf8");
  const wgslSource = await readFile(join(dist, "engine", "wgsl", "base.js"), "utf8");
  assert.match(denseSource, /cu\[11\] = cfg\.rope_interleaved \? 1 : 0/,
    "built DenseEngine must pass the adapter RoPE layout into the uniform");
  assert.match(wgslSource, /ropeInterleaved: u32/,
    "built WGSL config must contain the RoPE layout flag");
  assert.match(wgslSource, /select\(i, 2u \* i, cfg\.ropeInterleaved != 0u\)/,
    "built RoPE kernel must select adjacent Llama rotary pairs");
  assert.match(wgslSource, /select\(i \+ half, 2u \* i \+ 1u, cfg\.ropeInterleaved != 0u\)/,
    "built RoPE kernel must retain half-split Qwen pairs when the flag is off");

  console.log("FIELD STATION build verification passed: Llama 3 adapter, chat framing, guardrails and interleaved RoPE are present.");
}
