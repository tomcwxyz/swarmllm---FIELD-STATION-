import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

function llamaHeader(overrides = {}, tensorOverrides = {}) {
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
      ...tensorOverrides,
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
  const sources = await import(pathToFileURL(join(dist, "field-station", "sources.js")).href + suffix);
  const catalogue = await import(pathToFileURL(join(dist, "room", "models.js")).href + suffix);

  // Original Llama 3 remains unchanged: interleaved rotary pairs, no scaling.
  const cfg = adapters.denseConfigFromGGUF(llamaHeader());
  assert.equal(cfg.model_type, "llama");
  assert.equal(cfg.hidden_size, 4096);
  assert.equal(cfg.head_dim, 128);
  assert.equal(cfg.rope_theta, 500_000);
  assert.equal(cfg.rope_interleaved, true, "Llama GGUF must select interleaved rotary pairs");
  assert.equal(cfg.rope_scaling_type, null);

  // Llama 3.1/3.2 may declare llama3 scaling only when the GGUF carries the
  // exact llama.cpp-generated frequency factor tensor.
  const scaled = llamaHeader(
    { "llama.context_length": 131072, "llama.rope.scaling.type": "llama3", "llama.rope.scaling.factor": 32 },
    { "rope_freqs.weight": { shape: [64], ggmlType: 0, nElems: 64, byteLength: 256 } },
  );
  const scaledCfg = adapters.denseConfigFromGGUF(scaled);
  assert.equal(scaledCfg.rope_scaling_type, "llama3");
  assert.equal(scaledCfg.max_position_embeddings, 131072);
  assert.throws(
    () => adapters.denseConfigFromGGUF(llamaHeader({ "llama.rope.scaling.type": "llama3" })),
    /missing rope_freqs\.weight/,
    "scaled Llama must fail closed without exact GGUF frequency factors",
  );
  assert.throws(
    () => adapters.denseConfigFromGGUF(llamaHeader({ "llama.rope.scaling.type": "yarn" })),
    /RoPE scaling yarn is not implemented/,
  );

  const tok = llamaTokenizer();
  const chat = adapters.chatRuntimeFromGGUF(llamaHeader(), tok);
  const sourceContext = "LOCAL SOURCE EXCERPTS\n[Source: brief.md · excerpt 1]\nThe lighthouse opens in May.";
  const prompt = conversation.buildConversationPrompt({
    tok,
    vocab: tok.vocab,
    chat,
    history: [{ user: "remember lighthouse", assistant: "OK" }],
    currentText: "when does it open?",
    sourceContext,
    maxSeq: 1024,
  });
  assert.equal(prompt.ids[0], 128000, "Llama conversation must start with BOS");
  assert.equal(prompt.ids.filter((id) => id === 128000).length, 1, "Llama BOS must appear once even with Sources");
  assert.ok(prompt.sourceTokens > 0, "source excerpts must be represented in the current prompt");

  // Sources retrieval is local and bounded before room.js ever sees the excerpt bundle.
  const chunks = sources.chunkSource({ name: "brief.md", text: "The office opens in June.\n\nThe lighthouse opens in May.\n\nThe garden opens in July." }, { chunkChars: 42, overlapChars: 0 });
  const retrieved = sources.buildSourceContext(chunks, "When does the lighthouse open?", { maxChars: 180, maxChunks: 2 });
  assert.match(retrieved.text, /lighthouse opens in May/i);
  assert.deepEqual(retrieved.labels, ["brief.md"]);
  assert.ok(retrieved.chunks >= 1 && retrieved.chunks <= 2);

  const llama1 = catalogue.MODELS["llama32-1b-q4"];
  const llama3 = catalogue.MODELS["llama32-3b-q4"];
  const llama8 = catalogue.MODELS["llama3-8b-q4"];
  assert.match(llama1?.gguf || "", /^https:\/\/huggingface\.co\/QuantFactory\/Llama-3\.2-1B-Instruct-GGUF\/resolve\/main\/Llama-3\.2-1B-Instruct\.Q4_0\.gguf$/);
  assert.match(llama3?.gguf || "", /^https:\/\/huggingface\.co\/QuantFactory\/Llama-3\.2-3B-Instruct-GGUF\/resolve\/main\/Llama-3\.2-3B-Instruct\.Q4_0\.gguf$/);
  assert.match(llama8?.gguf || "", /^https:\/\/huggingface\.co\/QuantFactory\/Meta-Llama-3-8B-Instruct-GGUF\/resolve\/main\/Meta-Llama-3-8B-Instruct\.Q4_0\.gguf$/);

  const denseSource = await readFile(join(dist, "engine", "dense.js"), "utf8");
  const wgslSource = await readFile(join(dist, "engine", "wgsl", "base.js"), "utf8");
  const ggufSource = await readFile(join(dist, "engine", "gguf.js"), "utf8");
  const roomSource = await readFile(join(dist, "room.js"), "utf8");
  const htmlSource = await readFile(join(dist, "field-room.html"), "utf8");

  assert.match(denseSource, /cu\[11\] = cfg\.rope_interleaved \? 1 : 0/);
  assert.match(denseSource, /ropeFreqBuf/,
    "DenseEngine must upload exact or identity RoPE factors");
  assert.match(denseSource, /\[this\.q, this\.nHBuf, this\.ropeFreqBuf\]/,
    "RoPE Q bind group must include frequency factors");
  assert.match(wgslSource, /rp_freq_factor: array<f32>/,
    "WGSL must bind GGUF frequency factors");
  assert.match(wgslSource, /let freq = baseFreq \/ freqFactor/,
    "WGSL must apply exact llama.cpp scaling factors");
  assert.match(wgslSource, /select\(i, 2u \* i, cfg\.ropeInterleaved != 0u\)/,
    "Llama must retain adjacent rotary pairs");
  assert.match(ggufSource, /GGML_ROPE_FREQS = "rope_freqs\.weight"/,
    "built GGUF loader must name the global frequency tensor");
  assert.match(ggufSource, /out\.ropeFreqs = await entry\(GGML_ROPE_FREQS, true\)/,
    "built GGUF loader must load optional frequency factors on every shard");

  assert.match(roomSource, /fieldStationSources\?\.contextFor/,
    "room runtime must retrieve Sources locally at question time");
  assert.match(roomSource, /sourceContext: sources\.text/,
    "only selected source excerpts must enter prompt assembly");
  assert.match(htmlSource, /id="sources-panel"/);
  assert.match(htmlSource, /field-station-sources\.js/);
  assert.match(htmlSource, /llama32-1b-q4/);
  assert.match(htmlSource, /llama32-3b-q4/);
  await access(join(dist, "field-station-sources.js"));
  await access(join(dist, "field-station-sources.css"));

  console.log("FIELD STATION build verification passed: Llama 3/3.2 adapters, exact scaled-RoPE factors, QuantFactory sources and local document Sources are present.");
}
