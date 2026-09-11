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

  const cfg = adapters.denseConfigFromGGUF(llamaHeader());
  assert.equal(cfg.model_type, "llama");
  assert.equal(cfg.hidden_size, 4096);
  assert.equal(cfg.head_dim, 128);
  assert.equal(cfg.rope_theta, 500_000);
  assert.equal(cfg.rope_interleaved, true, "Llama GGUF must select interleaved rotary pairs");
  assert.equal(cfg.rope_scaling_type, null);

  const llama70Header = llamaHeader({
    "general.name": "Meta-Llama-3-70B-Instruct",
    "llama.embedding_length": 8192,
    "llama.block_count": 80,
    "llama.feed_forward_length": 28672,
    "llama.attention.head_count": 64,
    "llama.attention.head_count_kv": 8,
  }, {
    "token_embd.weight": { shape: [128256, 8192] },
    "blk.0.attn_q.weight": { shape: [8192, 8192] },
    "blk.0.attn_k.weight": { shape: [1024, 8192] },
    "blk.0.ffn_up.weight": { shape: [28672, 8192] },
    "blk.0.ffn_gate.weight": { shape: [28672, 8192] },
    "output.weight": { shape: [128256, 8192] },
  });
  const cfg70 = adapters.denseConfigFromGGUF(llama70Header);
  assert.equal(cfg70.model_type, "llama");
  assert.equal(cfg70.hidden_size, 8192);
  assert.equal(cfg70.num_hidden_layers, 80);
  assert.equal(cfg70.intermediate_size, 28672);
  assert.equal(cfg70.num_attention_heads, 64);
  assert.equal(cfg70.num_key_value_heads, 8);
  assert.equal(cfg70.head_dim, 128);
  assert.equal(cfg70.rope_theta, 500_000);
  assert.equal(cfg70.rope_interleaved, true);
  assert.equal(cfg70.rope_scaling_type, null,
    "original Llama 3 70B must stay on the proven unscaled-RoPE path");

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

  const chunks = sources.chunkSource({ name: "brief.md", text: "The office opens in June.\n\nThe lighthouse opens in May.\n\nThe garden opens in July." }, { chunkChars: 42, overlapChars: 0 });
  const retrieved = sources.buildSourceContext(chunks, "When does the lighthouse open?", { maxChars: 180, maxChunks: 2 });
  assert.match(retrieved.text, /lighthouse opens in May/i);
  assert.deepEqual(retrieved.labels, ["brief.md"]);
  assert.ok(retrieved.chunks >= 1 && retrieved.chunks <= 2);

  // CSV Sources are datasets, not prose documents. A vague request such as
  // "analyse the source" must still receive whole-dataset structure/statistics,
  // while a specific term can additionally retrieve matching rows.
  const csvText = [
    "organisation,amount,region,date,note",
    'Alpha,10,North,2026-01-01,"first, quoted note"',
    "Beta,30,South,2026-02-01,second",
    "Alpha,20,North,2026-03-01,third",
  ].join("\n");
  const dataset = sources.profileCSV({ name: "sample.csv", text: csvText });
  assert.equal(dataset.rowCount, 3);
  assert.equal(dataset.columnCount, 5);
  assert.equal(dataset.rows[0][4], "first, quoted note", "CSV parser must preserve quoted commas");
  const amount = dataset.columns.find((c) => c.name === "amount");
  assert.equal(amount.kind, "numeric");
  assert.equal(amount.mean, 20);
  const region = dataset.columns.find((c) => c.name === "region");
  assert.equal(region.kind, "categorical");
  assert.deepEqual(region.top[0], ["North", 2]);
  const csvChunks = sources.chunkCSVRows(dataset, { rowsPerChunk: 2 });
  const genericDatasetContext = sources.buildSourceContext(csvChunks, "Analyse the source I gave you", {
    datasets: [dataset], maxChars: 2400, maxChunks: 2,
  });
  assert.match(genericDatasetContext.text, /you HAVE received/i);
  assert.match(genericDatasetContext.text, /3 data rows · 5 columns/);
  assert.match(genericDatasetContext.text, /amount: numeric; 0 missing; min 10, max 30, mean 20/);
  assert.match(genericDatasetContext.text, /region: categorical; 0 missing; 2 distinct; common North \(2\)/);
  assert.doesNotMatch(genericDatasetContext.text, /Row 1:/,
    "generic dataset analysis should prefer whole-dataset profile over arbitrary first rows");
  const targetedDatasetContext = sources.buildSourceContext(csvChunks, "What does the South row say?", {
    datasets: [dataset], maxChars: 3000, maxChunks: 2,
  });
  assert.match(targetedDatasetContext.text, /matching rows/);
  assert.match(targetedDatasetContext.text, /region=South/);

  const llama1 = catalogue.MODELS["llama32-1b-q4"];
  const llama3 = catalogue.MODELS["llama32-3b-q4"];
  const llama8 = catalogue.MODELS["llama3-8b-q4"];
  const llama70 = catalogue.MODELS["llama3-70b-q4"];
  assert.match(llama1?.gguf || "", /^https:\/\/huggingface\.co\/QuantFactory\/Llama-3\.2-1B-Instruct-GGUF\/resolve\/main\/Llama-3\.2-1B-Instruct\.Q4_0\.gguf$/);
  assert.match(llama3?.gguf || "", /^https:\/\/huggingface\.co\/QuantFactory\/Llama-3\.2-3B-Instruct-GGUF\/resolve\/main\/Llama-3\.2-3B-Instruct\.Q4_0\.gguf$/);
  assert.match(llama8?.gguf || "", /^https:\/\/huggingface\.co\/QuantFactory\/Meta-Llama-3-8B-Instruct-GGUF\/resolve\/main\/Meta-Llama-3-8B-Instruct\.Q4_0\.gguf$/);
  assert.match(llama70?.gguf || "", /^https:\/\/huggingface\.co\/QuantFactory\/Meta-Llama-3-70B-Instruct-GGUF\/resolve\/main\/Meta-Llama-3-70B-Instruct\.Q4_0\.gguf$/);
  assert.ok((catalogue.NEED_GB["llama3-70b-q4"] || 0) >= 41,
    "70B room requirement must include weight and KV-cache headroom");
  assert.ok((llama70?.hostMinBindGB || 0) >= 0.5,
    "70B must select a host capable of binding its large output head");

  const denseSource = await readFile(join(dist, "engine", "dense.js"), "utf8");
  const wgslSource = await readFile(join(dist, "engine", "wgsl", "base.js"), "utf8");
  const ggufSource = await readFile(join(dist, "engine", "gguf.js"), "utf8");
  const roomSource = await readFile(join(dist, "room.js"), "utf8");
  const htmlSource = await readFile(join(dist, "field-room.html"), "utf8");
  const sourceUi = await readFile(join(dist, "field-station-sources.js"), "utf8");

  assert.match(denseSource, /cu\[11\] = cfg\.rope_interleaved \? 1 : 0/);
  assert.match(denseSource, /this\.ropeFreqBuf = this\._buf\(ropeFactors, GPUBufferUsage\.UNIFORM\)/,
    "DenseEngine must keep tiny RoPE factors in a portable uniform buffer");
  assert.match(denseSource, /new Float32Array\(64\)\.fill\(1\)/,
    "RoPE factor uniform must be fixed-size and identity padded");
  assert.match(denseSource, /if \(hasHead && !W\.head\) this\.headEntry = up\(W\.embed\)/,
    "untied 70B must not upload a duplicate GPU copy of token embeddings");
  const ropeBindLines = denseSource.split("\n").filter((line) =>
    line.includes("this.pipes.rope") && (line.includes("this._bg(") || line.includes("this._bg2res(")));
  assert.equal(ropeBindLines.length, 4,
    `expected exactly four DenseEngine RoPE bind-group constructions, found ${ropeBindLines.length}`);
  for (const line of ropeBindLines) assert.match(line, /ropeFreqBuf/,
    `every single-token and batched-prefill RoPE bind group must include frequency factors: ${line.trim()}`);
  assert.match(wgslSource, /struct RopeFactors \{ values: array<vec4<f32>, 16>, \}/,
    "WGSL must expose a fixed 256-byte uniform for frequency factors");
  assert.match(wgslSource, /var<uniform> rp_freq_factor: RopeFactors/,
    "RoPE factors must not use a runtime-sized storage binding");
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
    "only selected source material must enter prompt assembly");
  assert.match(roomSource, /ai\.gpuFatal = gmsg/,
    "first WebGPU validation error must poison the current model instance");
  assert.match(roomSource, /if \(ai\.gpuFatal\) throw new Error\(`GPU validation failed:/,
    "generation must stop after a GPU validation failure rather than sample garbage");
  assert.match(roomSource, /maxStorageBindingMB/,
    "GPU diagnostics must record relevant adapter limits");
  assert.match(roomSource, /meta\.maxBindGB = .*maxStorageBufferBindingSize/,
    "room roster must advertise storage-binding capacity for large-head host selection");
  assert.match(roomSource, /hi - lo \+ 1 <= 192 \* 2 \*\* 20/,
    "giant 70B embedding\/head ranges must not be cloned into the browser weight cache");
  assert.match(roomSource, /name === GGML_EMBED && G\.tensors\[GGML_OUTPUT\]/,
    "untied Llama embeddings must remain CPU-side instead of being duplicated on GPU");
  assert.match(roomSource, /layerBytes \+= 2 \* MAX_SEQ \* kvDim \* 4/,
    "dense room planner must budget distributed KV-cache memory per layer");
  assert.match(roomSource, /hostMinBindGB/,
    "70B host selection must enforce the output-head storage-binding requirement");

  assert.match(sourceUi, /profileCSV/,
    "CSV uploads must be profiled locally instead of treated as prose");
  assert.match(sourceUi, /rowCount.*columns/,
    "CSV Sources UI must surface dataset dimensions");
  assert.match(htmlSource, /id="sources-panel"/);
  assert.match(htmlSource, /field-station-sources\.js/);
  assert.match(htmlSource, /llama32-1b-q4/);
  assert.match(htmlSource, /llama32-3b-q4/);
  await access(join(dist, "field-station-sources.js"));
  await access(join(dist, "field-station-sources.css"));

  console.log("FIELD STATION build verification passed: Llama 3/3.2 adapters, 70B collective planning, all RoPE bind paths, portable scaled-RoPE uniforms, GPU fail-fast guard, dataset-aware CSV Sources and local document Sources are present.");
}
