import { readFile, writeFile } from "node:fs/promises";

function replaceOnce(source, marker, replacement, label) {
  const first = source.indexOf(marker);
  if (first < 0) throw new Error(`FIELD STATION model runtime patch: missing ${label} marker`);
  if (source.indexOf(marker, first + marker.length) >= 0) throw new Error(`FIELD STATION model runtime patch: ${label} marker is not unique`);
  return source.slice(0, first) + replacement + source.slice(first + marker.length);
}

/**
 * Make dense GGUF models self-describing in the FIELD STATION distributable.
 *
 * Upstream currently fetches config.json/tokenizer.json for Qwen3 dense models.
 * FIELD STATION instead derives DenseEngine config from GGUF metadata and takes
 * the tokenizer from GGUF metadata on the host. This is the runtime half of the
 * "paste one GGUF URL, inspect it, then run it" model-adapter work.
 */
export async function patchFieldModelRuntime(roomPath) {
  let source = await readFile(roomPath, "utf8");

  const importMarker = 'import { MODELS, NEED_GB, MAX_SEQ, MAX_NEW, MIN_ROOM } from "./room/models.js";';
  source = replaceOnce(
    source,
    importMarker,
    `${importMarker}\nimport { denseConfigFromGGUF, chatRuntimeFromGGUF, chatRuntimeForStyle } from "./engine/model-adapters.js";\n\n// FIELD STATION custom models are ephemeral room descriptors, not trusted config blobs.\n// Only the small set of fields needed to identify a remote GGUF crosses the control channel.\nfunction fieldModelDescriptor(modelKey) {\n  const model = MODELS[modelKey];\n  if (!model?.fieldCustom || model.kind !== "gguf") return null;\n  return {\n    label: String(model.label || "Custom GGUF").slice(0, 160),\n    kind: "gguf",\n    gguf: String(model.gguf || ""),\n    fieldCustom: true,\n  };\n}\n\nfunction installFieldModel(modelKey, descriptor, needGB) {\n  if (!modelKey || !descriptor?.fieldCustom || descriptor.kind !== "gguf") return false;\n  let url;\n  try { url = new URL(String(descriptor.gguf || "")); } catch { return false; }\n  if (url.protocol !== "https:" || !/\\.gguf(?:$|[?#])/i.test(url.href)) return false;\n  const label = String(descriptor.label || "Custom GGUF").slice(0, 160);\n  MODELS[modelKey] = { label, kind: "gguf", gguf: url.toString(), fieldCustom: true };\n  const gb = Number(needGB);\n  if (Number.isFinite(gb) && gb >= 0.2 && gb <= 128) NEED_GB[modelKey] = gb;\n  const select = $("ai-model");\n  if (select && ![...select.options].some((option) => option.value === modelKey)) {\n    const option = document.createElement("option");\n    option.value = modelKey; option.textContent = label; select.appendChild(option);\n  }\n  return true;\n}`,
    "model catalogue import",
  );

  const externalConfigMarker = `  if (M.cfg) {\n    ai.cfg = await (await fetch(M.cfg)).json();\n    if (hasEmbed || hasHead) ai.tok = makeTokenizer(await (await fetch(M.tok)).json());\n  }`;
  source = replaceOnce(
    source,
    externalConfigMarker,
    `  // Safetensors still needs its external config/tokenizer. Dense GGUF is\n  // self-describing and is initialised from its own metadata below.\n  if (M.kind === "safetensors" && M.cfg) {\n    ai.cfg = await (await fetch(M.cfg)).json();\n    if (hasEmbed || hasHead) {\n      ai.tok = makeTokenizer(await (await fetch(M.tok)).json());\n      // SmolLM's existing catalogue path uses the same ChatML envelope; keeping this\n      // explicit makes it a migration fallback rather than a generation-loop assumption.\n      ai.chat = chatRuntimeForStyle("chatml-qwen", ai.tok);\n    }\n  }`,
    "external dense config bootstrap",
  );

  // Match the semantic loader block rather than its status copy: upstream keeps the
  // status string as a literal unicode escape, which should not be part of our drift guard.
  const ggufLoadMarker = `    const G = ai.G && ai.GModel === modelKey ? ai.G : await fetchGGUFHeader(M.gguf, false);   // vocab comes from tokenizer.json\n    ai.G = G; ai.GModel = modelKey;\n    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead };`;
  source = replaceOnce(
    source,
    ggufLoadMarker,
    `    const needTok = hasEmbed || hasHead;\n    const cachedOk = ai.G && ai.GModel === modelKey && (!needTok || ai.G.meta["tokenizer.ggml.tokens"]);\n    const G = cachedOk ? ai.G : await fetchGGUFHeader(M.gguf, needTok);\n    ai.G = G; ai.GModel = modelKey;\n    ai.cfg = denseConfigFromGGUF(G);\n    if (needTok) {\n      ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));\n      ai.chat = chatRuntimeFromGGUF(G, ai.tok);\n    }\n    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead };`,
    "dense GGUF shard loader",
  );

  const hybridTokenizerMarker = `    if (hasEmbed || hasHead) ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));`;
  source = replaceOnce(
    source,
    hybridTokenizerMarker,
    `    if (hasEmbed || hasHead) {\n      ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));\n      ai.chat = chatRuntimeFromGGUF(G, ai.tok);\n    }`,
    "hybrid GGUF chat formatter",
  );

  const startConfigMarker = `    } else {\n      cfg = await (await fetch(M.cfg)).json();\n      L = cfg.num_hidden_layers;\n    }`;
  source = replaceOnce(
    source,
    startConfigMarker,
    `    } else if (M.kind === "gguf") {\n      aiStatus("reading model index\\u2026");\n      ai.G = await fetchGGUFHeader(M.gguf, false);\n      ai.GModel = modelKey;\n      cfg = denseConfigFromGGUF(ai.G);\n      L = cfg.num_hidden_layers;\n    } else {\n      cfg = await (await fetch(M.cfg)).json();\n      L = cfg.num_hidden_layers;\n    }`,
    "host dense model config",
  );

  const duplicateHeaderMarker = `      ai.G = await fetchGGUFHeader(M.gguf, false);\n      ai.GModel = modelKey;\n      layerBytes = Object.values(ggmlLayerNames(0))`;
  source = replaceOnce(
    source,
    duplicateHeaderMarker,
    `      if (!ai.G || ai.GModel !== modelKey) ai.G = await fetchGGUFHeader(M.gguf, false);\n      ai.GModel = modelKey;\n      layerBytes = Object.values(ggmlLayerNames(0))`,
    "duplicate dense GGUF header fetch",
  );

  const startRequestMarker = `  broadcastAll({ t: "ai-start-req", model, boss, by: myName });`;
  source = replaceOnce(
    source,
    startRequestMarker,
    `  broadcastAll({\n    t: "ai-start-req", model, boss, by: myName,\n    modelDef: fieldModelDescriptor(model), needGB: NEED_GB[model],\n  });`,
    "start request custom model descriptor",
  );

  const workerLoadMarker = `      const msg = {\n        t: "ai-load", model: modelKey, range: ranges[i + 1],\n        next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host",\n        host: peer.id,\n      };`;
  source = replaceOnce(
    source,
    workerLoadMarker,
    `      const msg = {\n        t: "ai-load", model: modelKey, range: ranges[i + 1],\n        modelDef: fieldModelDescriptor(modelKey), needGB: NEED_GB[modelKey],\n        next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host",\n        host: peer.id,\n      };`,
    "worker custom model descriptor",
  );

  const workerCaseMarker = `    case "ai-load": {\n      if (MODELS[d.model]) $("ai-model").value = d.model;`;
  source = replaceOnce(
    source,
    workerCaseMarker,
    `    case "ai-load": {\n      installFieldModel(d.model, d.modelDef, d.needGB);\n      if (MODELS[d.model]) $("ai-model").value = d.model;`,
    "worker installs custom model",
  );

  const startCaseMarker = `    case "ai-start-req":\n      if (MODELS[d.model]) $("ai-model").value = d.model;   // every screen shows the model that was actually started`;
  source = replaceOnce(
    source,
    startCaseMarker,
    `    case "ai-start-req":\n      installFieldModel(d.model, d.modelDef, d.needGB);\n      if (MODELS[d.model]) $("ai-model").value = d.model;   // every screen shows the model that was actually started`,
    "boss installs custom model",
  );

  await writeFile(roomPath, source);
}

if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(process.argv[1].replaceAll("\\", "/"))) {
  const target = process.argv[2];
  if (!target) throw new Error("Usage: node scripts/patch-field-model-runtime.mjs <room.js>");
  await patchFieldModelRuntime(target);
  console.log("FIELD STATION dense GGUF adapter runtime patch applied.");
}
