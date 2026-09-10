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
    `${importMarker}\nimport { denseConfigFromGGUF } from "./engine/model-adapters.js";`,
    "model catalogue import",
  );

  const externalConfigMarker = `  if (M.cfg) {\n    ai.cfg = await (await fetch(M.cfg)).json();\n    if (hasEmbed || hasHead) ai.tok = makeTokenizer(await (await fetch(M.tok)).json());\n  }`;
  source = replaceOnce(
    source,
    externalConfigMarker,
    `  // Safetensors still needs its external config/tokenizer. Dense GGUF is\n  // self-describing and is initialised from its own metadata below.\n  if (M.kind === "safetensors" && M.cfg) {\n    ai.cfg = await (await fetch(M.cfg)).json();\n    if (hasEmbed || hasHead) ai.tok = makeTokenizer(await (await fetch(M.tok)).json());\n  }`,
    "external dense config bootstrap",
  );

  // Match the semantic loader block rather than its status copy: upstream keeps the
  // status string as a literal unicode escape, which should not be part of our drift guard.
  const ggufLoadMarker = `    const G = ai.G && ai.GModel === modelKey ? ai.G : await fetchGGUFHeader(M.gguf, false);   // vocab comes from tokenizer.json\n    ai.G = G; ai.GModel = modelKey;\n    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead };`;
  source = replaceOnce(
    source,
    ggufLoadMarker,
    `    const needTok = hasEmbed || hasHead;\n    const cachedOk = ai.G && ai.GModel === modelKey && (!needTok || ai.G.meta["tokenizer.ggml.tokens"]);\n    const G = cachedOk ? ai.G : await fetchGGUFHeader(M.gguf, needTok);\n    ai.G = G; ai.GModel = modelKey;\n    ai.cfg = denseConfigFromGGUF(G);\n    if (needTok) ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));\n    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead };`,
    "dense GGUF shard loader",
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

  await writeFile(roomPath, source);
}

if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(process.argv[1].replaceAll("\\", "/"))) {
  const target = process.argv[2];
  if (!target) throw new Error("Usage: node scripts/patch-field-model-runtime.mjs <room.js>");
  await patchFieldModelRuntime(target);
  console.log("FIELD STATION dense GGUF adapter runtime patch applied.");
}
