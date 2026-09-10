import { readFile, writeFile } from "node:fs/promises";

/**
 * Apply deliberately small FIELD STATION runtime patches to the copied upstream room.js.
 *
 * We patch dist/ rather than the upstream source so future SwarmLLM updates remain easy
 * to merge. Every patch asserts its source marker so an upstream change fails the build
 * instead of silently producing an unreviewed runtime.
 */
export async function patchFieldRuntime(path) {
  let source = await readFile(path, "utf8");

  const importMarker = 'import { makeLink, attachWire, wireReady, sendFrame } from "./room/transport.js";';
  if (!source.includes(importMarker)) {
    throw new Error("FIELD STATION runtime patch failed: transport import marker changed upstream");
  }
  source = source.replace(
    importMarker,
    `${importMarker}\nimport { buildConversationPrompt, generationFinishReason } from "./field-station/conversation.js";`,
  );

  const streamMarker = '  const streamOpts = { pace: isPhone ? 300 : 0, staging: isPhone ? 2 * 2 ** 20 : 8 * 2 ** 20 };';
  if (!source.includes(streamMarker)) {
    throw new Error("FIELD STATION runtime patch failed: phone stream pacing marker changed upstream");
  }
  source = source.replace(
    streamMarker,
    `  // FIELD STATION: the old 300 ms pause ran after every streamed tensor on both\n  // iPhone and Android. Keep a small allocator yield only for iPhone/WebKit; Android\n  // and desktop should use the network/GPU as fast as they safely can.\n  const streamOpts = { pace: myMeta?.ua === "iPhone" ? 75 : 0, staging: isPhone ? 2 * 2 ** 20 : 8 * 2 ** 20 };`,
  );

  const pacerStart = '  // every device (host included, even when its weights come from cache) keeps within a few\n';
  const pacerEnd = '  pacerHook = pacer;\n';
  const start = source.indexOf(pacerStart);
  const end = source.indexOf(pacerEnd, start);
  if (start < 0 || end < 0) {
    throw new Error("FIELD STATION runtime patch failed: room pacing block changed upstream");
  }
  source = source.slice(0, start)
    + `  // FIELD STATION: readiness is already gated by every assigned shard completing.\n  // Do not slow a fast device merely to make progress bars move in step with the slowest.\n  pacerHook = null;\n`
    + source.slice(end + pacerEnd.length);

  const fetchMarker = '  const r = await fetch(url, { headers: { Range: `bytes=${lo}-${hi}` } });';
  if (!source.includes(fetchMarker)) {
    throw new Error("FIELD STATION runtime patch failed: range-fetch marker changed upstream");
  }
  source = source.replace(
    fetchMarker,
    `  ai.loadNetworkRequests = (ai.loadNetworkRequests || 0) + 1;\n  const r = await fetch(url, { headers: { Range: \`bytes=\${lo}-\${hi}\` } });`,
  );

  const progressMarker = `function aiProgress(done, total, note) {\n  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;\n  $("ldg-fill").style.width = pct + "%";\n  $("ldg-sub").textContent = \`${'${(done / 2 ** 20).toFixed(0)}'} MB of ${'${(total / 2 ** 20).toFixed(0)}'} MB · ${'${pct}'}%\` + (note ? " · " + note : "");\n}`;
  if (!source.includes(progressMarker)) {
    throw new Error("FIELD STATION runtime patch failed: download progress marker changed upstream");
  }
  source = source.replace(
    progressMarker,
    `function aiProgress(done, total, note) {\n  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;\n  $("ldg-fill").style.width = pct + "%";\n  const elapsed = ai.loadStartedAt ? (performance.now() - ai.loadStartedAt) / 1000 : 0;\n  const cachedThisLoad = Math.max(0, cacheHits - (ai.loadCacheStart || 0));\n  const networkBytes = Math.max(0, done - cachedThisLoad);\n  const rate = elapsed > 0.75 && networkBytes > 0 ? (networkBytes / 2 ** 20 / elapsed).toFixed(1) + " MB/s" : "";\n  const req = ai.loadNetworkRequests ? ai.loadNetworkRequests + " req" : "";\n  const cached = cachedThisLoad > 2 ** 20 ? (cachedThisLoad / 2 ** 20).toFixed(0) + " MB cached" : "";\n  $("ldg-sub").textContent = \`${'${(done / 2 ** 20).toFixed(0)}'} MB of ${'${(total / 2 ** 20).toFixed(0)}'} MB · ${'${pct}'}%\`\n    + (rate ? " · " + rate : "") + (req ? " · " + req : "") + (cached ? " · " + cached : "")\n    + (note ? " · " + note : "");\n}`,
  );

  const timingMarker = `  ai.myPct = 0;\n  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };`;
  if (!source.includes(timingMarker)) {
    throw new Error("FIELD STATION runtime patch failed: load timing marker changed upstream");
  }
  source = source.replace(
    timingMarker,
    `  ai.myPct = 0;\n  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };\n  ai.loadStartedAt = performance.now();\n  ai.loadCacheStart = cacheHits;\n  ai.loadNetworkRequests = 0;`,
  );

  const aiStateMarker = `  waiters: new Map(),    // pos -> resolve(hiddenF32) for host awaiting return\n  busy: false,\n};`;
  if (!source.includes(aiStateMarker)) {
    throw new Error("FIELD STATION runtime patch failed: AI state marker changed upstream");
  }
  source = source.replace(
    aiStateMarker,
    `  waiters: new Map(),    // pos -> resolve(hiddenF32) for host awaiting return\n  busy: false,\n  // FIELD STATION: canonical room conversation lives in host memory only. It is rebuilt\n  // into the model prompt each turn and disappears when the host closes the room.\n  history: [],\n  contextMeta: { usedTurns: 0, droppedTurns: 0, promptTokens: 0 },\n  chat: null,\n};`,
  );

  const crumbMarker = 'function crumb(s) { try { localStorage.setItem("swarm-crumb", JSON.stringify({ s, t: Date.now(), mem: performance.memory?.usedJSHeapSize })); } catch {} }';
  if (!source.includes(crumbMarker)) {
    throw new Error("FIELD STATION runtime patch failed: breadcrumb marker changed upstream");
  }
  source = source.replace(
    crumbMarker,
    `function crumb(s) {\n  const mem = performance.memory?.usedJSHeapSize;\n  try { localStorage.setItem("swarm-crumb", JSON.stringify({ s, t: Date.now(), mem })); } catch {}\n  window.fieldStationDiagnostics?.record("runtime", { status: s, heapBytes: mem || null });\n}`,
  );

  const promptMarker = `  const V = ai.tok.vocab;\n  const imStart = V["<|im_start|>"], imEnd = V["<|im_end|>"], eot = V["<|endoftext|>"];\n  const ids = [imStart, ...ai.tok.encode("user\\n" + text), imEnd,\n    ...ai.tok.encode("\\n"), imStart, ...ai.tok.encode("assistant\\n")];\n  // qwen3 thinking models: pre-close the think block so answers come straight\n  if (V["<think>"] !== undefined && V["</think>"] !== undefined)\n    ids.push(V["<think>"], ...ai.tok.encode("\\n\\n"), V["</think>"], ...ai.tok.encode("\\n\\n"));`;
  if (!source.includes(promptMarker)) {
    throw new Error("FIELD STATION runtime patch failed: generation prompt marker changed upstream");
  }
  source = source.replace(
    promptMarker,
    `  const V = ai.tok.vocab;\n  // The active architecture adapter owns message representation and stopping. Keep a\n  // ChatML fallback only for older/safetensors catalogue entries while they migrate.\n  const legacyStops = new Set([V["<|im_end|>"], V["<|endoftext|>"]].filter(Number.isInteger));\n  const isStop = (token) => ai.chat?.isStop ? ai.chat.isStop(token) : legacyStops.has(token);\n  const built = buildConversationPrompt({\n    tok: ai.tok, vocab: V, chat: ai.chat, history: ai.history || [], currentText: text,\n    maxSeq: MAX_SEQ, minRoom: MIN_ROOM,\n  });\n  const ids = built.ids;\n  ai.contextMeta = { usedTurns: built.usedTurns, droppedTurns: built.droppedTurns, promptTokens: ids.length };\n  const contextLine = $("context-line");\n  if (contextLine) contextLine.textContent = \`CONTEXT / \${built.usedTurns} prior turn\${built.usedTurns === 1 ? "" : "s"} · \${ids.length}/\${MAX_SEQ} prompt tok\${built.droppedTurns ? \` · \${built.droppedTurns} older dropped\` : ""}\`;\n  broadcastAll({ t: "ai-context", ...ai.contextMeta, maxSeq: MAX_SEQ });\n  window.fieldStationDiagnostics?.record("generation:start", {\n    promptTokens: ids.length, usedTurns: built.usedTurns, droppedTurns: built.droppedTurns,\n    devices: ai.chain.length + 1, model: $("ai-model")?.value || null, chatFormat: ai.chat?.id || "legacy-chatml",\n  });`,
  );

  const stopMarkers = [
    ['if (next === imEnd || next === eot) done = true; else emit(next);', 'if (isStop(next)) done = true; else emit(next);'],
    ['if (tk === imEnd || tk === eot) { done = true; break; }', 'if (isStop(tk)) { done = true; break; }'],
    ['if (next === imEnd || next === eot) { await aiPipeToken(next, false); break; }', 'if (isStop(next)) { await aiPipeToken(next, false); break; }'],
  ];
  for (const [marker, replacement] of stopMarkers) {
    if (!source.includes(marker)) throw new Error("FIELD STATION runtime patch failed: generation stop marker changed upstream");
    source = source.replace(marker, replacement);
  }

  const statsMarker = `    const secs = (performance.now() - t0) / 1000;\n    const stats = \`${'${count}'} tok · ${'${(count / secs).toFixed(1)}'} tok/s · ${'${ai.chain.length + 1}'} devices${'${capped ? ` · stopped: context full (${MAX_SEQ} tokens)` : ""}'}\`;\n    chatBotEnd(reply, stats);\n    sendChat({ t: "ai-gendone", stats }, askerId);`;
  if (!source.includes(statsMarker)) {
    throw new Error("FIELD STATION runtime patch failed: generation stats marker changed upstream");
  }
  source = source.replace(
    statsMarker,
    `    const secs = (performance.now() - t0) / 1000;\n    const finish = generationFinishReason({ count, maxNew, maxSeq: MAX_SEQ, promptTokens: ids.length, contextCapped: capped });\n    ai.history = [...(ai.history || []), { user: text, assistant: reply }].slice(-24);\n    const stats = \`${'${count}'} tok · ${'${(count / secs).toFixed(1)}'} tok/s · ${'${ai.chain.length + 1}'} devices · finish: ${'${finish}'}\`;\n    chatBotEnd(reply, stats);\n    sendChat({ t: "ai-gendone", stats }, askerId);\n    const contextLine = $("context-line");\n    if (contextLine) contextLine.textContent = \`CONTEXT / \${ai.history.length} room turn\${ai.history.length === 1 ? "" : "s"} · last prompt \${ids.length}/\${MAX_SEQ} tok\`;\n    broadcastAll({ t: "ai-context", turns: ai.history.length, promptTokens: ids.length, usedTurns: built.usedTurns, droppedTurns: built.droppedTurns, maxSeq: MAX_SEQ });\n    window.fieldStationDiagnostics?.record("generation:end", {\n      outputTokens: count, tokensPerSecond: Number((count / secs).toFixed(2)), finish,\n      promptTokens: ids.length, historyTurns: ai.history.length, prefillSeconds: Number(((t0 - tPre) / 1000).toFixed(2)),\n    });`,
  );

  const contextCaseMarker = `    case "ai-visibility":\n      ai.visibility = d.mode;\n      toast(d.mode === "all" ? "the host shows the chat to everyone" : d.mode === "host" ? "the host keeps the chat private" : "the host shows each answer to whoever asked");\n      break;`;
  if (!source.includes(contextCaseMarker)) {
    throw new Error("FIELD STATION runtime patch failed: visibility case marker changed upstream");
  }
  source = source.replace(
    contextCaseMarker,
    `${contextCaseMarker}\n    case "ai-context": {\n      const contextLine = $("context-line");\n      if (contextLine) {\n        const turns = d.turns ?? d.usedTurns ?? 0;\n        contextLine.textContent = \`CONTEXT / \${turns} room turn\${turns === 1 ? "" : "s"} · \${d.promptTokens || 0}/\${d.maxSeq || MAX_SEQ} prompt tok\${d.droppedTurns ? \` · \${d.droppedTurns} older dropped\` : ""}\`;\n      }\n      break;\n    }`,
  );

  await writeFile(path, source);
}

if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(process.argv[1].replaceAll("\\", "/"))) {
  const target = process.argv[2];
  if (!target) throw new Error("Usage: node scripts/patch-field-runtime.mjs <room.js>");
  await patchFieldRuntime(target);
  console.log("FIELD STATION runtime patches applied: downloads, conversation context, adapter stops, finish reasons and diagnostics.");
}
