import { readFile, writeFile } from "node:fs/promises";

export async function patchFieldRuntime(path) {
  let source = await readFile(path, "utf8");

  const importMarker = 'import { makeLink, attachWire, wireReady, sendFrame } from "./room/transport.js";';
  if (!source.includes(importMarker)) throw new Error("FIELD STATION runtime patch failed: transport import marker changed upstream");
  source = source.replace(importMarker, `${importMarker}\nimport { buildConversationPrompt, generationFinishReason } from "./field-station/conversation.js";`);

  const streamMarker = '  const streamOpts = { pace: isPhone ? 300 : 0, staging: isPhone ? 2 * 2 ** 20 : 8 * 2 ** 20 };';
  if (!source.includes(streamMarker)) throw new Error("FIELD STATION runtime patch failed: phone stream pacing marker changed upstream");
  source = source.replace(streamMarker, `  // FIELD STATION: keep a small allocator yield only for iPhone/WebKit.\n  const streamOpts = { pace: myMeta?.ua === "iPhone" ? 75 : 0, staging: isPhone ? 2 * 2 ** 20 : 8 * 2 ** 20 };`);

  const pacerStart = '  // every device (host included, even when its weights come from cache) keeps within a few\n';
  const pacerEnd = '  pacerHook = pacer;\n';
  const start = source.indexOf(pacerStart), end = source.indexOf(pacerEnd, start);
  if (start < 0 || end < 0) throw new Error("FIELD STATION runtime patch failed: room pacing block changed upstream");
  source = source.slice(0, start) + `  // FIELD STATION: readiness already gates on assigned shards; do not throttle fast peers.\n  pacerHook = null;\n` + source.slice(end + pacerEnd.length);

  const fetchMarker = '  const r = await fetch(url, { headers: { Range: `bytes=${lo}-${hi}` } });';
  if (!source.includes(fetchMarker)) throw new Error("FIELD STATION runtime patch failed: range-fetch marker changed upstream");
  source = source.replace(fetchMarker, `  ai.loadNetworkRequests = (ai.loadNetworkRequests || 0) + 1;\n  const r = await fetch(url, { headers: { Range: \`bytes=\${lo}-\${hi}\` } });`);

  const rangeErrorMarker = '  if (r.status !== 206) throw new Error("model host refused range requests");';
  if (!source.includes(rangeErrorMarker)) throw new Error("FIELD STATION runtime patch failed: range error marker changed upstream");
  source = source.replace(rangeErrorMarker, '  if (r.status !== 206) throw new Error(`model host refused range requests (HTTP ${r.status}, ${new URL(url).host})`);');

  const progressMarker = `function aiProgress(done, total, note) {\n  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;\n  $("ldg-fill").style.width = pct + "%";\n  $("ldg-sub").textContent = \`${'${(done / 2 ** 20).toFixed(0)}'} MB of ${'${(total / 2 ** 20).toFixed(0)}'} MB · ${'${pct}'}%\` + (note ? " · " + note : "");\n}`;
  if (!source.includes(progressMarker)) throw new Error("FIELD STATION runtime patch failed: download progress marker changed upstream");
  source = source.replace(progressMarker, `function aiProgress(done, total, note) {\n  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;\n  $("ldg-fill").style.width = pct + "%";\n  const elapsed = ai.loadStartedAt ? (performance.now() - ai.loadStartedAt) / 1000 : 0;\n  const cachedThisLoad = Math.max(0, cacheHits - (ai.loadCacheStart || 0));\n  const networkBytes = Math.max(0, done - cachedThisLoad);\n  const rate = elapsed > 0.75 && networkBytes > 0 ? (networkBytes / 2 ** 20 / elapsed).toFixed(1) + " MB/s" : "";\n  const req = ai.loadNetworkRequests ? ai.loadNetworkRequests + " req" : "";\n  const cached = cachedThisLoad > 2 ** 20 ? (cachedThisLoad / 2 ** 20).toFixed(0) + " MB cached" : "";\n  $("ldg-sub").textContent = \`${'${(done / 2 ** 20).toFixed(0)}'} MB of ${'${(total / 2 ** 20).toFixed(0)}'} MB · ${'${pct}'}%\`\n    + (rate ? " · " + rate : "") + (req ? " · " + req : "") + (cached ? " · " + cached : "") + (note ? " · " + note : "");\n}`);

  const timingMarker = `  ai.myPct = 0;\n  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };`;
  if (!source.includes(timingMarker)) throw new Error("FIELD STATION runtime patch failed: load timing marker changed upstream");
  source = source.replace(timingMarker, `  ai.myPct = 0;\n  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };\n  ai.loadStartedAt = performance.now();\n  ai.loadCacheStart = cacheHits;\n  ai.loadNetworkRequests = 0;`);

  const aiStateMarker = `  waiters: new Map(),    // pos -> resolve(hiddenF32) for host awaiting return\n  busy: false,\n};`;
  if (!source.includes(aiStateMarker)) throw new Error("FIELD STATION runtime patch failed: AI state marker changed upstream");
  source = source.replace(aiStateMarker, `  waiters: new Map(),\n  busy: false,\n  history: [],\n  contextMeta: { usedTurns: 0, droppedTurns: 0, promptTokens: 0 },\n  chat: null,\n};\n\nfunction fieldSourceBundle(raw) {\n  if (!raw || typeof raw !== "object") return { text: "", labels: [], chunks: 0 };\n  const text = String(raw.text || "").slice(0, 3200);\n  const labels = Array.isArray(raw.labels) ? raw.labels.slice(0, 6).map((v) => String(v).slice(0, 120)) : [];\n  const chunks = Math.max(0, Math.min(6, Number(raw.chunks) || 0));\n  return { text, labels, chunks };\n}`);

  const crumbMarker = 'function crumb(s) { try { localStorage.setItem("swarm-crumb", JSON.stringify({ s, t: Date.now(), mem: performance.memory?.usedJSHeapSize })); } catch {} }';
  if (!source.includes(crumbMarker)) throw new Error("FIELD STATION runtime patch failed: breadcrumb marker changed upstream");
  source = source.replace(crumbMarker, `function crumb(s) {\n  const mem = performance.memory?.usedJSHeapSize;\n  try { localStorage.setItem("swarm-crumb", JSON.stringify({ s, t: Date.now(), mem })); } catch {}\n  window.fieldStationDiagnostics?.record("runtime", { status: s, heapBytes: mem || null });\n}`);

  const generateMarker = 'async function aiGenerate(textArg, who, askerId = peer.id) {\n  const text = (textArg ?? $("ai-prompt").value).trim();\n  const asker = who || myName;';
  if (!source.includes(generateMarker)) throw new Error("FIELD STATION runtime patch failed: aiGenerate signature marker changed upstream");
  source = source.replace(generateMarker, 'async function aiGenerate(textArg, who, askerId = peer.id, sourceArg = null) {\n  const text = (textArg ?? $("ai-prompt").value).trim();\n  const asker = who || myName;\n  const localSource = askerId === peer.id ? window.fieldStationSources?.contextFor?.(text) : null;\n  const sources = fieldSourceBundle(sourceArg || localSource);');

  const promptMarker = `  const V = ai.tok.vocab;\n  const imStart = V["<|im_start|>"], imEnd = V["<|im_end|>"], eot = V["<|endoftext|>"];\n  const ids = [imStart, ...ai.tok.encode("user\\n" + text), imEnd,\n    ...ai.tok.encode("\\n"), imStart, ...ai.tok.encode("assistant\\n")];\n  // qwen3 thinking models: pre-close the think block so answers come straight\n  if (V["<think>"] !== undefined && V["</think>"] !== undefined)\n    ids.push(V["<think>"], ...ai.tok.encode("\\n\\n"), V["</think>"], ...ai.tok.encode("\\n\\n"));`;
  if (!source.includes(promptMarker)) throw new Error("FIELD STATION runtime patch failed: generation prompt marker changed upstream");
  source = source.replace(promptMarker, `  const V = ai.tok.vocab;\n  const legacyStops = new Set([V["<|im_end|>"], V["<|endoftext|>"]].filter(Number.isInteger));\n  const isStop = (token) => ai.chat?.isStop ? ai.chat.isStop(token) : legacyStops.has(token);\n  const built = buildConversationPrompt({\n    tok: ai.tok, vocab: V, chat: ai.chat, history: ai.history || [], currentText: text, sourceContext: sources.text,\n    maxSeq: MAX_SEQ, minRoom: MIN_ROOM,\n  });\n  const ids = built.ids;\n  ai.contextMeta = { usedTurns: built.usedTurns, droppedTurns: built.droppedTurns, promptTokens: ids.length, sourceTokens: built.sourceTokens || 0 };\n  const contextLine = $("context-line");\n  if (contextLine) contextLine.textContent = \`CONTEXT / \${built.usedTurns} prior turn\${built.usedTurns === 1 ? "" : "s"} · \${ids.length}/\${MAX_SEQ} prompt tok\${sources.labels.length ? \` · \${sources.labels.length} source\${sources.labels.length === 1 ? "" : "s"}\` : ""}\${built.droppedTurns ? \` · \${built.droppedTurns} older dropped\` : ""}\`;\n  broadcastAll({ t: "ai-context", ...ai.contextMeta, sourceCount: sources.labels.length, maxSeq: MAX_SEQ });\n  window.fieldStationDiagnostics?.record("generation:start", {\n    promptTokens: ids.length, usedTurns: built.usedTurns, droppedTurns: built.droppedTurns, sourceTokens: built.sourceTokens || 0, sourceChunks: sources.chunks,\n    devices: ai.chain.length + 1, model: $("ai-model")?.value || null, chatFormat: ai.chat?.id || "legacy-chatml",\n  });`);

  for (const [marker, replacement] of [
    ['if (next === imEnd || next === eot) done = true; else emit(next);', 'if (isStop(next)) done = true; else emit(next);'],
    ['if (tk === imEnd || tk === eot) { done = true; break; }', 'if (isStop(tk)) { done = true; break; }'],
    ['if (next === imEnd || next === eot) { await aiPipeToken(next, false); break; }', 'if (isStop(next)) { await aiPipeToken(next, false); break; }'],
  ]) {
    if (!source.includes(marker)) throw new Error("FIELD STATION runtime patch failed: generation stop marker changed upstream");
    source = source.replace(marker, replacement);
  }

  const statsMarker = `    const secs = (performance.now() - t0) / 1000;\n    const stats = \`${'${count}'} tok · ${'${(count / secs).toFixed(1)}'} tok/s · ${'${ai.chain.length + 1}'} devices${'${capped ? ` · stopped: context full (${MAX_SEQ} tokens)` : ""}'}\`;\n    chatBotEnd(reply, stats);\n    sendChat({ t: "ai-gendone", stats }, askerId);`;
  if (!source.includes(statsMarker)) throw new Error("FIELD STATION runtime patch failed: generation stats marker changed upstream");
  source = source.replace(statsMarker, `    const secs = (performance.now() - t0) / 1000;\n    const finish = generationFinishReason({ count, maxNew, maxSeq: MAX_SEQ, promptTokens: ids.length, contextCapped: capped });\n    ai.history = [...(ai.history || []), { user: text, assistant: reply }].slice(-24);\n    const stats = \`${'${count}'} tok · ${'${(count / secs).toFixed(1)}'} tok/s · ${'${ai.chain.length + 1}'} devices · finish: ${'${finish}'}\`;\n    chatBotEnd(reply, stats);\n    sendChat({ t: "ai-gendone", stats }, askerId);\n    const contextLine = $("context-line");\n    if (contextLine) contextLine.textContent = \`CONTEXT / \${ai.history.length} room turn\${ai.history.length === 1 ? "" : "s"} · last prompt \${ids.length}/\${MAX_SEQ} tok\${sources.labels.length ? \` · \${sources.labels.length} source\${sources.labels.length === 1 ? "" : "s"}\` : ""}\`;\n    broadcastAll({ t: "ai-context", turns: ai.history.length, promptTokens: ids.length, usedTurns: built.usedTurns, droppedTurns: built.droppedTurns, sourceCount: sources.labels.length, maxSeq: MAX_SEQ });\n    window.fieldStationDiagnostics?.record("generation:end", { outputTokens: count, tokensPerSecond: Number((count / secs).toFixed(2)), finish, promptTokens: ids.length, historyTurns: ai.history.length, sourceChunks: sources.chunks, prefillSeconds: Number(((t0 - tPre) / 1000).toFixed(2)) });`);

  const visibilityMarker = `    case "ai-visibility":\n      ai.visibility = d.mode;\n      toast(d.mode === "all" ? "the host shows the chat to everyone" : d.mode === "host" ? "the host keeps the chat private" : "the host shows each answer to whoever asked");\n      break;`;
  if (!source.includes(visibilityMarker)) throw new Error("FIELD STATION runtime patch failed: visibility case marker changed upstream");
  source = source.replace(visibilityMarker, `${visibilityMarker}\n    case "ai-context": {\n      const contextLine = $("context-line");\n      if (contextLine) {\n        const turns = d.turns ?? d.usedTurns ?? 0;\n        contextLine.textContent = \`CONTEXT / \${turns} room turn\${turns === 1 ? "" : "s"} · \${d.promptTokens || 0}/\${d.maxSeq || MAX_SEQ} prompt tok\${d.sourceCount ? \` · \${d.sourceCount} source\${d.sourceCount === 1 ? "" : "s"}\` : ""}\${d.droppedTurns ? \` · \${d.droppedTurns} older dropped\` : ""}\`;\n      }\n      break;\n    }`);

  const askMarker = '      aiGenerate(d.text, d.name, from);';
  if (!source.includes(askMarker)) throw new Error("FIELD STATION runtime patch failed: remote ask marker changed upstream");
  source = source.replace(askMarker, '      aiGenerate(d.text, d.name, from, fieldSourceBundle(d.sources));');

  const submitMarker = `function aiSubmit() {\n  const text = $("ai-prompt").value.trim();\n  if (!text) return;\n  if (ai.role === "host") { aiGenerate(); return; }\n  const hostId = ai.hostId;\n  if (!conns.has(hostId)) { toast("not connected to the host"); return; }\n  $("ai-prompt").value = "";\n  sendTo(hostId, { t: "ai-ask", text, name: myName });\n}`;
  if (!source.includes(submitMarker)) throw new Error("FIELD STATION runtime patch failed: submit marker changed upstream");
  source = source.replace(submitMarker, `function aiSubmit() {\n  const text = $("ai-prompt").value.trim();\n  if (!text) return;\n  const sources = fieldSourceBundle(window.fieldStationSources?.contextFor?.(text));\n  if (ai.role === "host") { aiGenerate(text, myName, peer.id, sources); return; }\n  const hostId = ai.hostId;\n  if (!conns.has(hostId)) { toast("not connected to the host"); return; }\n  $("ai-prompt").value = "";\n  sendTo(hostId, { t: "ai-ask", text, name: myName, sources });\n}`);

  await writeFile(path, source);
}

if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(process.argv[1].replaceAll("\\", "/"))) {
  const target = process.argv[2];
  if (!target) throw new Error("Usage: node scripts/patch-field-runtime.mjs <room.js>");
  await patchFieldRuntime(target);
  console.log("FIELD STATION runtime patches applied: downloads, conversation context, local Sources, adapter stops, finish reasons and diagnostics.");
}
