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

  const progressMarker = `function aiProgress(done, total, note) {\n  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;\n  $("ldg-fill").style.width = pct + "%";\n  $("ldg-sub").textContent = \`${'${(done / 2 ** 20).toFixed(0)}'} MB of ${'${(total / 2 ** 20).toFixed(0)}'} MB · ${'${pct}'}%\` + (note ? " · " + note : "");\n}`;
  if (!source.includes(progressMarker)) {
    throw new Error("FIELD STATION runtime patch failed: download progress marker changed upstream");
  }
  source = source.replace(
    progressMarker,
    `function aiProgress(done, total, note) {\n  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;\n  $("ldg-fill").style.width = pct + "%";\n  const elapsed = ai.loadStartedAt ? (performance.now() - ai.loadStartedAt) / 1000 : 0;\n  const networkBytes = Math.max(0, done - (ai.loadCacheStart || 0));\n  const rate = elapsed > 0.75 && networkBytes > 0 ? (networkBytes / 2 ** 20 / elapsed).toFixed(1) + " MB/s" : "";\n  $("ldg-sub").textContent = \`${'${(done / 2 ** 20).toFixed(0)}'} MB of ${'${(total / 2 ** 20).toFixed(0)}'} MB · ${'${pct}'}%\`\n    + (rate ? " · " + rate : "") + (note ? " · " + note : "");\n}`,
  );

  const timingMarker = `  ai.myPct = 0;\n  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };`;
  if (!source.includes(timingMarker)) {
    throw new Error("FIELD STATION runtime patch failed: load timing marker changed upstream");
  }
  source = source.replace(
    timingMarker,
    `  ai.myPct = 0;\n  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };\n  ai.loadStartedAt = performance.now();\n  ai.loadCacheStart = cacheHits;`,
  );

  await writeFile(path, source);
}

if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(process.argv[1].replaceAll("\\", "/"))) {
  const target = process.argv[2];
  if (!target) throw new Error("Usage: node scripts/patch-field-runtime.mjs <room.js>");
  await patchFieldRuntime(target);
  console.log("FIELD STATION runtime patches applied: fast independent downloads + load telemetry.");
}
