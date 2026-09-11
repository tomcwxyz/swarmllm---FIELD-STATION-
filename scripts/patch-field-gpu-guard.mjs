import { readFile, writeFile } from "node:fs/promises";

function replaceOnce(source, marker, replacement, label) {
  const first = source.indexOf(marker);
  if (first < 0) throw new Error(`FIELD STATION GPU guard: missing ${label} marker`);
  if (source.indexOf(marker, first + marker.length) >= 0) throw new Error(`FIELD STATION GPU guard: ${label} marker is not unique`);
  return source.slice(0, first) + replacement + source.slice(first + marker.length);
}

/**
 * Make WebGPU validation failures terminal for the current model instance.
 *
 * WebGPU reports the useful validation error first, then often emits a cascade
 * of "Invalid BindGroup" / "Invalid CommandBuffer" messages. Continuing to
 * sample after that produces plausible-looking garbage. FIELD STATION keeps the
 * first error, records the adapter limits that matter, and aborts on the next
 * pipeline boundary instead.
 */
export async function patchFieldGpuGuard(path) {
  let source = await readFile(path, "utf8");

  source = replaceOnce(
    source,
    "  ai.firstGpuError = null;",
    "  ai.firstGpuError = null;\n  ai.gpuFatal = null;",
    "GPU error reset",
  );

  const handler = `  ai.device.addEventListener?.("uncapturederror", (ev) => {\n    const gmsg = ev.error?.message || "";\n    if (!ai.firstGpuError) { ai.firstGpuError = gmsg; aiStatus("GPU error: " + gmsg.slice(0, 300)); log("swarm", "⚠ FIRST GPU error on " + myName + ": " + gmsg.slice(0, 600)); }\n    crumb("GPU validation error: " + gmsg.slice(0, 400));\n    if (ai.hostId && ai.role !== "host") sendTo(ai.hostId, { t: "ai-error", message: "GPU error: " + (ev.error?.message || "").slice(0, 300) });\n    log("swarm", "⚠ GPU error on " + myName + ": " + (ev.error?.message || "").slice(0, 140));\n  });`;
  if (!source.includes(handler)) throw new Error("FIELD STATION GPU guard: uncaptured-error handler changed upstream");
  source = source.replace(handler, `  ai.device.addEventListener?.("uncapturederror", (ev) => {\n    const gmsg = ev.error?.message || "WebGPU validation error";\n    if (ai.firstGpuError) return; // secondary InvalidBindGroup/CommandBuffer errors obscure the cause\n    ai.firstGpuError = gmsg;\n    ai.gpuFatal = gmsg;\n    const limits = {\n      maxBufferMB: Math.round((adapter.limits.maxBufferSize || 0) / 2 ** 20),\n      maxStorageBindingMB: Math.round((adapter.limits.maxStorageBufferBindingSize || 0) / 2 ** 20),\n      maxStorageBuffers: adapter.limits.maxStorageBuffersPerShaderStage || null,\n    };\n    aiStatus("GPU validation failed — stop and reload model: " + gmsg.slice(0, 260));\n    log("swarm", "⚠ FIRST GPU error on " + myName + ": " + gmsg.slice(0, 600));\n    window.fieldStationDiagnostics?.record("gpu:first-error", { error: gmsg.slice(0, 1200), ...limits, gpu: myMeta?.gpu || null, model: ai.model || $("ai-model")?.value || null });\n    crumb("GPU validation error: " + gmsg.slice(0, 400));\n    if (ai.hostId && ai.role !== "host") sendTo(ai.hostId, { t: "ai-error", message: "GPU error: " + gmsg.slice(0, 300) });\n  });`);

  source = replaceOnce(
    source,
    "async function aiPipeToken(id, needLogits = true) {\n  const pos = ai.pos;",
    "async function aiPipeToken(id, needLogits = true) {\n  if (ai.gpuFatal) throw new Error(`GPU validation failed: ${ai.gpuFatal}`);\n  const pos = ai.pos;",
    "token-pipeline guard",
  );

  await writeFile(path, source);
}
