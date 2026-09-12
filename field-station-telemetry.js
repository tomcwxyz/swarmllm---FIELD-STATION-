import { DenseEngine } from "./engine/dense.js";
import { Qwen35Engine } from "./engine/qwen35.js";

const TELEMETRY_EVENT = "field-station-telemetry";
const PATCHED = Symbol.for("field-station.telemetry.patched");
let engineRunStartedAt = null;
let engineFirstTokenSeen = false;

function emit(type, data = {}) {
  window.dispatchEvent(new CustomEvent(TELEMETRY_EVENT, {
    detail: { type, atPerf: performance.now(), ...data },
  }));
}

function layerInfo(engine) {
  const range = engine?.layerRange || engine?.range;
  return Array.isArray(range) && range.length >= 2
    ? { layerStart: range[0], layerEnd: range[1], layers: Math.max(0, range[1] - range[0]) }
    : {};
}

function tokenCountFor(method, args) {
  if (method === "runHiddenBatch") {
    const xs = args[0];
    const dim = this?.dims?.dim || this?.dim;
    return dim && xs?.length ? Math.max(1, Math.round(xs.length / dim)) : null;
  }
  if (method === "embedRunBatch" || method === "prefillTokens") return args[0]?.length ?? null;
  if (["runHidden", "embedRun", "prefillToken", "headFromHidden"].includes(method)) return 1;
  return null;
}

function patchMethod(proto, method) {
  const original = proto?.[method];
  if (typeof original !== "function" || original[PATCHED]) return;

  async function instrumented(...args) {
    const started = performance.now();
    try {
      const result = await original.apply(this, args);
      const ms = performance.now() - started;
      const tokens = tokenCountFor.call(this, method, args);
      emit("engine:operation", {
        operation: method,
        ms: Number(ms.toFixed(3)),
        tokens,
        ...layerInfo(this),
      });

      if (method === "headFromHidden" && engineRunStartedAt != null && !engineFirstTokenSeen) {
        engineFirstTokenSeen = true;
        emit("engine:ttft", { ms: Number((performance.now() - engineRunStartedAt).toFixed(3)) });
      }
      return result;
    } catch (error) {
      emit("engine:operation-error", {
        operation: method,
        ms: Number((performance.now() - started).toFixed(3)),
        error: error?.message || String(error),
        ...layerInfo(this),
      });
      throw error;
    }
  }

  instrumented[PATCHED] = true;
  proto[method] = instrumented;
}

function patchReset(proto) {
  const original = proto?.reset;
  if (typeof original !== "function" || original[PATCHED]) return;

  function instrumentedReset(...args) {
    engineRunStartedAt = performance.now();
    engineFirstTokenSeen = false;
    emit("engine:run-start");
    return original.apply(this, args);
  }

  instrumentedReset[PATCHED] = true;
  proto.reset = instrumentedReset;
}

function patchSpecStep(proto) {
  const original = proto?.specStep;
  if (typeof original !== "function" || original[PATCHED]) return;

  async function instrumentedSpecStep(...args) {
    const before = this?.mtp?.stats ? { ...this.mtp.stats } : null;
    const requestedDepth = Number.isFinite(args[2]) ? args[2] : null;
    const started = performance.now();
    try {
      const result = await original.apply(this, args);
      const after = this?.mtp?.stats ? { ...this.mtp.stats } : null;
      emit("engine:spec-step", {
        ms: Number((performance.now() - started).toFixed(3)),
        requestedDepth,
        returnedTokens: Array.isArray(result) || ArrayBuffer.isView(result) ? result.length : null,
        accepted: before && after ? Math.max(0, (after.accepted || 0) - (before.accepted || 0)) : null,
        drafts: before && after ? Math.max(0, (after.drafts || 0) - (before.drafts || 0)) : null,
        ...layerInfo(this),
      });
      return result;
    } catch (error) {
      emit("engine:spec-error", {
        requestedDepth,
        ms: Number((performance.now() - started).toFixed(3)),
        error: error?.message || String(error),
      });
      throw error;
    }
  }

  instrumentedSpecStep[PATCHED] = true;
  proto.specStep = instrumentedSpecStep;
}

function patchEngine(Engine) {
  const proto = Engine?.prototype;
  if (!proto || proto[PATCHED]) return;
  patchReset(proto);
  for (const method of [
    "runHidden",
    "runHiddenBatch",
    "embedRun",
    "embedRunBatch",
    "headFromHidden",
    "prefillToken",
    "prefillTokens",
  ]) patchMethod(proto, method);
  patchSpecStep(proto);
  proto[PATCHED] = true;
}

function classifyResource(name) {
  const url = String(name || "");
  if (!/huggingface\.co|\.gguf(?:\?|$)|\.safetensors(?:\?|$)|model[^/]*\.json(?:\?|$)/i.test(url)) return null;
  if (/\.gguf(?:\?|$)/i.test(url)) return "gguf";
  if (/\.safetensors(?:\?|$)/i.test(url)) return "safetensors";
  if (/tokenizer/i.test(url)) return "tokenizer";
  if (/config|model[^/]*\.json/i.test(url)) return "metadata";
  return "model-resource";
}

function installResourceObserver() {
  if (!("PerformanceObserver" in window)) return;
  try {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const resourceKind = classifyResource(entry.name);
        if (!resourceKind) continue;
        const transferSize = Number(entry.transferSize || 0);
        const encodedBodySize = Number(entry.encodedBodySize || 0);
        emit("model:resource", {
          resourceKind,
          durationMs: Number(entry.duration.toFixed(3)),
          transferBytes: transferSize || null,
          encodedBytes: encodedBodySize || null,
          decodedBytes: Number(entry.decodedBodySize || 0) || null,
          cacheSignal: transferSize === 0 && encodedBodySize > 0 ? "likely-cache-or-revalidated" : "network-or-unknown",
          host: (() => { try { return new URL(entry.name).host; } catch { return null; } })(),
        });
      }
    });
    observer.observe({ type: "resource", buffered: true });
  } catch {}
}

async function installBatterySignal() {
  if (typeof navigator.getBattery !== "function") return;
  try {
    const battery = await navigator.getBattery();
    const report = () => emit("device:battery", {
      level: Number.isFinite(battery.level) ? Number(battery.level.toFixed(4)) : null,
      charging: !!battery.charging,
      chargingTime: Number.isFinite(battery.chargingTime) ? battery.chargingTime : null,
      dischargingTime: Number.isFinite(battery.dischargingTime) ? battery.dischargingTime : null,
    });
    report();
    battery.addEventListener("levelchange", report);
    battery.addEventListener("chargingchange", report);
  } catch {}
}

patchEngine(DenseEngine);
patchEngine(Qwen35Engine);
installResourceObserver();
installBatterySignal();
emit("telemetry:ready", { version: 2 });
