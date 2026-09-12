(() => {
  const MAX_ENTRIES = 600;
  const MAX_RUNS = 250;
  const MAX_PENDING_TELEMETRY = 500;
  const RUN_STORAGE_KEY = "field-station:experiment-runs:v1";
  const PRIVATE_KEY = /^(prompt|promptText|message|messages|reply|answer|response|content|contents|text)$/i;
  const PRIVATE_SUFFIX = /(prompt|message|reply|answer|response)(Text|Content)$/i;

  const entries = [];
  const startedAt = Date.now();
  let experimentRuns = loadRuns();
  let activeRun = null;
  let lastStatus = "";
  let pendingEngineRunAt = null;
  let pendingTelemetry = [];
  let batteryState = null;
  let modelLoad = null;
  let lastModelLoad = null;

  const sessionTelemetry = {
    engineExecutionMs: 0,
    engineOperations: {},
    transportWireBytesSent: 0,
    transportWireBytesReceived: 0,
    transportFramesSent: 0,
    transportFramesReceived: 0,
    modelResources: 0,
    modelTransferBytes: 0,
  };

  function safe(value, depth = 0) {
    if (depth > 4) return "[depth]";
    if (value == null || ["string", "number", "boolean"].includes(typeof value)) return value;
    if (Array.isArray(value)) return value.slice(0, 100).map((item) => safe(item, depth + 1));
    if (typeof value === "object") {
      const out = {};
      for (const [key, val] of Object.entries(value).slice(0, 100)) {
        if (PRIVATE_KEY.test(key) || PRIVATE_SUFFIX.test(key)) continue;
        out[key] = safe(val, depth + 1);
      }
      return out;
    }
    return String(value);
  }

  function loadRuns() {
    try {
      const parsed = JSON.parse(localStorage.getItem(RUN_STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed.slice(-MAX_RUNS) : [];
    } catch {
      return [];
    }
  }

  function persistRuns() {
    try { localStorage.setItem(RUN_STORAGE_KEY, JSON.stringify(experimentRuns.slice(-MAX_RUNS))); } catch {}
  }

  function record(type, data = {}) {
    entries.push({ at: new Date().toISOString(), ms: Date.now() - startedAt, type, data: safe(data) });
    if (entries.length > MAX_ENTRIES) entries.splice(0, entries.length - MAX_ENTRIES);
    render();
  }

  function parseNumber(value) {
    const match = String(value || "").match(/-?\d+(?:\.\d+)?/);
    return match ? Number(match[0]) : null;
  }

  function round(value, places = 2) {
    if (!Number.isFinite(value)) return null;
    const scale = 10 ** places;
    return Math.round(value * scale) / scale;
  }

  function percentile(values, p) {
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!sorted.length) return null;
    if (sorted.length === 1) return round(sorted[0], 3);
    const index = (sorted.length - 1) * p;
    const lo = Math.floor(index), hi = Math.ceil(index);
    const value = lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
    return round(value, 3);
  }

  function median(values) {
    return percentile(values, 0.5);
  }

  function currentModel() {
    const select = document.getElementById("ai-model");
    return {
      key: select?.value || null,
      label: select?.selectedOptions?.[0]?.textContent?.trim() || null,
    };
  }

  function currentWire() {
    const wire = (new URLSearchParams(location.search).get("wire") || "stripe4").toLowerCase();
    const stripes = wire === "off" ? 0 : wire.startsWith("stripe")
      ? Math.max(1, Math.min(8, Number.parseInt(wire.slice(6), 10) || 1))
      : 1;
    return { wire, stripes };
  }

  function clusterSnapshot() {
    const devicesDetail = [...document.querySelectorAll(".peer-card")].map((card) => {
      const gpu = card.querySelector(".peer-gpu")?.textContent?.trim() || null;
      const rtt = card.querySelector(".rtt")?.textContent || "";
      const bandwidth = card.querySelector(".bw")?.textContent || "";
      const contribution = card.querySelector(".buf")?.textContent || "";
      return {
        self: card.classList.contains("self"),
        gpu,
        rttMs: /ms/i.test(rtt) ? parseNumber(rtt) : null,
        bandwidthMbps: /mbps/i.test(bandwidth) ? parseNumber(bandwidth) : null,
        contributionGB: /gb/i.test(contribution) ? parseNumber(contribution) : null,
      };
    });

    const summary = document.getElementById("cluster-summary")?.textContent || "";
    const match = summary.match(/(\d+)\s+devices?\s+·\s+(\d+)\s+WebGPU\s+·\s+([\d.]+)\s+GB pledged/i);
    const rtts = devicesDetail.map((device) => device.rttMs).filter(Number.isFinite);
    const bandwidths = devicesDetail.map((device) => device.bandwidthMbps).filter(Number.isFinite);

    return {
      devices: match ? Number(match[1]) : devicesDetail.length || null,
      webgpuDevices: match ? Number(match[2]) : devicesDetail.filter((device) => device.gpu && !/no WebGPU/i.test(device.gpu)).length,
      pledgedGB: match ? Number(match[3]) : devicesDetail.reduce((sum, device) => sum + (device.contributionGB || 0), 0) || null,
      medianRttMs: median(rtts),
      minBandwidthMbps: bandwidths.length ? Math.min(...bandwidths) : null,
      maxBandwidthMbps: bandwidths.length ? Math.max(...bandwidths) : null,
      devicesDetail,
    };
  }

  function newRunTelemetry() {
    return {
      phase: "prefill",
      engineExecutionMs: 0,
      engineOperations: {},
      engineTtftMs: null,
      pipelineStarts: new Map(),
      pipelineLapMs: [],
      decodePipelineLapMs: [],
      remoteWorkerMs: [],
      decodeRemoteWorkerMs: [],
      transport: {
        sendWireBytes: 0,
        receiveWireBytes: 0,
        prefillWireBytes: 0,
        decodeWireBytes: 0,
        framesSent: 0,
        framesReceived: 0,
      },
      spec: {
        steps: 0,
        accepted: 0,
        drafts: 0,
        returnedTokens: 0,
        totalMs: 0,
        depthCounts: {},
      },
    };
  }

  function beginRun(promptTokens = null, startedPerf = null) {
    const model = currentModel();
    const cluster = clusterSnapshot();
    activeRun = {
      id: crypto.randomUUID?.() || `run-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      startedAt: new Date().toISOString(),
      startedPerf: Number.isFinite(startedPerf) ? startedPerf : performance.now(),
      promptTokens: Number.isFinite(promptTokens) ? promptTokens : null,
      modelKey: model.key,
      modelLabel: model.label,
      ...currentWire(),
      clusterAtStart: cluster,
      batteryStart: batteryState ? { ...batteryState } : null,
      modelLoad: lastModelLoad ? { ...lastModelLoad } : null,
      firstOutputObservedMs: null,
      lastObservedOutputTokens: 0,
      lastObservedTokensPerSecond: null,
      telemetry: newRunTelemetry(),
    };
    record("experiment:run-start", {
      id: activeRun.id,
      modelKey: activeRun.modelKey,
      promptTokens: activeRun.promptTokens,
      devices: cluster.devices,
      wire: activeRun.wire,
    });

    const buffered = pendingTelemetry;
    pendingTelemetry = [];
    for (const detail of buffered) applyRunTelemetry(detail);
  }

  function ensureRun(promptTokens = null) {
    if (!activeRun) beginRun(promptTokens, pendingEngineRunAt);
    if (Number.isFinite(promptTokens)) activeRun.promptTokens = promptTokens;
    return activeRun;
  }

  function saveRun(run) {
    experimentRuns.push(safe(run));
    if (experimentRuns.length > MAX_RUNS) experimentRuns.splice(0, experimentRuns.length - MAX_RUNS);
    persistRuns();
    record("experiment:run", run);
    activeRun = null;
    pendingEngineRunAt = null;
    pendingTelemetry = [];
  }

  function soloBaseline(modelKey) {
    const candidates = experimentRuns.filter((run) =>
      run.status === "complete" && run.modelKey === modelKey && run.devices === 1 && Number.isFinite(run.outputTokensPerSecond));
    if (!candidates.length) return null;
    return Math.max(...candidates.map((run) => run.outputTokensPerSecond));
  }

  function summariseTelemetry(run, devices, outputTokens) {
    const t = run.telemetry;
    const observedWireBytes = t.transport.sendWireBytes + t.transport.receiveWireBytes;
    const decodeObservedWireBytes = t.transport.decodeWireBytes;
    const clusterScale = devices > 1 ? devices / 2 : 0;
    const estimatedClusterWireBytes = devices > 1 ? Math.round(observedWireBytes * clusterScale) : 0;
    const estimatedDecodeWireBytes = devices > 1 ? Math.round(decodeObservedWireBytes * clusterScale) : 0;
    const specAcceptance = t.spec.drafts > 0 ? round(t.spec.accepted / t.spec.drafts, 4) : null;
    const weightedDepth = Object.entries(t.spec.depthCounts)
      .reduce((sum, [depth, count]) => sum + Number(depth) * count, 0);

    return {
      engineTtftMs: t.engineTtftMs,
      localEngineExecutionMs: round(t.engineExecutionMs, 3),
      localEngineOperations: t.engineOperations,
      pipelineLapP50Ms: percentile(t.pipelineLapMs, 0.5),
      pipelineLapP95Ms: percentile(t.pipelineLapMs, 0.95),
      decodePipelineLapP50Ms: percentile(t.decodePipelineLapMs, 0.5),
      decodePipelineLapP95Ms: percentile(t.decodePipelineLapMs, 0.95),
      remoteWorkerP50Ms: percentile(t.remoteWorkerMs, 0.5),
      remoteWorkerP95Ms: percentile(t.remoteWorkerMs, 0.95),
      decodeRemoteWorkerP50Ms: percentile(t.decodeRemoteWorkerMs, 0.5),
      decodeRemoteWorkerP95Ms: percentile(t.decodeRemoteWorkerMs, 0.95),
      hostWireBytes: observedWireBytes,
      hostPrefillWireBytes: t.transport.prefillWireBytes,
      hostDecodeWireBytes: decodeObservedWireBytes,
      estimatedClusterWireBytes,
      estimatedDecodeWireBytes,
      estimatedDecodeWireBytesPerOutputToken: outputTokens > 0 ? round(estimatedDecodeWireBytes / outputTokens, 1) : null,
      transportFramesSent: t.transport.framesSent,
      transportFramesReceived: t.transport.framesReceived,
      speculativeSteps: t.spec.steps,
      speculativeAccepted: t.spec.accepted,
      speculativeDrafts: t.spec.drafts,
      speculativeAcceptanceRate: specAcceptance,
      speculativeAverageDepth: t.spec.steps > 0 ? round(weightedDepth / t.spec.steps, 3) : null,
      speculativeDepthCounts: t.spec.depthCounts,
      speculativeStepMs: round(t.spec.totalMs, 3),
    };
  }

  function completeRun({ prefillSeconds, outputTokens, outputTokensPerSecond, devices }) {
    const run = ensureRun();
    const cluster = clusterSnapshot();
    const actualDevices = devices || cluster.devices || run.clusterAtStart.devices || 1;
    const prefillMs = Number.isFinite(prefillSeconds) ? Math.round(prefillSeconds * 1000) : null;
    const baselineTps = soloBaseline(run.modelKey);
    const collectiveSpeedup = baselineTps && actualDevices > 1 ? round(outputTokensPerSecond / baselineTps, 4) : null;
    const collectiveEfficiency = collectiveSpeedup != null ? round(collectiveSpeedup / actualDevices, 4) : null;
    const deep = summariseTelemetry(run, actualDevices, outputTokens);
    const batteryEnd = batteryState ? { ...batteryState } : null;
    const batteryDropPct = run.batteryStart && batteryEnd && !run.batteryStart.charging && !batteryEnd.charging
      ? round(Math.max(0, (run.batteryStart.level - batteryEnd.level) * 100), 3)
      : null;

    saveRun({
      id: run.id,
      startedAt: run.startedAt,
      finishedAt: new Date().toISOString(),
      status: "complete",
      modelKey: run.modelKey,
      modelLabel: run.modelLabel,
      promptTokens: run.promptTokens,
      outputTokens,
      prefillMs,
      prefillTokensPerSecond: Number.isFinite(prefillSeconds) && prefillSeconds > 0 && Number.isFinite(run.promptTokens)
        ? round(run.promptTokens / prefillSeconds, 2)
        : null,
      decodeMs: Number.isFinite(outputTokensPerSecond) && outputTokensPerSecond > 0
        ? Math.round((outputTokens / outputTokensPerSecond) * 1000)
        : null,
      outputTokensPerSecond,
      firstOutputObservedMs: run.firstOutputObservedMs,
      totalObservedMs: Math.round(performance.now() - run.startedPerf),
      devices: actualDevices,
      webgpuDevices: cluster.webgpuDevices,
      pledgedGB: cluster.pledgedGB,
      medianRttMs: cluster.medianRttMs,
      minBandwidthMbps: cluster.minBandwidthMbps,
      maxBandwidthMbps: cluster.maxBandwidthMbps,
      wire: run.wire,
      stripes: run.stripes,
      hostDevice: cluster.devicesDetail.find((device) => device.self)?.gpu || null,
      modelReadyMs: run.modelLoad?.readyMs ?? null,
      modelResourceCount: run.modelLoad?.resourceCount ?? null,
      modelTransferBytes: run.modelLoad?.transferBytes ?? null,
      modelLikelyCacheResources: run.modelLoad?.likelyCacheResources ?? null,
      soloBaselineTokensPerSecond: baselineTps,
      collectiveSpeedup,
      collectiveEfficiency,
      batteryLevelStart: run.batteryStart?.level ?? null,
      batteryLevelEnd: batteryEnd?.level ?? null,
      batteryDropPct,
      ...deep,
      cluster: cluster.devicesDetail,
    });
  }

  function promptTokensFromError(error) {
    const match = String(error || "").match(/prompt is\s+(\d+)\s+tokens/i);
    return match ? Number(match[1]) : null;
  }

  function failRun(error) {
    const run = ensureRun(promptTokensFromError(error));
    const cluster = clusterSnapshot();
    const actualDevices = cluster.devices || run.clusterAtStart.devices || 1;
    const deep = summariseTelemetry(run, actualDevices, run.lastObservedOutputTokens || 0);
    saveRun({
      id: run.id,
      startedAt: run.startedAt,
      finishedAt: new Date().toISOString(),
      status: "failed",
      error: String(error || "generation failed").slice(0, 240),
      modelKey: run.modelKey,
      modelLabel: run.modelLabel,
      promptTokens: run.promptTokens,
      outputTokens: run.lastObservedOutputTokens,
      outputTokensPerSecond: run.lastObservedTokensPerSecond,
      totalObservedMs: Math.round(performance.now() - run.startedPerf),
      devices: actualDevices,
      webgpuDevices: cluster.webgpuDevices,
      pledgedGB: cluster.pledgedGB,
      medianRttMs: cluster.medianRttMs,
      minBandwidthMbps: cluster.minBandwidthMbps,
      maxBandwidthMbps: cluster.maxBandwidthMbps,
      wire: run.wire,
      stripes: run.stripes,
      hostDevice: cluster.devicesDetail.find((device) => device.self)?.gpu || null,
      modelReadyMs: run.modelLoad?.readyMs ?? null,
      ...deep,
      cluster: cluster.devicesDetail,
    });
  }

  function maybeStartModelLoad(status) {
    if (modelLoad) return;
    if (!/(reading model index|requesting GPU|downloading layers|building GPU pipelines|starting\s+)/i.test(status)) return;
    const model = currentModel();
    modelLoad = {
      modelKey: model.key,
      startedPerf: performance.now(),
      resourceCount: 0,
      transferBytes: 0,
      likelyCacheResources: 0,
    };
  }

  function maybeFinishModelLoad(status) {
    if (!modelLoad) return;
    if (!/(cluster online|solo:.*ready|layers\s+\d+.*ready)/i.test(status)) return;
    lastModelLoad = {
      modelKey: modelLoad.modelKey,
      readyMs: Math.round(performance.now() - modelLoad.startedPerf),
      resourceCount: modelLoad.resourceCount,
      transferBytes: modelLoad.transferBytes || null,
      likelyCacheResources: modelLoad.likelyCacheResources,
    };
    record("model:ready", lastModelLoad);
    modelLoad = null;
  }

  function observeAiStatus() {
    const status = document.getElementById("ai-status");
    if (!status) return;
    const inspect = () => {
      const value = status.textContent?.trim() || "";
      if (!value || value === lastStatus) return;
      lastStatus = value;
      maybeStartModelLoad(value);
      maybeFinishModelLoad(value);

      const prefill = value.match(/^prefill:\s*(\d+)\s+tokens/i);
      if (prefill) {
        const promptTokens = Number(prefill[1]);
        if (!activeRun) beginRun(promptTokens, pendingEngineRunAt);
        else activeRun.promptTokens = promptTokens;
        return;
      }

      const generating = value.match(/^generating…?\s*(\d+)\s+tok\s*·\s*([\d.]+)\s+tok\/s/i);
      if (generating) {
        const run = ensureRun();
        if (run.firstOutputObservedMs == null) run.firstOutputObservedMs = Math.round(performance.now() - run.startedPerf);
        run.lastObservedOutputTokens = Number(generating[1]);
        run.lastObservedTokensPerSecond = Number(generating[2]);
        run.telemetry.phase = "decode";
        return;
      }

      const ready = value.match(/^ready\s+[—-]\s+prefill\s+([\d.]+)s,\s*(\d+)\s+tok\s*·\s*([\d.]+)\s+tok\/s\s*·\s*(\d+)\s+devices?/i);
      if (ready) {
        completeRun({
          prefillSeconds: Number(ready[1]),
          outputTokens: Number(ready[2]),
          outputTokensPerSecond: Number(ready[3]),
          devices: Number(ready[4]),
        });
        return;
      }

      const failed = value.match(/^generation failed:\s*(.+)$/i);
      if (failed) failRun(failed[1]);
    };
    new MutationObserver(inspect).observe(status, { childList: true, subtree: true, characterData: true });
    inspect();
  }

  function applySessionTelemetry(detail) {
    if (detail.type === "engine:operation") {
      sessionTelemetry.engineExecutionMs += Number(detail.ms || 0);
      const op = sessionTelemetry.engineOperations[detail.operation] || { calls: 0, ms: 0, tokens: 0 };
      op.calls++;
      op.ms = round(op.ms + Number(detail.ms || 0), 3);
      op.tokens += Number(detail.tokens || 0);
      sessionTelemetry.engineOperations[detail.operation] = op;
    } else if (detail.type === "transport:frame-send") {
      sessionTelemetry.transportFramesSent++;
      sessionTelemetry.transportWireBytesSent += Number(detail.wireBytes || 0);
    } else if (detail.type === "transport:frame-receive") {
      sessionTelemetry.transportFramesReceived++;
      sessionTelemetry.transportWireBytesReceived += Number(detail.wireBytes || 0);
    } else if (detail.type === "model:resource") {
      sessionTelemetry.modelResources++;
      sessionTelemetry.modelTransferBytes += Number(detail.transferBytes || 0);
    }
  }

  function applyRunTelemetry(detail) {
    if (!activeRun) return;
    const t = activeRun.telemetry;

    if (detail.type === "engine:operation") {
      t.engineExecutionMs += Number(detail.ms || 0);
      const op = t.engineOperations[detail.operation] || { calls: 0, ms: 0, tokens: 0 };
      op.calls++;
      op.ms = round(op.ms + Number(detail.ms || 0), 3);
      op.tokens += Number(detail.tokens || 0);
      t.engineOperations[detail.operation] = op;
      return;
    }

    if (detail.type === "engine:ttft") {
      t.engineTtftMs = Number(detail.ms || 0);
      t.phase = "decode";
      return;
    }

    if (detail.type === "engine:spec-step") {
      t.spec.steps++;
      t.spec.accepted += Number(detail.accepted || 0);
      t.spec.drafts += Number(detail.drafts || 0);
      t.spec.returnedTokens += Number(detail.returnedTokens || 0);
      t.spec.totalMs += Number(detail.ms || 0);
      if (Number.isFinite(detail.requestedDepth)) {
        t.spec.depthCounts[detail.requestedDepth] = (t.spec.depthCounts[detail.requestedDepth] || 0) + 1;
      }
      return;
    }

    if (detail.type === "transport:frame-send" || detail.type === "transport:frame-receive") {
      const bytes = Number(detail.wireBytes || 0);
      const isSend = detail.type === "transport:frame-send";
      if (isSend) {
        t.transport.sendWireBytes += bytes;
        t.transport.framesSent++;
      } else {
        t.transport.receiveWireBytes += bytes;
        t.transport.framesReceived++;
      }
      if (t.phase === "decode") t.transport.decodeWireBytes += bytes;
      else t.transport.prefillWireBytes += bytes;

      const family = String(detail.kind || "").endsWith("-b") ? "batch" : "single";
      const key = `${family}:${detail.position}`;
      if (isSend && (detail.kind === "ai-hidden" || detail.kind === "ai-hidden-b")) {
        t.pipelineStarts.set(key, { atPerf: detail.atPerf, phase: t.phase });
      } else if (!isSend && (detail.kind === "ai-hiddenret" || detail.kind === "ai-hiddenret-b")) {
        const start = t.pipelineStarts.get(key);
        if (start) {
          const lap = detail.atPerf - start.atPerf;
          t.pipelineLapMs.push(lap);
          if (start.phase === "decode") t.decodePipelineLapMs.push(lap);
          t.pipelineStarts.delete(key);
        }
        if (Number.isFinite(detail.cumulativeWorkerMs) && detail.cumulativeWorkerMs > 0) {
          t.remoteWorkerMs.push(detail.cumulativeWorkerMs);
          if (t.phase === "decode") t.decodeRemoteWorkerMs.push(detail.cumulativeWorkerMs);
        }
      }
    }
  }

  function onTelemetry(event) {
    const detail = event.detail || {};
    if (!detail.type) return;

    if (detail.type === "device:battery") {
      batteryState = {
        level: Number.isFinite(detail.level) ? detail.level : null,
        charging: !!detail.charging,
      };
      return;
    }

    if (detail.type === "model:resource") {
      if (modelLoad) {
        modelLoad.resourceCount++;
        modelLoad.transferBytes += Number(detail.transferBytes || 0);
        if (detail.cacheSignal === "likely-cache-or-revalidated") modelLoad.likelyCacheResources++;
      }
      applySessionTelemetry(detail);
      return;
    }

    if (detail.type === "telemetry:ready") {
      record("telemetry:ready", { version: detail.version });
      return;
    }

    if (detail.type === "engine:run-start") {
      pendingEngineRunAt = Number.isFinite(detail.atPerf) ? detail.atPerf : performance.now();
      pendingTelemetry = [];
      return;
    }

    applySessionTelemetry(detail);
    if (activeRun) applyRunTelemetry(detail);
    else if (pendingEngineRunAt != null) {
      pendingTelemetry.push(detail);
      if (pendingTelemetry.length > MAX_PENDING_TELEMETRY) pendingTelemetry.shift();
    }
  }

  function snapshot() {
    return {
      fieldStation: "collective-compute",
      capturedAt: new Date().toISOString(),
      page: location.origin + location.pathname,
      userAgent: navigator.userAgent,
      online: navigator.onLine,
      battery: batteryState,
      localDeviceTelemetry: {
        ...sessionTelemetry,
        engineExecutionMs: round(sessionTelemetry.engineExecutionMs, 3),
      },
      latestModelLoad: lastModelLoad,
      experimentRuns: [...experimentRuns],
      entries: [...entries],
    };
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(snapshot(), null, 2));
      record("diagnostics:copied", { entries: entries.length, runs: experimentRuns.length });
      return true;
    } catch {
      return false;
    }
  }

  function timestamp() {
    return new Date().toISOString().replaceAll(":", "-");
  }

  function downloadBlob(content, type, filename) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function downloadJson() {
    downloadBlob(JSON.stringify(snapshot(), null, 2), "application/json", `field-station-diagnostics-${timestamp()}.json`);
    record("diagnostics:exported", { entries: entries.length, runs: experimentRuns.length });
  }

  function csvCell(value) {
    if (value == null) return "";
    const string = typeof value === "object" ? JSON.stringify(value) : String(value);
    return /[",\n]/.test(string) ? `"${string.replaceAll('"', '""')}"` : string;
  }

  function downloadRunsCsv() {
    const fields = [
      "startedAt", "status", "modelKey", "modelLabel", "promptTokens", "outputTokens",
      "prefillMs", "prefillTokensPerSecond", "decodeMs", "outputTokensPerSecond",
      "engineTtftMs", "firstOutputObservedMs", "totalObservedMs", "devices", "webgpuDevices", "pledgedGB",
      "medianRttMs", "minBandwidthMbps", "maxBandwidthMbps", "wire", "stripes", "hostDevice",
      "modelReadyMs", "modelResourceCount", "modelTransferBytes", "modelLikelyCacheResources",
      "pipelineLapP50Ms", "pipelineLapP95Ms", "decodePipelineLapP50Ms", "decodePipelineLapP95Ms",
      "remoteWorkerP50Ms", "remoteWorkerP95Ms", "decodeRemoteWorkerP50Ms", "decodeRemoteWorkerP95Ms",
      "hostWireBytes", "hostPrefillWireBytes", "hostDecodeWireBytes", "estimatedClusterWireBytes",
      "estimatedDecodeWireBytes", "estimatedDecodeWireBytesPerOutputToken",
      "speculativeSteps", "speculativeAccepted", "speculativeDrafts", "speculativeAcceptanceRate",
      "speculativeAverageDepth", "speculativeDepthCounts", "localEngineExecutionMs",
      "soloBaselineTokensPerSecond", "collectiveSpeedup", "collectiveEfficiency",
      "batteryLevelStart", "batteryLevelEnd", "batteryDropPct", "error",
    ];
    const rows = [fields.join(","), ...experimentRuns.map((run) => fields.map((field) => csvCell(run[field])).join(","))];
    downloadBlob(rows.join("\n"), "text/csv;charset=utf-8", `field-station-runs-${timestamp()}.csv`);
    record("experiment:csv-exported", { runs: experimentRuns.length });
  }

  function clearRuns() {
    experimentRuns = [];
    persistRuns();
    record("experiment:runs-cleared");
  }

  function ensureDialog() {
    let dialog = document.getElementById("fs-diagnostics-dialog");
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "fs-diagnostics-dialog";
    dialog.innerHTML = `
      <div class="diag-head">
        <div><div class="fs-type">APPARATUS · EXPERIMENT LOG</div><h2>What is the room doing?</h2></div>
        <button type="button" class="diag-close" aria-label="Close diagnostics">Close</button>
      </div>
      <p class="diag-note">Run metrics are kept in this browser so experiments can be compared later. Prompt and answer text are deliberately excluded. Battery change is recorded only where the browser exposes it and is not treated as an energy measurement.</p>
      <div class="diag-run-summary" aria-live="polite"></div>
      <div class="diag-actions">
        <button type="button" data-diag-runs-csv>Export runs CSV</button>
        <button type="button" data-diag-export>Export JSON</button>
        <button type="button" data-diag-copy>Copy JSON</button>
        <button type="button" data-diag-clear>Clear session log</button>
        <button type="button" data-diag-clear-runs>Clear run history</button>
      </div>
      <pre class="diag-log" aria-live="polite"></pre>`;
    document.body.appendChild(dialog);
    dialog.querySelector(".diag-close").addEventListener("click", () => dialog.close());
    dialog.querySelector("[data-diag-copy]").addEventListener("click", async () => {
      if (!(await copy())) dialog.querySelector(".diag-note").textContent = "Clipboard access failed. Use Export JSON instead.";
    });
    dialog.querySelector("[data-diag-export]").addEventListener("click", downloadJson);
    dialog.querySelector("[data-diag-runs-csv]").addEventListener("click", downloadRunsCsv);
    dialog.querySelector("[data-diag-clear]").addEventListener("click", () => { entries.length = 0; render(); });
    dialog.querySelector("[data-diag-clear-runs]").addEventListener("click", clearRuns);
    return dialog;
  }

  function runLine(run) {
    const when = run.startedAt ? new Date(run.startedAt).toLocaleString() : "unknown time";
    const model = run.modelLabel || run.modelKey || "model";
    if (run.status !== "complete") return `${when}  ${model}  failed${run.error ? ` · ${run.error}` : ""}`;
    const deeper = [
      Number.isFinite(run.engineTtftMs) ? `TTFT ${Math.round(run.engineTtftMs)} ms` : null,
      Number.isFinite(run.decodePipelineLapP95Ms) ? `lap p95 ${Math.round(run.decodePipelineLapP95Ms)} ms` : null,
      Number.isFinite(run.collectiveEfficiency) ? `collective ${(run.collectiveEfficiency * 100).toFixed(0)}%` : null,
    ].filter(Boolean).join(" · ");
    return `${when}  ${model}  ${run.devices || "?"} device${run.devices === 1 ? "" : "s"}  ${run.outputTokens ?? "?"} tok · ${run.outputTokensPerSecond ?? "?"} tok/s${deeper ? ` · ${deeper}` : ""}`;
  }

  function render() {
    const log = document.querySelector("#fs-diagnostics-dialog .diag-log");
    if (log) {
      log.textContent = entries.slice(-120).map((entry) => {
        const seconds = (entry.ms / 1000).toFixed(1).padStart(7);
        const data = Object.keys(entry.data || {}).length ? `  ${JSON.stringify(entry.data)}` : "";
        return `${seconds}s  ${entry.type}${data}`;
      }).join("\n");
      log.scrollTop = log.scrollHeight;
    }

    const summary = document.querySelector("#fs-diagnostics-dialog .diag-run-summary");
    if (summary) {
      const recent = experimentRuns.slice(-5).reverse().map(runLine).join("\n");
      summary.textContent = experimentRuns.length
        ? `${experimentRuns.length} saved run${experimentRuns.length === 1 ? "" : "s"}\n${recent}`
        : "No completed runs recorded yet. Run the model once and the useful performance measures will appear here.";
    }
  }

  function open() {
    const dialog = ensureDialog();
    render();
    dialog.showModal();
  }

  window.fieldStationDiagnostics = {
    record,
    snapshot,
    copy,
    download: downloadJson,
    downloadRunsCsv,
    experimentRuns: () => [...experimentRuns],
    open,
  };

  window.addEventListener("field-station-telemetry", onTelemetry);
  window.addEventListener("error", (event) => record("browser:error", {
    name: event.error?.name || "Error",
    error: event.error?.message || event.message || "unknown error",
    filename: event.filename?.split("/").pop() || null,
    line: event.lineno || null,
    column: event.colno || null,
  }));
  window.addEventListener("unhandledrejection", (event) => {
    const error = event.reason?.message || String(event.reason || "unknown rejection");
    record("browser:rejection", { error });
    if (pendingEngineRunAt != null && !activeRun) failRun(error);
  });
  window.addEventListener("online", () => record("network:online"));
  window.addEventListener("offline", () => record("network:offline"));
  document.addEventListener("visibilitychange", () => record("page:visibility", { state: document.visibilityState }));
  document.addEventListener("click", (event) => {
    if (event.target?.closest?.("#diagnostics-btn")) open();
  });

  observeAiStatus();
  record("session:start", { webgpu: !!navigator.gpu, savedRuns: experimentRuns.length });
})();