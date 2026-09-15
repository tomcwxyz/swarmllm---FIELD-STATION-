import {
  formatBytes,
  formatDuration,
  initialContributionState,
  reduceContribution,
} from "../../field-station/contribution.js";

const eq = (a, b, m) => {
  const ja = JSON.stringify(a), jb = JSON.stringify(b);
  if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb);
};

Deno.test("contribution: engine operations accumulate compute and layer range", () => {
  let state = initialContributionState();
  state = reduceContribution(state, {
    type: "engine:operation",
    operation: "embedRunBatch",
    ms: 12.5,
    tokens: 4,
    layerStart: 8,
    layerEnd: 16,
    atPerf: 10,
  });
  eq(state.computeMs, 12.5);
  eq(state.engineTokens, 4);
  eq([state.layerStart, state.layerEnd], [8, 16]);
  eq(state.lastActivity, "working");
});

Deno.test("contribution: wire traffic counts bytes and incoming token work once", () => {
  let state = initialContributionState();
  state = reduceContribution(state, { type: "transport:frame-send", wireBytes: 1000, tokens: 8, atPerf: 1 });
  state = reduceContribution(state, { type: "transport:frame-receive", wireBytes: 1200, tokens: 8, atPerf: 2 });
  eq(state.sentBytes, 1000);
  eq(state.recvBytes, 1200);
  eq(state.transportTokens, 8);
});

Deno.test("contribution: model resources and battery remain explicit signals", () => {
  let state = initialContributionState();
  state = reduceContribution(state, { type: "model:resource", encodedBytes: 2048, atPerf: 1 });
  state = reduceContribution(state, { type: "device:battery", level: 0.73, charging: false, atPerf: 2 });
  eq(state.modelBytes, 2048);
  eq(state.battery, { level: 0.73, charging: false });
});

Deno.test("contribution: human-readable units stay compact", () => {
  eq(formatBytes(0), "0 B");
  eq(formatBytes(1024), "1.0 KB");
  eq(formatBytes(5 * 1024 * 1024), "5.0 MB");
  eq(formatDuration(420), "420 ms");
  eq(formatDuration(12500), "12.5 s");
  eq(formatDuration(61000), "1m 1s");
});
