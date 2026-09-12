# FIELD STATION experiment log

The room keeps a local record of inference runs so FIELD STATION experiments can be compared across models, devices and network shapes without collecting conversation content.

## What is recorded

For each completed or failed generation the host browser records the basic run shape: model, prompt/output token counts, prefill time, decode tokens/second, device count, pledged GPU memory, transport configuration, RTT/bandwidth readings where available, host GPU description and failures.

The deeper apparatus layer also records:

- `engineTtftMs`: engine-level time from `engine.reset()` at generation start to completion of the first logits/head pass
- `localEngineExecutionMs` and `localEngineOperations`: execution time and call/token counts for instrumented engine operations on this device
- `pipelineLapP50Ms` / `pipelineLapP95Ms`: end-to-end host → worker chain → host activation lap percentiles
- decode-only pipeline lap percentiles
- `remoteWorkerP50Ms` / `remoteWorkerP95Ms`: cumulative processing/forwarding time carried back through the worker chain in the wire header
- decode-only remote worker processing percentiles
- observed host wire bytes, split into prefill and decode phases
- an estimated whole-cluster wire-byte count and decode bytes per generated token
- speculative decoding step count, accepted/drafted tokens, acceptance rate, requested depth distribution and average requested depth
- model-ready duration plus model-resource transfer/cache signals exposed by Resource Timing
- the fastest stored same-model solo baseline, distributed speed-up and `collectiveEfficiency`
- optional battery level change where the browser exposes the Battery Status API

The latest 250 runs are stored in `localStorage` under `field-station:experiment-runs:v1`. They stay on that browser unless somebody explicitly exports them.

## What is deliberately not recorded

The experiment log does not retain prompt text, answer text, chat messages, room codes, invitation keys or peer names. The diagnostics scrubber removes common prompt/message/answer/content fields before anything is persisted or exported.

## Reading the measures

`outputTokensPerSecond` is end-to-end room throughput, not a GPU-only benchmark. That is intentional: a distributed run is only useful if the whole apparatus produces tokens at a useful rate.

`engineTtftMs` starts at the runtime's engine reset and ends when the host has produced first-token logits. It includes tokenisation/prompt setup plus prefill and distributed inference up to that first result, but excludes UI rendering.

`firstOutputObservedMs` remains the browser/UI observation of first output. Comparing it with `engineTtftMs` is useful: the gap gives a rough indication of non-engine/UI overhead.

A **pipeline lap** starts when the host sends an activation into the first worker and ends when the corresponding return activation arrives back at the host. It therefore includes network delay, remote computation and forwarding.

The wire protocol now uses two previously unused header bytes to carry cumulative remote worker processing/forwarding milliseconds. This is deliberately compact and backwards-compatible. It gives a useful aggregate remote-compute signal, but it is not a profiler for individual named devices. Each participating browser's JSON export also contains its own `localDeviceTelemetry` if a per-device view is needed.

Whole-cluster transfer is estimated from the host's observed first/last leg transfer and the number of devices in the chain. It is useful for comparing room shapes, not for billing-grade network accounting. Decode bytes/token excludes prefill bytes.

`collectiveSpeedup` compares a distributed run with the **fastest stored single-device run for the same model**. `collectiveEfficiency` divides that speed-up by participating device count. For example, 1.8× solo speed on three devices is 60% collective efficiency. This is a deliberately demanding measure: it asks whether adding people/devices is actually buying useful throughput.

Resource Timing cache signals are best-effort. Cross-origin timing restrictions can hide byte counts, and a zero transfer size can mean cache/revalidation rather than a guaranteed cache hit.

Battery level is recorded only where the browser exposes it and only as a level change. It must not be presented as energy consumption: charging state, battery calibration, background activity and OS behaviour make that inference indefensible without external power measurement.

RTT is only present after the room has measured it. Bandwidth is only present when the optional bandwidth test has been run, so blank values are expected and should not be interpreted as zero.

## Export

Open **Diagnostics / logs** in the room. The drawer shows recent experiment runs and offers:

- **Export runs CSV** for comparison in a spreadsheet/notebook
- **Export JSON** for run history plus device/session telemetry
- **Copy JSON** for quick sharing/debugging
- separate controls for clearing the session log and persistent run history

## Suggested experiment pattern

For a useful field comparison, run the same model and prompt shape several times in each configuration rather than treating one generation as a benchmark. A strong first set is:

1. establish a solo baseline on each plausible host device;
2. repeat with two devices on the same local network;
3. add heterogeneous devices one at a time;
4. repeat over a higher-latency link;
5. compare output tok/s, TTFT, decode lap p95, remote-worker time, bytes/token and collective efficiency together.

The interesting question is not simply “how fast is the model?” but “under what conditions does pooled ordinary compute become more useful than the best machine acting alone?”
