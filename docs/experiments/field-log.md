# FIELD STATION experiment log

The room now keeps a small, local record of inference runs so FIELD STATION experiments can be compared across models, devices and network shapes without collecting conversation content.

## What is recorded

For each completed or failed generation the host browser records:

- model key and display label
- prompt and output token counts
- prefill duration and derived prompt tokens/second
- output/decode tokens/second and derived decode duration
- observed time from prefill start to the first rendered output token
- total observed generation time
- participating device count and WebGPU device count
- total pledged GPU memory
- host device/GPU description exposed by the browser
- median peer RTT where a reading is available
- minimum and maximum measured peer bandwidth where the optional bandwidth test has been run
- transport mode and stripe count
- per-device GPU description, contribution, RTT and measured bandwidth, without peer names
- failure state and a short error message when a generation fails

The latest 250 runs are stored in `localStorage` under `field-station:experiment-runs:v1`. They stay on that browser unless somebody explicitly exports them.

## What is deliberately not recorded

The experiment log does not retain prompt text, answer text, chat messages, room codes, invitation keys or peer names. The diagnostics scrubber removes common prompt/message/answer/content fields before anything is persisted or exported.

## Reading the measures

`outputTokensPerSecond` is the end-to-end decode rate already reported by the room runtime. In a multi-device room it therefore includes the cost of moving hidden states through the chain; this is more useful for FIELD STATION than a GPU-only kernel benchmark.

`prefillMs` is the runtime's existing prompt prefill duration. `prefillTokensPerSecond` is derived from the prompt token count and that duration.

`firstOutputObservedMs` is an observer-level measurement from the moment the UI reports prefill beginning until the first `generating…` status update. Treat it as an approximate time-to-first-token measure rather than a low-level engine timing.

RTT is only present after the room has measured it. Bandwidth is only present when the optional bandwidth test has been run, so blank values are expected and should not be interpreted as zero.

## Export

Open **Diagnostics / logs** in the room. The drawer now shows recent experiment runs and offers:

- **Export runs CSV** for analysis in a spreadsheet or notebook
- **Export JSON** for the complete run history plus current-session diagnostics
- **Copy JSON** for quick sharing/debugging
- separate controls for clearing the session log and persistent run history

## Useful next measurements

The current log deliberately reuses measurements the runtime already exposes. If the experiment needs a deeper second pass, the highest-value additions are exact engine-level time-to-first-token, model download/cache-hit duration, model-ready time, per-device layer execution time, pipeline lap percentiles, speculative-decoding acceptance rate/depth, bytes transferred per generated token, and energy/battery impact where the browser exposes a defensible signal.
