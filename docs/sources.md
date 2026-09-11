# Sources · local document context

FIELD STATION Sources lets a participant add small text-based files to their own browser and ask the collective model questions grounded in selected excerpts.

## Privacy model

The **whole file never uploads to FIELD STATION and is not synchronised to the room**. It is read with the browser File API and kept only in that tab's memory.

However, Sources is not a private-document feature. When a participant asks a question, the browser selects a small set of relevant excerpts. Those excerpts become part of the model prompt and therefore enter the distributed inference path. Treat any excerpt used for inference as visible to participating devices, consistent with the room's existing privacy warning.

Do not attach confidential, client, personal or sensitive material.

## v1 formats

- TXT
- Markdown (`.md`, `.markdown`)
- CSV
- JSON

Limits are deliberately conservative while the experiment is young: up to 6 files, 2 MB each.

PDF is the next format. It should be parsed locally in the browser and feed the same chunk/retrieval layer rather than creating a separate upload path.

## Retrieval

`field-station/sources.js` normalises and chunks source text locally. At question time it:

1. extracts meaningful terms from the question;
2. ranks chunks by simple lexical overlap and filename matches;
3. selects at most four chunks within a small character budget;
4. packages only those excerpts as a current-turn source context message.

Generic requests such as “summarise the attached document” fall back to the first chunks rather than returning nothing.

The source context is **not added to rolling chat history**. A later question performs retrieval again against the local files, so documents are not silently re-sent on every turn.

## Room flow

- If the host asks, retrieval happens in the host browser and the selected excerpt bundle goes directly into prompt assembly.
- If a guest asks, retrieval happens in the guest browser. Only the bounded excerpt bundle travels to the host with `ai-ask`.
- The host sanitises the bundle again before using it.
- Diagnostics record source token/chunk counts, not filenames or source text.

## Context budget

The room currently uses a 2,048-token runtime context even when the underlying model supports more. Sources therefore keeps a deliberately tight excerpt budget. Longer-context experiments should increase runtime context only after memory/performance behaviour has been measured across ordinary devices.

## Next

- local PDF parsing;
- better local retrieval/ranking when evidence shows lexical selection is insufficient;
- visible per-answer source chips/excerpts so people can inspect what the model actually saw;
- optional source-selection controls;
- test whether larger model context is worth the increased KV-cache cost across a room.
