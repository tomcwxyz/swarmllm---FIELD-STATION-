// FIELD STATION room conversation helpers.
// Conversation text stays in host memory. These helpers assemble a rolling chat prompt;
// source excerpts are injected for the current question only and are not persisted in history.

export const FIELD_CONTEXT_RESERVE = 512;
export const FIELD_LATEST_TURN_RESERVE = 256;

function legacyChatRuntime(tok, vocab) {
  const imStart = vocab["<|im_start|>"];
  const imEnd = vocab["<|im_end|>"];
  if (!Number.isInteger(imStart) || !Number.isInteger(imEnd)) throw new Error("chat special tokens are missing");
  const think = vocab["<think>"];
  const thinkEnd = vocab["</think>"];
  return {
    conversationPrefix() { return []; },
    encodeMessage(role, content) { return [imStart, ...tok.encode(`${role}\n${content ?? ""}`), imEnd, ...tok.encode("\n")]; },
    assistantPrefix() {
      const ids = [imStart, ...tok.encode("assistant\n")];
      if (Number.isInteger(think) && Number.isInteger(thinkEnd)) ids.push(think, ...tok.encode("\n\n"), thinkEnd, ...tok.encode("\n\n"));
      return ids;
    },
  };
}

/**
 * Build a coherent rolling prompt from complete previous user/assistant turns.
 * `sourceContext`, when present, is a current-turn system message containing only
 * the locally retrieved excerpts selected by the asker's browser. It deliberately
 * does not enter `history`, so attached documents are not silently re-sent forever.
 */
export function buildConversationPrompt({
  tok, vocab, chat = null, history = [], currentText, sourceContext = "",
  maxSeq, minRoom = 32, reserve = FIELD_CONTEXT_RESERVE,
}) {
  const format = chat || legacyChatRuntime(tok, vocab);
  if (typeof format.encodeMessage !== "function" || typeof format.assistantPrefix !== "function")
    throw new Error("model chat formatter is incomplete");

  const prefix = typeof format.conversationPrefix === "function" ? format.conversationPrefix() : [];
  const sourceIds = sourceContext ? format.encodeMessage("system", sourceContext) : [];
  const current = [...sourceIds, ...format.encodeMessage("user", currentText), ...format.assistantPrefix()];
  const hardPromptLimit = maxSeq - minRoom;
  if (prefix.length + current.length > hardPromptLimit) {
    const ids = [...prefix, ...current];
    return {
      ids, usedTurns: 0, droppedTurns: history.length,
      reserveTokens: Math.max(0, maxSeq - ids.length), sourceTokens: sourceIds.length,
    };
  }

  const baseLength = prefix.length + current.length;
  const preferredPromptLimit = Math.max(baseLength, maxSeq - Math.max(minRoom, reserve));
  const latestTurnLimit = Math.max(preferredPromptLimit, maxSeq - Math.max(minRoom, FIELD_LATEST_TURN_RESERVE));
  const selected = [];
  let promptLength = baseLength;
  let usedTurns = 0;

  for (let i = history.length - 1; i >= 0; i--) {
    const turn = history[i];
    const turnIds = [
      ...format.encodeMessage("user", turn.user || ""),
      ...format.encodeMessage("assistant", turn.assistant || ""),
    ];
    const limit = usedTurns === 0 ? latestTurnLimit : preferredPromptLimit;
    if (promptLength + turnIds.length > limit) break;
    selected.unshift(turnIds);
    promptLength += turnIds.length;
    usedTurns++;
  }

  return {
    ids: [...prefix, ...selected.flat(), ...current],
    usedTurns,
    droppedTurns: Math.max(0, history.length - usedTurns),
    reserveTokens: maxSeq - promptLength,
    sourceTokens: sourceIds.length,
  };
}

export function generationFinishReason({ count, maxNew, maxSeq, promptTokens, contextCapped }) {
  if (contextCapped || promptTokens + count >= maxSeq - 1) return "context full";
  if (count >= maxNew) return "output limit";
  return "end of turn";
}
