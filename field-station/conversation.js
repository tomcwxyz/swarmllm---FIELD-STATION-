// FIELD STATION room conversation helpers.
// Conversation text stays in host memory. These helpers only assemble a rolling chat prompt;
// they do not persist or transmit transcript content beyond the existing room chat flow.

export const FIELD_CONTEXT_RESERVE = 512;
export const FIELD_LATEST_TURN_RESERVE = 256;

function messageIds(tok, vocab, role, content) {
  const imStart = vocab["<|im_start|>"];
  const imEnd = vocab["<|im_end|>"];
  return [
    imStart,
    ...tok.encode(`${role}\n${content}`),
    imEnd,
    ...tok.encode("\n"),
  ];
}

/**
 * Build a coherent rolling prompt from complete previous user/assistant turns.
 * The immediately preceding turn gets priority so short follow-ups such as "continue"
 * remain meaningful. Older turns fall away first to retain useful answer room.
 */
export function buildConversationPrompt({ tok, vocab, history = [], currentText, maxSeq, minRoom = 32, reserve = FIELD_CONTEXT_RESERVE }) {
  const imStart = vocab["<|im_start|>"];
  const imEnd = vocab["<|im_end|>"];
  if (!Number.isInteger(imStart) || !Number.isInteger(imEnd)) throw new Error("chat special tokens are missing");

  const assistantPrefix = [imStart, ...tok.encode("assistant\n")];
  const thinkingClosure = vocab["<think>"] !== undefined && vocab["</think>"] !== undefined
    ? [vocab["<think>"], ...tok.encode("\n\n"), vocab["</think>"], ...tok.encode("\n\n")]
    : [];
  const current = [...messageIds(tok, vocab, "user", currentText), ...assistantPrefix, ...thinkingClosure];
  const hardPromptLimit = maxSeq - minRoom;
  if (current.length > hardPromptLimit) {
    return { ids: current, usedTurns: 0, droppedTurns: history.length, reserveTokens: Math.max(0, maxSeq - current.length) };
  }

  // Normally reserve 512 tokens for the answer. For the immediately preceding turn, relax
  // that to 256 tokens when necessary: preserving what the user is referring to is more useful
  // than forgetting it merely to guarantee a longer response. Older history uses the full reserve.
  const preferredPromptLimit = Math.max(current.length, maxSeq - Math.max(minRoom, reserve));
  const latestTurnLimit = Math.max(preferredPromptLimit, maxSeq - Math.max(minRoom, FIELD_LATEST_TURN_RESERVE));
  const selected = [];
  let promptLength = current.length;
  let usedTurns = 0;

  for (let i = history.length - 1; i >= 0; i--) {
    const turn = history[i];
    const turnIds = [
      ...messageIds(tok, vocab, "user", turn.user || ""),
      ...messageIds(tok, vocab, "assistant", turn.assistant || ""),
    ];
    const limit = usedTurns === 0 ? latestTurnLimit : preferredPromptLimit;
    if (promptLength + turnIds.length > limit) break;
    selected.unshift(turnIds);
    promptLength += turnIds.length;
    usedTurns++;
  }

  return {
    ids: [...selected.flat(), ...current],
    usedTurns,
    droppedTurns: Math.max(0, history.length - usedTurns),
    reserveTokens: maxSeq - promptLength,
  };
}

export function generationFinishReason({ count, maxNew, maxSeq, promptTokens, contextCapped }) {
  if (contextCapped || promptTokens + count >= maxSeq - 1) return "context full";
  if (count >= maxNew) return "output limit";
  return "end of turn";
}
