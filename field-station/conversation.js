// FIELD STATION room conversation helpers.
// Conversation text stays in host memory. These helpers only assemble a rolling chat prompt;
// they do not persist or transmit transcript content beyond the existing room chat flow.

export const FIELD_CONTEXT_RESERVE = 512;
export const FIELD_LATEST_TURN_RESERVE = 256;

// Backwards-compatible ChatML formatter for callers that have not moved onto the
// architecture adapter yet. The live FIELD STATION room now supplies `chat` explicitly.
function legacyChatRuntime(tok, vocab) {
  const imStart = vocab["<|im_start|>"];
  const imEnd = vocab["<|im_end|>"];
  if (!Number.isInteger(imStart) || !Number.isInteger(imEnd)) throw new Error("chat special tokens are missing");
  const think = vocab["<think>"];
  const thinkEnd = vocab["</think>"];
  return {
    encodeMessage(role, content) {
      return [imStart, ...tok.encode(`${role}\n${content ?? ""}`), imEnd, ...tok.encode("\n")];
    },
    assistantPrefix() {
      const ids = [imStart, ...tok.encode("assistant\n")];
      if (Number.isInteger(think) && Number.isInteger(thinkEnd))
        ids.push(think, ...tok.encode("\n\n"), thinkEnd, ...tok.encode("\n\n"));
      return ids;
    },
  };
}

/**
 * Build a coherent rolling prompt from complete previous user/assistant turns.
 * The immediately preceding turn gets priority so short follow-ups such as "continue"
 * remain meaningful. Older turns fall away first to retain useful answer room.
 *
 * `chat` is the architecture adapter's token-level conversation formatter. Keeping
 * context selection here but message representation in the adapter means Llama/Gemma
 * can retain exactly the same rolling-history policy without pretending to speak ChatML.
 */
export function buildConversationPrompt({ tok, vocab, chat = null, history = [], currentText, maxSeq, minRoom = 32, reserve = FIELD_CONTEXT_RESERVE }) {
  const format = chat || legacyChatRuntime(tok, vocab);
  if (typeof format.encodeMessage !== "function" || typeof format.assistantPrefix !== "function")
    throw new Error("model chat formatter is incomplete");

  const current = [...format.encodeMessage("user", currentText), ...format.assistantPrefix()];
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
