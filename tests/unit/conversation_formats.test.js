import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { chatRuntimeForStyle } from "../../engine/model-adapters.js";
import { buildConversationPrompt } from "../../field-station/conversation.js";

function llamaTokenizer() {
  return {
    vocab: {
      "<|begin_of_text|>": 128000,
      "<|end_of_text|>": 128001,
      "<|start_header_id|>": 128006,
      "<|end_header_id|>": 128007,
      "<|eot_id|>": 128009,
    },
    encode(text) { return [...text].map((ch) => ch.codePointAt(0)); },
  };
}

Deno.test("rolling Llama 3 conversation emits BOS exactly once", () => {
  const tok = llamaTokenizer();
  const chat = chatRuntimeForStyle("llama3-header", tok);
  const prompt = buildConversationPrompt({
    tok,
    vocab: tok.vocab,
    chat,
    history: [
      { user: "first", assistant: "answer one" },
      { user: "second", assistant: "answer two" },
    ],
    currentText: "continue",
    maxSeq: 2048,
  });

  assertEquals(prompt.usedTurns, 2);
  assertEquals(prompt.ids[0], 128000);
  assertEquals(prompt.ids.filter((id) => id === 128000).length, 1);
  assertEquals(prompt.ids.filter((id) => id === 128009).length, 5);
  assertEquals(prompt.ids.slice(-chat.assistantPrefix().length), chat.assistantPrefix());
});

Deno.test("conversation prefix participates in the context budget", () => {
  const tok = llamaTokenizer();
  const chat = chatRuntimeForStyle("llama3-header", tok);
  const prompt = buildConversationPrompt({
    tok,
    vocab: tok.vocab,
    chat,
    history: [],
    currentText: "x".repeat(20),
    maxSeq: 40,
    minRoom: 8,
  });

  assertEquals(prompt.ids[0], 128000);
  assertEquals(prompt.reserveTokens, 40 - prompt.ids.length);
});
