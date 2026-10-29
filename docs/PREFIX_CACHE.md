# Prefix cache (opt-in)

Cross-turn KV prefix reuse for chat, behind `AURORA_PREFIX_CACHE=1` (default
**off**). Aurora owns it; Grio only passes a session id; Candle is untouched.

## Behaviour

- Toggle: `AURORA_PREFIX_CACHE=1` enables it. Without it, the normal path runs
  (zero overhead).
- Supported backends: **Gemma** (`QuantizedGemma3`) for now. For any other
  backend the engine prints once, in English,
  `[aurora] prefix cache: backend not supported, falling back to the normal path`
  and uses the normal path (no crash).
- Algorithm: tokenize the full template prompt, compute the longest common
  token prefix with the session, truncate the model KV to that length, and
  prefill only the suffix (`index_pos = lcp`).
- The session stores the **raw** generated text (including the `thought`
  channel) so the KV aligns with the next prompt; the UI still shows the
  cleaned answer.
- Console trace per request:
  `[aurora] prefix cache: reusing N/M prompt tokens`.
- Reset: the demo app calls `reset_session("grio-default")` on the **Effacer**
  button; the next turn starts from zero.

## Measured (Gemma 4 31B Q4_K_XL, R9700)

Two-turn LRU test (`Q1` long answer, `Q2` follow-up referencing point 5):

- Q1: `reusing 0/592` → 83.9 s.
- Q2: `reusing 1392/1664` tokens (~84 %) → 64.7 s, correct follow-up answer.

That run had the first prompt sent twice, so the history was duplicated and
the reuse capped. A clean `Q1 → A1 → Q2` session reuses the previous turn
almost entirely (only the new user message + template are prefilled).

## Limits

- Only one KV is resident (the model's own cache); switching models or
  `Effacer` resets it.
- VRAM grows with the conversation: 31.8 GB observed on the 31B at ~1.6k-token
  context. Clear the chat to release it.
- Rotation/sliding-window caches (other models) are not handled yet; the
  backend returns unsupported and the engine falls back.
- The `tok/s` shown by Grio includes prefill; it is not a pure decode rate.

## Files

- `src/models/text.rs` — `generate_chat_cached`, session map, LCP.
- `src/models/quantized_gemma3.rs` — `prefix_cache_len` / `truncate_prefix_cache`
  / `reset_prefix_cache`.
- `src/bin/grio_qwen_reasoning.rs` — uses the cached path and resets on
  **Effacer**.
