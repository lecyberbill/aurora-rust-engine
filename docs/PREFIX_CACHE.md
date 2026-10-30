# Prefix cache (opt-in)

Cross-turn KV prefix reuse for chat, behind `AURORA_PREFIX_CACHE=1` (default
**off**). Aurora owns it; Grio only passes a session id; Candle is untouched.

## Behaviour

- Toggle: `AURORA_PREFIX_CACHE=1` enables it. Without it, the normal path runs
  (zero overhead).
- Supported backends: **Gemma 3** (`QuantizedGemma3`) and **Qwen 3.5/3.8** (`QuantizedQwen35`). For any other
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

Clean two-turn LRU test (Q1 = short prompt + long answer, Q2 = follow-up):

- Q1: `reusing 0/39` â€” prompt 39 tokens, prefill 253 ms, total 71.1 s.
- Q2: `reusing 408/1110` â€” prefill 2454 ms (~700 new tokens), total 53.0 s.

The reuse stops **inside the assistant answer**: the generated tokens are
decoded to text and re-tokenized for the next prompt, and BPE round-trips are
not always exact (the divergence appears ~370 answer tokens in). A token-level
prompt builder (append the stored generated token ids instead of re-encoding
the text) would be needed for full reuse.

The dominant cost is still **decode** (50â€“70 s); the saved prefill is ~1.5 s on
Q2. VRAM reached **33.6 GB** at ~1.1k-token context â€” clear the chat to release
the KV.

A duplicated-history run (Q1 sent twice) showed `1392/1664` reused, which
confirms the mechanism; the cap above is the text re-tokenization, not the
cache.

## Limits

- Only one KV is resident (the model's own cache); switching models or
  `Effacer` resets it.
- VRAM grows with the conversation: 31.8 GB observed on the 31B at ~1.6k-token
  context. Clear the chat to release it.
- The Grio app caps the history in characters (`AURORA_CHAT_MAX_CHARS`, default
  4000 for 31B, 8000 for 27B, 12000 for 12B/other, from the model path): the
  oldest turns are dropped to bound the prompt/KV.
  Lower it (e.g. `AURORA_CHAT_MAX_CHARS=4000`) on the 31B to keep more VRAM
  headroom; raising it lets longer follow-ups through. The console prints the
  kept message count and character budget.
- Reuse is exact with a stable history; the session resets when the cap drops a
  message (message-count check), so the two levers do not compose on long
  chats.
- Rotation/sliding-window caches (other models) are not handled yet; the
  backend returns unsupported and the engine falls back.
- The `tok/s` shown by Grio includes prefill; it is not a pure decode rate.
- Gemma chat now samples with `LogitsProcessor` (temperature, top-p 0.95, top-k
  50, repetition 1.1). It used to be greedy, which ignored the slider and could
  loop (e.g. repeating "L'année 1789."), inflating the answer and OOMing the
  31B. If you still see loops, lower temperature or raise the repetition penalty.

## Files

- `src/models/text.rs` â€” `generate_chat_cached`, session map, LCP.
- `src/models/quantized_gemma3.rs` â€” `prefix_cache_len` / `truncate_prefix_cache`
  / `reset_prefix_cache`.
- `src/bin/grio_qwen_reasoning.rs` â€” uses the cached path and resets on
  **Effacer**.
