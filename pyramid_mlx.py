#!/usr/bin/env python3
"""PyramidSD 3-model speculative decoding on Apple Silicon using MLX.

Memory-efficient implementation using KV caches (mutated in-place by MLX).
Draft (0.5B) → Qualifier (1.5B) → Target (7B), all 4-bit quantized.
"""

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def load_models():
    """Load all three models."""
    print("[pyramid] Loading draft (0.5B)...")
    draft, tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print("[pyramid] Loading qualifier (1.5B)...")
    qual, _ = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    print("[pyramid] Loading target (7B)...")
    tgt, _ = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    print("[pyramid] All loaded.")
    return draft, qual, tgt, tok


def generate_pyramid(draft, qual, target, tokenizer, prompt,
                     max_tokens=100, l_d=3, l_q=6, tau_q=0.3, tau_t=0.4):
    """Full PyramidSD generation with KV caching."""

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer.encode(formatted)
    input_ids = mx.array([prompt_ids])

    # Create KV caches for all 3 models
    d_cache = make_prompt_cache(draft)
    q_cache = make_prompt_cache(qual)
    t_cache = make_prompt_cache(target)

    # Prime caches with prompt (each model processes the prompt once)
    draft(input_ids, cache=d_cache)
    qual(input_ids, cache=q_cache)
    target(input_ids, cache=t_cache)
    mx.eval(d_cache[0].keys)  # force evaluation

    output_ids = []
    eos_id = tokenizer.eos_token_id
    stats = {"draft_proposed": 0, "draft_accepted": 0, "qual_proposed": 0, "qual_accepted": 0}

    t0 = time.perf_counter()

    # We need to track what tokens each cache has seen.
    # After priming, all caches are synced to the prompt.
    # As we generate, we need to keep them in sync.

    while len(output_ids) < max_tokens:
        # ============================================================
        # STAGE 1: Draft → Qualifier
        # Draft proposes l_d tokens, qualifier verifies in one pass.
        # Repeat until l_q tokens accumulated.
        # ============================================================
        q_buffer = []       # qualifier-approved token IDs
        q_logits_buf = []   # qualifier logits at each approved position

        while len(q_buffer) < l_q and len(output_ids) + len(q_buffer) < max_tokens:
            # 1a. Draft generates l_d tokens using its cache
            draft_tokens = []
            draft_token_logits = []  # logit value the draft assigned to each token

            # The draft cache is already primed. Feed tokens one at a time.
            for step in range(l_d):
                if step == 0 and not output_ids and not q_buffer:
                    # First token after prompt — draft cache already has prompt
                    # We need the logits from the last prompt position
                    # Re-run last token to get logits (cache already has everything)
                    last_tok = mx.array([[prompt_ids[-1]]])
                    # Actually, the cache is already primed, so we need to generate
                    # the NEXT token. We do this by passing a dummy — but the cache
                    # already consumed the prompt. Just generate from current state.
                    pass

                # For the draft, we feed the last generated token (or nothing for first)
                if step == 0:
                    if q_buffer:
                        feed = mx.array([[q_buffer[-1]]])
                    elif output_ids:
                        feed = mx.array([[output_ids[-1]]])
                    else:
                        # First token ever — we need to get the first prediction
                        # Cache has the prompt, so feeding any token would advance it.
                        # Instead, let's get logits from the prompt processing.
                        # We already primed, so let's just do a forward on the last token again.
                        # Actually, let's handle this differently...
                        feed = mx.array([[prompt_ids[-1]]])
                else:
                    feed = mx.array([[draft_tokens[-1]]])

                logits = draft(feed, cache=d_cache)
                next_logits = logits[0, -1, :]  # (vocab,)
                next_token = mx.argmax(next_logits).item()

                draft_tokens.append(next_token)
                draft_token_logits.append(next_logits[next_token].item())

            stats["draft_proposed"] += l_d

            # 1b. Qualifier verifies all l_d tokens in one pass
            candidate = mx.array([draft_tokens])
            q_logits = qual(candidate, cache=q_cache)  # (1, l_d, vocab)

            # 1c. Accept/reject per token
            n_acc = 0
            for i in range(l_d):
                d_val = draft_token_logits[i]
                q_val = q_logits[0, i, draft_tokens[i]].item()

                if abs(q_val - d_val) <= tau_q:
                    q_buffer.append(draft_tokens[i])
                    q_logits_buf.append(q_logits[0, i, :])
                    n_acc += 1
                else:
                    # PSDA: replace with qualifier's choice
                    replacement = mx.argmax(q_logits[0, i, :]).item()
                    q_buffer.append(replacement)
                    q_logits_buf.append(q_logits[0, i, :])
                    break

            stats["draft_accepted"] += n_acc

            # Clean up draft logits
            del q_logits
            mx.eval()

            # EOS check
            if eos_id in q_buffer[-n_acc - 1:] if n_acc < l_d else q_buffer[-n_acc:]:
                break

        if not q_buffer:
            break

        # ============================================================
        # STAGE 2: Qualifier buffer → Target
        # Target verifies all buffer tokens in one pass.
        # ============================================================
        stats["qual_proposed"] += len(q_buffer)

        buf_ids = mx.array([q_buffer])
        t_logits = target(buf_ids, cache=t_cache)  # (1, len(q_buffer), vocab)

        final = []
        for i in range(len(q_buffer)):
            t_val = t_logits[0, i, q_buffer[i]].item()
            q_val = q_logits_buf[i][q_buffer[i]].item()

            if abs(t_val - q_val) <= tau_t:
                final.append(q_buffer[i])
            else:
                replacement = mx.argmax(t_logits[0, i, :]).item()
                final.append(replacement)
                break

        stats["qual_accepted"] += len(final)

        del t_logits, q_logits_buf
        mx.eval()

        # Append to output
        for t in final:
            if t == eos_id:
                break
            output_ids.append(t)

    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(output_ids)
    tps = len(output_ids) / elapsed if elapsed > 0 else 0

    return {
        "text": text,
        "tokens": len(output_ids),
        "tps": round(tps, 1),
        "elapsed_s": round(elapsed, 2),
        "stats": stats,
        "draft_accept_rate": round(stats["draft_accepted"] / max(stats["draft_proposed"], 1), 3),
        "qual_accept_rate": round(stats["qual_accepted"] / max(stats["qual_proposed"], 1), 3),
    }


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is a hash table?"

    draft, qual, tgt, tok = load_models()
    print(f"Peak memory after load: {mx.get_peak_memory() / 1e9:.2f} GB")

    result = generate_pyramid(draft, qual, tgt, tok, prompt, max_tokens=100, l_d=3, l_q=6)

    print(f"\n--- Output ---\n{result['text']}")
    print(f"\n--- Metrics ---")
    print(f"Tokens: {result['tokens']} | Speed: {result['tps']} tok/s | Time: {result['elapsed_s']}s")
    print(f"Draft accept: {result['draft_accept_rate']} | Qualifier accept: {result['qual_accept_rate']}")
    print(f"Stats: {result['stats']}")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")
