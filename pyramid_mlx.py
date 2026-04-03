#!/usr/bin/env python3
"""PyramidSD 3-model speculative decoding on Apple Silicon using MLX.

Simple correct implementation: draft uses KV cache for sequential generation,
verifiers do full-context forward passes (no cache) for correctness.
This is slower than a fully-cached version but guaranteed correct.

Draft (0.5B) → Qualifier (1.5B) → Target (7B), all 4-bit quantized.
"""

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.nn as nn


def load_models():
    print("[pyramid] Loading draft (0.5B)...")
    draft, tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print("[pyramid] Loading qualifier (1.5B)...")
    qual, _ = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    print("[pyramid] Loading target (7B)...")
    tgt, _ = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    print("[pyramid] All loaded.")
    return draft, qual, tgt, tok


def generate_baseline(model, tokenizer, prompt, max_tokens=100):
    """Simple autoregressive baseline for speed comparison."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(formatted)
    input_ids = mx.array([ids])

    cache = make_prompt_cache(model)
    logits = model(input_ids, cache=cache)
    mx.eval(cache[0].keys)

    output = []
    eos = tokenizer.eos_token_id
    t0 = time.perf_counter()

    for _ in range(max_tokens):
        next_tok = mx.argmax(logits[0, -1, :]).item()
        if next_tok == eos:
            break
        output.append(next_tok)
        logits = model(mx.array([[next_tok]]), cache=cache)

    elapsed = time.perf_counter() - t0
    return {
        "text": tokenizer.decode(output),
        "tokens": len(output),
        "tps": round(len(output) / elapsed, 1) if elapsed > 0 else 0,
        "elapsed_s": round(elapsed, 2),
    }


def generate_pyramid(draft, qual, target, tokenizer, prompt,
                     max_tokens=100, l_d=3, l_q=6, tau_q=0.3, tau_t=0.4):
    """PyramidSD with correct cache handling.
    
    Strategy:
    - Draft: uses KV cache for fast sequential token generation
    - Qualifier: full-context forward pass (no cache, guarantees correctness)
    - Target: full-context forward pass (no cache, guarantees correctness)
    
    On rejection: draft cache is rebuilt from scratch (simple & correct).
    """

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer.encode(formatted)

    output_ids = []
    eos_id = tokenizer.eos_token_id
    stats = {"d_proposed": 0, "d_accepted": 0, "q_proposed": 0, "q_accepted": 0, "iters": 0}
    t0 = time.perf_counter()

    while len(output_ids) < max_tokens:
        stats["iters"] += 1

        # Build the committed context: prompt + all accepted output so far
        context_ids = prompt_ids + output_ids

        # ================================================================
        # STAGE 1: Draft → Qualifier
        # ================================================================
        q_buffer = []
        q_logits_buf = []

        # Rebuild draft cache from committed context
        d_cache = make_prompt_cache(draft)
        ctx_tensor = mx.array([context_ids])
        d_logits = draft(ctx_tensor, cache=d_cache)
        mx.eval(d_cache[0].keys)
        last_logits = d_logits[0, -1, :]  # predicts first new token

        while len(q_buffer) < l_q and len(output_ids) + len(q_buffer) < max_tokens:

            # 1a. Draft generates l_d tokens using its cache
            draft_tokens = []
            draft_logit_vals = []
            draft_logits_vecs = []  # full logit vectors for softmax

            for step in range(l_d):
                if step == 0:
                    logits_vec = last_logits
                else:
                    sl = draft(mx.array([[draft_tokens[-1]]]), cache=d_cache)
                    logits_vec = sl[0, -1, :]

                next_tok = mx.argmax(logits_vec).item()
                draft_tokens.append(next_tok)
                draft_logit_vals.append(logits_vec[next_tok].item())
                draft_logits_vecs.append(logits_vec)

            # Feed last draft token to keep cache consistent
            draft(mx.array([[draft_tokens[-1]]]), cache=d_cache)

            stats["d_proposed"] += l_d

            # 1b. Qualifier: full context + q_buffer + candidates in ONE pass
            full_seq = context_ids + q_buffer + draft_tokens
            full_tensor = mx.array([full_seq])
            q_all_logits = qual(full_tensor)  # no cache — full context
            mx.eval(q_all_logits)

            # Qualifier logits for candidates start at position len(context_ids + q_buffer) - 1
            # because logit at position i predicts token at position i+1
            base_pos = len(context_ids) + len(q_buffer) - 1

            # 1c. Accept/reject using softmax probabilities
            n_acc = 0
            for i in range(l_d):
                d_probs = mx.softmax(draft_logits_vecs[i])
                q_probs = mx.softmax(q_all_logits[0, base_pos + i, :])
                d_val = d_probs[draft_tokens[i]].item()
                q_val = q_probs[draft_tokens[i]].item()

                if abs(q_val - d_val) <= tau_q:
                    q_buffer.append(draft_tokens[i])
                    q_logits_buf.append(q_probs)
                    n_acc += 1
                else:
                    replacement = mx.argmax(q_probs).item()
                    q_buffer.append(replacement)
                    q_logits_buf.append(q_probs)
                    break

            stats["d_accepted"] += n_acc

            del q_all_logits
            mx.eval()

            # Rebuild draft cache to: context + accepted q_buffer
            d_cache = make_prompt_cache(draft)
            new_ctx = mx.array([context_ids + q_buffer])
            d_logits = draft(new_ctx, cache=d_cache)
            mx.eval(d_cache[0].keys)
            last_logits = d_logits[0, -1, :]

            if eos_id is not None and eos_id in q_buffer:
                break

        if not q_buffer:
            break

        # ================================================================
        # STAGE 2: Qualifier buffer → Target
        # Full context forward pass — no cache.
        # Compare P_T vs P_Q (paper's key insight).
        # ================================================================
        stats["q_proposed"] += len(q_buffer)

        full_seq = context_ids + q_buffer
        full_tensor = mx.array([full_seq])
        t_all_logits = target(full_tensor)  # no cache — full context
        mx.eval(t_all_logits)

        base_pos = len(context_ids) - 1

        final = []
        for i in range(len(q_buffer)):
            t_probs = mx.softmax(t_all_logits[0, base_pos + i, :])
            t_val = t_probs[q_buffer[i]].item()
            q_val = q_logits_buf[i][q_buffer[i]].item()

            if abs(t_val - q_val) <= tau_t:
                final.append(q_buffer[i])
            else:
                replacement = mx.argmax(t_probs).item()
                final.append(replacement)
                break

        stats["q_accepted"] += len(final)

        del t_all_logits, q_logits_buf
        mx.eval()

        # Append accepted tokens
        eos_hit = False
        for t in final:
            if t == eos_id:
                eos_hit = True
                break
            output_ids.append(t)

        if eos_hit:
            break

    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(output_ids)
    tps = len(output_ids) / elapsed if elapsed > 0 else 0

    return {
        "text": text,
        "tokens": len(output_ids),
        "tps": round(tps, 1),
        "elapsed_s": round(elapsed, 2),
        "stats": stats,
        "draft_accept_rate": round(stats["d_accepted"] / max(stats["d_proposed"], 1), 3),
        "qual_accept_rate": round(stats["q_accepted"] / max(stats["q_proposed"], 1), 3),
    }


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is a hash table?"

    draft, qual, tgt, tok = load_models()
    print(f"Peak memory after load: {mx.get_peak_memory() / 1e9:.2f} GB")

    # Baseline
    print("\n=== BASELINE (7B only) ===")
    base = generate_baseline(tgt, tok, prompt, max_tokens=100)
    print(f"Output: {base['text'][:200]}...")
    print(f"Speed: {base['tps']} tok/s | Time: {base['elapsed_s']}s")

    # PyramidSD
    print("\n=== PYRAMID SD (0.5B → 1.5B → 7B) ===")
    result = generate_pyramid(draft, qual, tgt, tok, prompt, max_tokens=100, l_d=3, l_q=6)
    print(f"Output: {result['text'][:200]}...")
    print(f"\n--- Metrics ---")
    print(f"Tokens: {result['tokens']} | Speed: {result['tps']} tok/s | Time: {result['elapsed_s']}s")
    print(f"Draft accept: {result['draft_accept_rate']} | Qualifier accept: {result['qual_accept_rate']}")
    print(f"Stats: {result['stats']}")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")

    # Speedup
    if base['tps'] > 0:
        print(f"\nSpeedup: {result['tps'] / base['tps']:.2f}x vs baseline")
