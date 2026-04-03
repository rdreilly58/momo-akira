#!/usr/bin/env python3
"""PyramidSD 3-model speculative decoding on Apple Silicon using MLX.

Full KV-cached implementation. All three models maintain a rolling cache;
rollback is done via trim(). q_next_logits and t_next_logits mirror
draft_next_logits so every verifier comparison uses the distribution BEFORE
the token being assessed (not after), eliminating the off-by-one bug.
"""
import time
import random
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def load_models():
    print("Loading all models...")
    d, tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    q, _   = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    tgt, _ = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    return d, q, tgt, tok


def trim_cache(cache, n):
    if n > 0:
        for layer in cache:
            layer.trim(n)


def _sample_categorical(probs):
    """Sample a token index from a probability distribution via gumbel-max trick."""
    log_probs = mx.log(mx.maximum(probs, 1e-10))
    uniform = mx.random.uniform(shape=log_probs.shape)
    gumbel = -mx.log(-mx.log(mx.maximum(uniform, 1e-10)))
    return mx.argmax(log_probs + gumbel).item()


def generate_baseline(t_model, tokenizer, prompt, max_tokens=100, verbose=True):
    """Autoregressive generation with the 7B target model (greedy)."""
    prompt_ids = mx.array([tokenizer.encode(prompt)])
    cache = make_prompt_cache(t_model)
    logits = t_model(prompt_ids, cache=cache)
    mx.eval(cache[0].keys)

    cur_logits = logits[0, -1, :]
    output_ids = []
    t0 = time.perf_counter()

    while len(output_ids) < max_tokens:
        tok_id = mx.argmax(cur_logits).item()
        output_ids.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
        out = t_model(mx.array([[tok_id]]), cache=cache)
        cur_logits = out[0, -1, :]
        mx.eval(cur_logits)

    elapsed = time.perf_counter() - t0
    n_tok = len(output_ids)
    tok_s = n_tok / elapsed if elapsed > 0 else 0
    if verbose:
        print(f"\n[Baseline] {n_tok} tokens | {elapsed:.1f}s | {tok_s:.1f} tok/s")
    return {
        "text": tokenizer.decode(output_ids),
        "tokens": output_ids,
        "tok_s": tok_s,
        "n_tokens": n_tok,
        "elapsed": elapsed,
    }


def generate_pyramid(d_model, q_model, t_model, tokenizer, prompt,
                     max_tokens=100, l_d=3, l_q=6, tau_q=0.3, tau_t=0.4,
                     mode="fuzzy", verbose=True):
    """PyramidSD speculative decoding.

    mode="fuzzy"    — original fuzzy acceptance (tau_q / tau_t thresholds).
    mode="lossless" — Leviathan et al. 2023 rejection sampling; output
                      distribution is identical to baseline greedy; tau_q /
                      tau_t are ignored.
    """
    prompt_ids = mx.array([tokenizer.encode(prompt)])

    # ── Prime all three caches with the prompt ────────────────────────────────
    d_cache = make_prompt_cache(d_model)
    q_cache = make_prompt_cache(q_model)
    t_cache = make_prompt_cache(t_model)

    d_prime = d_model(prompt_ids, cache=d_cache)
    q_prime = q_model(prompt_ids, cache=q_cache)
    t_prime = t_model(prompt_ids, cache=t_cache)
    mx.eval(d_cache[0].keys, q_cache[0].keys, t_cache[0].keys)

    # Each *_next_logits holds the softmax distribution for the *next* token to
    # be generated, computed from the model's output after seeing the full
    # committed context. This is the "position 0" logit that corrects the
    # off-by-one: model output[i] predicts position i+1, so to assess position
    # i we need output[i-1], and for i=0 we need the pre-batch logit kept here.
    draft_next_logits = d_prime[0, -1, :]
    q_next_logits     = mx.softmax(q_prime[0, -1, :])
    t_next_logits     = mx.softmax(t_prime[0, -1, :])
    del d_prime, q_prime, t_prime
    mx.eval()

    d_proposed = d_accepted = q_proposed = q_accepted = 0
    output_ids = []
    t0 = time.perf_counter()

    while len(output_ids) < max_tokens:
        # Cache invariant: all three caches at the same committed offset.
        assert d_cache[0].offset == q_cache[0].offset == t_cache[0].offset, (
            f"Cache offset mismatch: d={d_cache[0].offset} "
            f"q={q_cache[0].offset} t={t_cache[0].offset}"
        )

        q_buffer    = []   # tokens qualifier accepted (or replaced)
        # q_probs_buf[i] = qualifier's softmax distribution BEFORE seeing
        # q_buffer[i], so Stage 2 can compare it against the target's
        # distribution at the same position.
        q_probs_buf = []

        # ── Stage 1: Draft → Qualifier ────────────────────────────────────────
        while len(q_buffer) < l_q and len(output_ids) + len(q_buffer) < max_tokens:
            draft_toks    = []
            d_logits_vecs = []
            cur_logits    = draft_next_logits

            # Autoregressive draft: l_d single-token passes, cache advances l_d.
            for _ in range(l_d):
                tok_id = mx.argmax(cur_logits).item()
                draft_toks.append(tok_id)
                d_logits_vecs.append(cur_logits)
                out = d_model(mx.array([[tok_id]]), cache=d_cache)
                cur_logits = out[0, -1, :]
            # d_cache: +l_d  |  cur_logits: logit for position after last draft tok
            d_proposed += l_d

            # Qualifier verifies all l_d candidates in one batched pass.
            # q_logits[0, i, :] = qualifier's distribution AFTER seeing draft_toks[i],
            # i.e., it predicts position i+1. To assess draft_toks[i] we use
            # q_logits[0, i-1, :] for i>0, and q_next_logits for i=0.
            q_logits = q_model(mx.array([draft_toks]), cache=q_cache)
            mx.eval(q_logits)
            # q_cache: +l_d

            # Accept / reject each draft token.
            n_acc    = 0
            has_repl = False
            repl     = None
            for i in range(l_d):
                q_p = q_next_logits if i == 0 else mx.softmax(q_logits[0, i - 1, :])
                d_p = mx.softmax(d_logits_vecs[i])
                tok = draft_toks[i]

                if mode == "fuzzy":
                    stage1_accept = abs(q_p[tok].item() - d_p[tok].item()) <= tau_q
                else:  # lossless: Leviathan et al. rejection sampling
                    ratio = q_p[tok].item() / max(d_p[tok].item(), 1e-10)
                    stage1_accept = random.random() < min(1.0, ratio)

                if stage1_accept:
                    q_buffer.append(tok)
                    q_probs_buf.append(q_p)
                    n_acc += 1
                else:
                    if mode == "fuzzy":
                        repl = mx.argmax(q_p).item()
                    else:  # resample from max(0, q - p) normalised
                        v = min(q_p.shape[0], d_p.shape[0])
                        adjusted = mx.maximum(q_p[:v] - d_p[:v], 0)
                        s = adjusted.sum().item()
                        repl = _sample_categorical(adjusted / s if s > 1e-10 else q_p[:v])
                    q_buffer.append(repl)
                    q_probs_buf.append(q_p)
                    has_repl = True
                    break

            d_accepted  += n_acc
            tokens_kept  = n_acc + (1 if has_repl else 0)
            trim_amount  = l_d - tokens_kept

            if has_repl:
                # Roll back d_cache past rejected drafts and the wrong draft at
                # position n_acc (which q replaced with repl), then re-feed repl.
                trim_cache(d_cache, trim_amount + 1)
                out = d_model(mx.array([[repl]]), cache=d_cache)
                draft_next_logits = out[0, -1, :]

                # Same rollback for q_cache: trim back past the draft token at
                # n_acc (which is wrong — cache must hold repl, not draft_toks[n_acc]),
                # re-feed repl so q_cache content matches the committed sequence,
                # and capture q_next_logits for the next inner-loop iteration.
                trim_cache(q_cache, trim_amount + 1)
                out_q = q_model(mx.array([[repl]]), cache=q_cache)
                q_next_logits = mx.softmax(out_q[0, -1, :])
                del out_q
            else:
                # All l_d tokens accepted.
                trim_cache(d_cache, trim_amount)   # trim_amount == 0, no-op
                draft_next_logits = cur_logits

                trim_cache(q_cache, trim_amount)   # no-op
                # q_logits[0, l_d-1, :] is qualifier's distribution after the last
                # accepted token, i.e., its prediction for the next position.
                q_next_logits = mx.softmax(q_logits[0, l_d - 1, :])

            del q_logits, d_logits_vecs, out
            mx.eval()

        if not q_buffer:
            break

        # ── Stage 2: Qualifier → Target ───────────────────────────────────────
        # t_logits[0, i, :] = target's distribution AFTER seeing q_buffer[i],
        # predicts position i+1. To assess q_buffer[i] we use t_logits[0, i-1, :]
        # for i>0, and t_next_logits for i=0.
        q_proposed += len(q_buffer)
        t_logits = t_model(mx.array([q_buffer]), cache=t_cache)
        mx.eval(t_logits)
        # t_cache: +len(q_buffer)

        final  = []
        repl_t = None
        for i in range(len(q_buffer)):
            t_p = t_next_logits if i == 0 else mx.softmax(t_logits[0, i - 1, :])
            q_p = q_probs_buf[i]
            tok = q_buffer[i]

            if mode == "fuzzy":
                stage2_accept = abs(t_p[tok].item() - q_p[tok].item()) <= tau_t
            else:  # lossless
                ratio = t_p[tok].item() / max(q_p[tok].item(), 1e-10)
                stage2_accept = random.random() < min(1.0, ratio)

            if stage2_accept:
                final.append(tok)
            else:
                if mode == "fuzzy":
                    repl_t = mx.argmax(t_p).item()
                else:  # resample from max(0, t - q) normalised
                    v = min(t_p.shape[0], q_p.shape[0])
                    adjusted = mx.maximum(t_p[:v] - q_p[:v], 0)
                    s = adjusted.sum().item()
                    repl_t = _sample_categorical(adjusted / s if s > 1e-10 else t_p[:v])
                final.append(repl_t)
                break

        q_accepted += len(final)
        trim_by = len(q_buffer) - len(final)

        # Trim q_cache and t_cache to the accepted prefix.
        trim_cache(q_cache, trim_by)
        trim_cache(t_cache, trim_by)

        if repl_t is not None:
            # A replacement occurred (at any position, including the last).
            # Both caches end with the rejected q_buffer token; trim it and
            # re-feed final[-1] so the cache content matches the committed sequence.
            trim_cache(t_cache, 1)
            out_t = t_model(mx.array([[final[-1]]]), cache=t_cache)
            t_next_logits = mx.softmax(out_t[0, -1, :])

            trim_cache(q_cache, 1)
            out_q = q_model(mx.array([[final[-1]]]), cache=q_cache)
            q_next_logits = mx.softmax(out_q[0, -1, :])
            del out_t, out_q
        else:
            # All tokens genuinely accepted: t_logits[0, -1, :] is the
            # distribution after the full q_buffer, which is the next-position
            # logit we need.
            t_next_logits = mx.softmax(t_logits[0, len(final) - 1, :])
            # q_next_logits is already correct from the inner loop's final update.

        # Roll back d_cache one extra position and re-feed final[-1] to obtain
        # draft_next_logits and keep d_cache in sync with q_cache and t_cache.
        trim_cache(d_cache, trim_by + 1)
        out = d_model(mx.array([[final[-1]]]), cache=d_cache)
        draft_next_logits = out[0, -1, :]

        del t_logits, out
        mx.eval()

        output_ids.extend(final)
        if tokenizer.eos_token_id in final:
            break

    elapsed = time.perf_counter() - t0
    n_tok   = len(output_ids)
    tok_s   = n_tok / elapsed if elapsed > 0 else 0
    if verbose:
        print(f"\n[Stats/{mode}] {n_tok} tokens | {elapsed:.1f}s | {tok_s:.1f} tok/s")
        print(f"  Draft:     {d_accepted}/{d_proposed} accepted "
              f"({100*d_accepted/max(1,d_proposed):.0f}%)")
        print(f"  Qualifier: {q_accepted}/{q_proposed} accepted "
              f"({100*q_accepted/max(1,q_proposed):.0f}%)")
    return {
        "text": tokenizer.decode(output_ids),
        "tokens": output_ids,
        "tok_s": tok_s,
        "n_tokens": n_tok,
        "elapsed": elapsed,
        "d_accept_rate": d_accepted / max(1, d_proposed),
        "q_accept_rate": q_accepted / max(1, q_proposed),
    }


if __name__ == "__main__":
    d, q, t, tok = load_models()
    print(f"Loaded. Peak mem: {mx.get_peak_memory()/1e9:.2f}GB")
    result = generate_pyramid(d, q, t, tok, "What is a hash table?", max_tokens=150)
    print(result["text"])
