#!/usr/bin/env python3
"""Self-Speculative Decoding: use the target model's own early layers as draft.

Based on Apple's approach (ReDrafter / LayerSkip). The key insight: the first
N layers of a 28-layer model can produce a rough draft via early exit through
the same lm_head. No separate draft model needed — saves ~0.5GB memory and
eliminates tokenizer mismatch issues.

Architecture:
  - Draft pass: embed → layers[0:N] → norm → lm_head  (fast, ~N/28 compute)
  - Verify pass: embed → layers[0:28] → norm → lm_head (full quality)
  - Same weights, same tokenizer, same lm_head
  - KV caches are separate (draft cache vs full cache)

This replaces the 3-model PyramidSD approach with a 1-model, 2-pass approach.
"""
import time
import random
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def load_model():
    """Load single 7B model (no draft/qualifier needed)."""
    print("Loading 7B target model...")
    model, tok = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    n_layers = model.args.num_hidden_layers
    print(f"  {n_layers} layers, vocab={model.args.vocab_size}")
    return model, tok


def _early_exit_forward(model, inputs, cache, n_layers, input_embeddings=None):
    """Run only the first n_layers of the model, then norm + lm_head.

    Returns logits with shape [batch, seq_len, vocab_size].
    Only cache entries for layers 0..n_layers-1 are updated.
    """
    inner = model.model

    # Embed
    if input_embeddings is not None:
        h = input_embeddings
    else:
        h = inner.embed_tokens(inputs)

    # Attention mask (uses first cache entry for offset calculation)
    from mlx_lm.models.base import create_attention_mask
    mask = create_attention_mask(h, cache[0] if cache else None)

    # Run only first n_layers
    for i in range(n_layers):
        h = inner.layers[i](h, mask, cache[i] if cache else None)

    # Apply final norm + lm_head (same as full model)
    h = inner.norm(h)
    if model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)

    return logits


def _full_forward(model, inputs, cache, input_embeddings=None):
    """Standard full forward pass through all layers."""
    return model(inputs, cache=cache, input_embeddings=input_embeddings)


def _compute_entropy(logits):
    """Entropy of softmax(logits) in float32."""
    logits32 = logits.astype(mx.float32)
    probs = mx.softmax(logits32)
    return -mx.sum(probs * mx.log(mx.clip(probs, 1e-10, 1.0))).item()


def _sample_categorical(probs):
    """Sample via gumbel-max trick."""
    log_probs = mx.log(mx.maximum(probs, 1e-10))
    uniform = mx.random.uniform(shape=log_probs.shape)
    gumbel = -mx.log(-mx.log(mx.maximum(uniform, 1e-10)))
    return mx.argmax(log_probs + gumbel).item()


def trim_cache(cache, n):
    """Trim the last n entries from each layer's KV cache."""
    if n > 0:
        for layer in cache:
            layer.trim(n)


def make_early_cache(model, n_layers):
    """Create a KV cache with entries only for the first n_layers.

    We still allocate slots for ALL layers (make_prompt_cache requires it),
    but only the first n_layers will be populated during early-exit forward.
    The remaining layers' cache entries stay empty.
    """
    return make_prompt_cache(model)


def generate_baseline(model, tokenizer, prompt, max_tokens=100, verbose=True):
    """Standard autoregressive generation with full model (greedy)."""
    prompt_ids = mx.array([tokenizer.encode(prompt)])
    cache = make_prompt_cache(model)
    logits = model(prompt_ids, cache=cache)
    mx.eval(cache[0].keys)

    cur_logits = logits[0, -1, :]
    output_ids = []
    t0 = time.perf_counter()

    while len(output_ids) < max_tokens:
        tok_id = mx.argmax(cur_logits).item()
        output_ids.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
        out = model(mx.array([[tok_id]]), cache=cache)
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


def generate_self_spec(model, tokenizer, prompt, max_tokens=100,
                       draft_layers=8, l_d=5, tau=0.3,
                       mode="fuzzy", adaptive=False,
                       l_d_min=1, l_d_max=8, entropy_threshold=2.0,
                       verbose=True):
    """Self-speculative decoding: early-exit draft + full verification.

    Parameters:
        draft_layers: Number of layers for early-exit draft (default 8 of 28)
        l_d:          Draft tokens per speculation round
        tau:          Acceptance threshold (fuzzy mode)
        mode:         "fuzzy" or "lossless" (Leviathan rejection sampling)
        adaptive:     Dynamic draft length based on entropy
        verbose:      Print stats
    """
    n_total_layers = model.args.num_hidden_layers
    prompt_ids = mx.array([tokenizer.encode(prompt)])

    # Separate caches: draft (early exit) and verify (full model)
    # Draft cache: only first draft_layers populated
    # Verify cache: all layers populated
    d_cache = make_prompt_cache(model)
    v_cache = make_prompt_cache(model)

    # Prime draft cache (early exit)
    d_logits = _early_exit_forward(model, prompt_ids, d_cache, draft_layers)
    # Prime verify cache (full model)
    v_logits = _full_forward(model, prompt_ids, v_cache)
    mx.eval(d_cache[0].keys, v_cache[0].keys)

    draft_next_logits = d_logits[0, -1, :]
    verify_next_logits = mx.softmax(v_logits[0, -1, :].astype(mx.float32))
    del d_logits, v_logits

    d_proposed = d_accepted = 0
    draft_lengths = []
    output_ids = []
    t0 = time.perf_counter()

    while len(output_ids) < max_tokens:
        # Draft phase: generate tokens using early-exit (fast)
        draft_toks = []
        d_logits_vecs = []
        cur_logits = draft_next_logits

        effective_l_d = l_d_max if adaptive else l_d

        for step in range(effective_l_d):
            # Adaptive: check entropy before committing
            if adaptive and step >= l_d_min:
                ent = _compute_entropy(cur_logits)
                if ent > entropy_threshold:
                    break

            tok_id = mx.argmax(cur_logits).item()
            draft_toks.append(tok_id)
            d_logits_vecs.append(cur_logits)

            # Early-exit forward for next draft token
            out = _early_exit_forward(
                model, mx.array([[tok_id]]), d_cache, draft_layers
            )
            cur_logits = out[0, -1, :]
            mx.eval(cur_logits)

        actual_l_d = len(draft_toks)
        if actual_l_d == 0:
            break
        draft_lengths.append(actual_l_d)
        d_proposed += actual_l_d

        # Verify phase: run all draft tokens through full model in one pass
        v_logits = _full_forward(model, mx.array([draft_toks]), v_cache)
        mx.eval(v_logits)
        # v_cache: +actual_l_d

        # Accept/reject each draft token
        final = []
        repl = None
        for i in range(actual_l_d):
            # verify distribution for position i
            v_p = verify_next_logits if i == 0 else mx.softmax(
                v_logits[0, i - 1, :].astype(mx.float32))
            d_p = mx.softmax(d_logits_vecs[i].astype(mx.float32))
            tok = draft_toks[i]

            if mode == "fuzzy":
                accept = abs(v_p[tok].item() - d_p[tok].item()) <= tau
            else:  # lossless
                ratio = v_p[tok].item() / max(d_p[tok].item(), 1e-10)
                accept = random.random() < min(1.0, ratio)

            if accept:
                final.append(tok)
                d_accepted += 1
            else:
                # Replacement token
                if mode == "fuzzy":
                    repl = mx.argmax(v_p).item()
                else:
                    v = min(v_p.shape[0], d_p.shape[0])
                    adjusted = mx.maximum(v_p[:v] - d_p[:v], 0)
                    s = adjusted.sum().item()
                    repl = _sample_categorical(
                        adjusted / s if s > 1e-10 else v_p[:v])
                final.append(repl)
                break

        trim_by = actual_l_d - len(final)

        # Sync verify cache
        trim_cache(v_cache, trim_by)
        if repl is not None:
            # Re-feed the replacement token through full model
            trim_cache(v_cache, 1)
            out_v = _full_forward(model, mx.array([[final[-1]]]), v_cache)
            verify_next_logits = mx.softmax(
                out_v[0, -1, :].astype(mx.float32))
            del out_v
        else:
            # All accepted — v_logits[-1] is next-position distribution
            verify_next_logits = mx.softmax(
                v_logits[0, len(final) - 1, :].astype(mx.float32))

        # Sync draft cache: trim back and re-feed last accepted token
        trim_cache(d_cache, trim_by + 1)
        out_d = _early_exit_forward(
            model, mx.array([[final[-1]]]), d_cache, draft_layers
        )
        draft_next_logits = out_d[0, -1, :]
        mx.eval(draft_next_logits)

        del v_logits, d_logits_vecs, out_d
        mx.eval()

        output_ids.extend(final)
        if tokenizer.eos_token_id in final:
            break

    elapsed = time.perf_counter() - t0
    n_tok = len(output_ids)
    tok_s = n_tok / elapsed if elapsed > 0 else 0
    avg_dl = sum(draft_lengths) / max(1, len(draft_lengths))

    tag = f"self-spec-{draft_layers}L/{mode}"
    if adaptive:
        tag += "+adaptive"

    if verbose:
        print(f"\n[{tag}] {n_tok} tokens | {elapsed:.1f}s | {tok_s:.1f} tok/s")
        print(f"  Draft ({draft_layers}/{n_total_layers} layers): "
              f"{d_accepted}/{d_proposed} accepted "
              f"({100*d_accepted/max(1,d_proposed):.0f}%)")
        if adaptive:
            print(f"  Avg draft length: {avg_dl:.1f} "
                  f"(min={min(draft_lengths, default=0)}, "
                  f"max={max(draft_lengths, default=0)})")

    return {
        "text": tokenizer.decode(output_ids),
        "tokens": output_ids,
        "tok_s": tok_s,
        "n_tokens": n_tok,
        "elapsed": elapsed,
        "d_accept_rate": d_accepted / max(1, d_proposed),
        "avg_draft_len": avg_dl,
        "draft_lengths": draft_lengths,
    }


if __name__ == "__main__":
    import sys
    model, tok = load_model()
    print(f"Peak mem: {mx.get_peak_memory()/1e9:.2f}GB")

    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is a hash table?"

    print("\n=== Baseline (full 28 layers) ===")
    b = generate_baseline(model, tok, prompt, max_tokens=100)
    print(b["text"][:200])

    # Test different draft layer counts
    for n_layers in [4, 8, 14]:
        print(f"\n=== Self-Spec ({n_layers} draft layers, fuzzy) ===")
        r = generate_self_spec(model, tok, prompt, max_tokens=100,
                               draft_layers=n_layers, l_d=5, tau=0.3,
                               mode="fuzzy")
        print(r["text"][:200])

    print(f"\n=== Self-Spec (8 layers, lossless) ===")
    r = generate_self_spec(model, tok, prompt, max_tokens=100,
                           draft_layers=8, l_d=5, mode="lossless")
    print(r["text"][:200])

    print(f"\n=== Self-Spec (8 layers, fuzzy+adaptive) ===")
    r = generate_self_spec(model, tok, prompt, max_tokens=100,
                           draft_layers=8, mode="fuzzy",
                           adaptive=True, l_d_min=1, l_d_max=8,
                           entropy_threshold=2.0)
    print(r["text"][:200])
