#!/usr/bin/env python3
"""Quality validation benchmark: PyramidSD (fuzzy + lossless) vs 7B baseline.

Usage:
    python benchmark.py           # 50 tokens per prompt
    python benchmark.py 100       # 100 tokens per prompt
"""
import math
import sys
from pyramid_mlx import load_models, generate_baseline, generate_pyramid

# ── 20 diverse test prompts ───────────────────────────────────────────────────
TEST_PROMPTS = [
    # Factual (5)
    "What is the capital of France?",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light in meters per second?",
    "How many planets are in the solar system?",
    # Medium reasoning (5)
    "Explain how TCP/IP works in 2 paragraphs.",
    "What is the difference between RAM and ROM?",
    "Explain what a neural network is in simple terms.",
    "How does HTTPS encryption protect your data?",
    "What is the difference between compiled and interpreted languages?",
    # Complex (4)
    "Compare merge sort and quicksort, including time complexity analysis.",
    "Explain the CAP theorem in distributed systems.",
    "What are the SOLID principles in software engineering?",
    "Compare REST and GraphQL APIs, discussing trade-offs.",
    # Creative (3)
    "Write a haiku about programming.",
    "Write a limerick about a software bug.",
    "Describe a sunset in two vivid sentences.",
    # Math (3)
    "What is the derivative of x^3 + 2x?",
    "Solve for x: 2x + 5 = 13.",
    "What is the integral of sin(x)?",
]


# ── Quality metrics ───────────────────────────────────────────────────────────

def token_match_rate(ref, hyp):
    """Exact positional token match rate over the shorter sequence."""
    length = min(len(ref), len(hyp))
    if length == 0:
        return 0.0
    return sum(a == b for a, b in zip(ref[:length], hyp[:length])) / length


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def ngram_precision(ref, hyp, n):
    """Clipped n-gram precision (as used in BLEU)."""
    hyp_ngrams = _ngrams(hyp, n)
    if not hyp_ngrams:
        return 0.0
    ref_counts = {}
    for g in _ngrams(ref, n):
        ref_counts[g] = ref_counts.get(g, 0) + 1
    clip = 0
    used = {}
    for g in hyp_ngrams:
        if ref_counts.get(g, 0) > used.get(g, 0):
            clip += 1
            used[g] = used.get(g, 0) + 1
    return clip / len(hyp_ngrams)


def bleu2(ref, hyp):
    """BLEU-2: geometric mean of 1-gram and 2-gram clipped precision."""
    p1 = ngram_precision(ref, hyp, 1)
    p2 = ngram_precision(ref, hyp, 2)
    if p1 == 0 or p2 == 0:
        return 0.0
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref) / max(len(hyp), 1)))
    return bp * math.sqrt(p1 * p2)


def unigram_f1(ref, hyp):
    """Token-level unigram F1 (shared / avg length)."""
    if not ref or not hyp:
        return 0.0
    ref_set = set(ref)
    hyp_set = set(hyp)
    shared = len(ref_set & hyp_set)
    precision = shared / len(hyp_set)
    recall = shared / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(max_tokens=50):
    print("=" * 72)
    print("  PyramidSD Quality Validation Benchmark")
    print(f"  {len(TEST_PROMPTS)} prompts × max_tokens={max_tokens}")
    print("=" * 72)

    d, q, t, tok = load_models()

    results = []

    for idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─'*60}")
        print(f"  Prompt {idx+1:2d}/{len(TEST_PROMPTS)}: {prompt}")
        print(f"{'─'*60}")

        print("\n[1/3] Baseline (7B autoregressive)")
        base = generate_baseline(t, tok, prompt, max_tokens=max_tokens)

        print("\n[2/3] PyramidSD – fuzzy mode")
        fuzzy = generate_pyramid(d, q, t, tok, prompt, max_tokens=max_tokens,
                                 mode="fuzzy")

        print("\n[3/3] PyramidSD – lossless mode")
        lossless = generate_pyramid(d, q, t, tok, prompt, max_tokens=max_tokens,
                                    mode="lossless")

        b = base["tokens"]
        f = fuzzy["tokens"]
        ls = lossless["tokens"]

        r = {
            "prompt":           prompt,
            # Speed
            "base_tok_s":       base["tok_s"],
            "fuzzy_tok_s":      fuzzy["tok_s"],
            "lossless_tok_s":   lossless["tok_s"],
            # Fuzzy quality vs baseline
            "f_match":          token_match_rate(b, f),
            "f_bleu2":          bleu2(b, f),
            "f_f1":             unigram_f1(b, f),
            # Lossless quality vs baseline
            "l_match":          token_match_rate(b, ls),
            "l_bleu2":          bleu2(b, ls),
            "l_f1":             unigram_f1(b, ls),
            # Acceptance rates
            "f_d_acc":          fuzzy["d_accept_rate"],
            "f_q_acc":          fuzzy["q_accept_rate"],
            "l_d_acc":          lossless["d_accept_rate"],
            "l_q_acc":          lossless["q_accept_rate"],
            # Output lengths
            "base_len":         len(b),
            "fuzzy_len":        len(f),
            "lossless_len":     len(ls),
        }
        results.append(r)

        # Per-prompt quick summary
        print(f"\n  Speed  — base:{r['base_tok_s']:.1f}  fuzzy:{r['fuzzy_tok_s']:.1f}"
              f"  lossless:{r['lossless_tok_s']:.1f} tok/s")
        print(f"  Fuzzy  — match:{r['f_match']:.3f}  BLEU-2:{r['f_bleu2']:.3f}"
              f"  F1:{r['f_f1']:.3f}")
        print(f"  Lossless — match:{r['l_match']:.3f}  BLEU-2:{r['l_bleu2']:.3f}"
              f"  F1:{r['l_f1']:.3f}")

    # ── Aggregate summary table ───────────────────────────────────────────────
    print("\n\n" + "=" * 96)
    print("  BENCHMARK SUMMARY TABLE")
    print("=" * 96)
    hdr = (f"{'#':>2}  {'Prompt':<36}  "
           f"{'Base':>5} {'Fuzz':>5} {'Loss':>5}  "
           f"{'F-mat':>5} {'F-bl2':>5} {'F-f1':>5}  "
           f"{'L-mat':>5} {'L-bl2':>5} {'L-f1':>5}")
    sub = (f"{'':>2}  {'':36}  "
           f"{'tok/s':>5} {'tok/s':>5} {'tok/s':>5}  "
           f"{'rate':>5} {'score':>5} {'score':>5}  "
           f"{'rate':>5} {'score':>5} {'score':>5}")
    print(hdr)
    print(sub)
    print("─" * 96)

    for i, r in enumerate(results):
        p = r["prompt"][:35]
        print(f"{i+1:>2}  {p:<36}  "
              f"{r['base_tok_s']:>5.1f} {r['fuzzy_tok_s']:>5.1f} {r['lossless_tok_s']:>5.1f}  "
              f"{r['f_match']:>5.3f} {r['f_bleu2']:>5.3f} {r['f_f1']:>5.3f}  "
              f"{r['l_match']:>5.3f} {r['l_bleu2']:>5.3f} {r['l_f1']:>5.3f}")

    print("─" * 96)
    n = len(results)
    avg = lambda k: sum(r[k] for r in results) / n
    print(f"{'AVG':>2}  {'':36}  "
          f"{avg('base_tok_s'):>5.1f} {avg('fuzzy_tok_s'):>5.1f} {avg('lossless_tok_s'):>5.1f}  "
          f"{avg('f_match'):>5.3f} {avg('f_bleu2'):>5.3f} {avg('f_f1'):>5.3f}  "
          f"{avg('l_match'):>5.3f} {avg('l_bleu2'):>5.3f} {avg('l_f1'):>5.3f}")

    print("\n" + "=" * 72)
    print("  AGGREGATE METRICS")
    print("=" * 72)

    base_avg  = avg("base_tok_s")
    fuzzy_avg = avg("fuzzy_tok_s")
    loss_avg  = avg("lossless_tok_s")

    print(f"  Speed")
    print(f"    Baseline:         {base_avg:>6.1f} tok/s")
    print(f"    Fuzzy:            {fuzzy_avg:>6.1f} tok/s  ({fuzzy_avg/base_avg:.2f}x speedup)")
    print(f"    Lossless:         {loss_avg:>6.1f} tok/s  ({loss_avg/base_avg:.2f}x speedup)")

    print(f"\n  Quality vs Baseline (Fuzzy mode)")
    print(f"    Exact match rate: {avg('f_match'):>6.3f}")
    print(f"    BLEU-2:           {avg('f_bleu2'):>6.3f}")
    print(f"    Unigram F1:       {avg('f_f1'):>6.3f}")
    print(f"    Draft accept:     {avg('f_d_acc'):>6.1%}")
    print(f"    Qualifier accept: {avg('f_q_acc'):>6.1%}")

    print(f"\n  Quality vs Baseline (Lossless mode)")
    print(f"    Exact match rate: {avg('l_match'):>6.3f}")
    print(f"    BLEU-2:           {avg('l_bleu2'):>6.3f}")
    print(f"    Unigram F1:       {avg('l_f1'):>6.3f}")
    print(f"    Draft accept:     {avg('l_d_acc'):>6.1%}")
    print(f"    Qualifier accept: {avg('l_q_acc'):>6.1%}")

    print()


if __name__ == "__main__":
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_benchmark(max_tokens)
