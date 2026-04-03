#!/usr/bin/env python3
"""Quality validation benchmark: PyramidSD modes vs 7B baseline.

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


# ── Mode configurations ──────────────────────────────────────────────────────
MODES = [
    {
        "label": "fuzzy",
        "short": "Fuzz",
        "kwargs": {"mode": "fuzzy"},
    },
    {
        "label": "lossless",
        "short": "Loss",
        "kwargs": {"mode": "lossless"},
    },
    {
        "label": "fuzzy+tree+adapt",
        "short": "FzTA",
        "kwargs": {"mode": "fuzzy", "tree": True, "top_k": 3,
                   "adaptive": True, "l_d_min": 1, "l_d_max": 8,
                   "entropy_threshold": 2.0},
    },
    {
        "label": "lossless+tree+adapt",
        "short": "LsTA",
        "kwargs": {"mode": "lossless", "tree": True, "top_k": 3,
                   "adaptive": True, "l_d_min": 1, "l_d_max": 8,
                   "entropy_threshold": 2.0},
    },
]


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(max_tokens=50):
    n_modes = len(MODES)
    print("=" * 72)
    print("  PyramidSD Quality Validation Benchmark")
    print(f"  {len(TEST_PROMPTS)} prompts × max_tokens={max_tokens}")
    print(f"  Modes: baseline + {', '.join(m['label'] for m in MODES)}")
    print("=" * 72)

    d, q, t, tok = load_models()

    results = []

    for idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─'*60}")
        print(f"  Prompt {idx+1:2d}/{len(TEST_PROMPTS)}: {prompt}")
        print(f"{'─'*60}")

        # Baseline
        print(f"\n[1/{n_modes+1}] Baseline (7B autoregressive)")
        base = generate_baseline(t, tok, prompt, max_tokens=max_tokens)
        b = base["tokens"]

        r = {
            "prompt": prompt,
            "base_tok_s": base["tok_s"],
        }

        # Run each mode
        mode_results = {}
        for mi, m in enumerate(MODES):
            print(f"\n[{mi+2}/{n_modes+1}] PyramidSD – {m['label']}")
            res = generate_pyramid(d, q, t, tok, prompt,
                                   max_tokens=max_tokens, **m["kwargs"])
            h = res["tokens"]
            prefix = m["short"].lower()
            r[f"{prefix}_tok_s"] = res["tok_s"]
            r[f"{prefix}_match"] = token_match_rate(b, h)
            r[f"{prefix}_bleu2"] = bleu2(b, h)
            r[f"{prefix}_f1"] = unigram_f1(b, h)
            r[f"{prefix}_d_acc"] = res["d_accept_rate"]
            r[f"{prefix}_q_acc"] = res["q_accept_rate"]
            r[f"{prefix}_tree_saves"] = res.get("tree_saves", 0)
            r[f"{prefix}_avg_dl"] = res.get("avg_draft_len", 0)
            mode_results[m["short"]] = res

        results.append(r)

        # Per-prompt summary
        print(f"\n  Speed — base:{r['base_tok_s']:.1f}", end="")
        for m in MODES:
            p = m["short"].lower()
            print(f"  {m['short']}:{r[f'{p}_tok_s']:.1f}", end="")
        print(" tok/s")
        for m in MODES:
            p = m["short"].lower()
            print(f"  {m['label']:22s} — match:{r[f'{p}_match']:.3f}"
                  f"  BLEU-2:{r[f'{p}_bleu2']:.3f}  F1:{r[f'{p}_f1']:.3f}")

    # ── Aggregate summary table ───────────────────────────────────────────────
    n = len(results)
    avg = lambda k: sum(r[k] for r in results) / n

    col_w = 6
    mode_shorts = [m["short"] for m in MODES]

    print("\n\n" + "=" * 120)
    print("  BENCHMARK SUMMARY TABLE")
    print("=" * 120)

    # Header
    hdr = f"{'#':>2}  {'Prompt':<32}  {'Base':>{col_w}}"
    for s in mode_shorts:
        hdr += f" {s:>{col_w}}"
    hdr += "  "
    for s in mode_shorts:
        hdr += f" {s[:4]+'M':>{col_w}}"
    print(hdr)

    sub = f"{'':>2}  {'':32}  {'tok/s':>{col_w}}"
    for _ in mode_shorts:
        sub += f" {'tok/s':>{col_w}}"
    sub += "  "
    for _ in mode_shorts:
        sub += f" {'match':>{col_w}}"
    print(sub)
    print("─" * 120)

    for i, r in enumerate(results):
        p = r["prompt"][:31]
        line = f"{i+1:>2}  {p:<32}  {r['base_tok_s']:>{col_w}.1f}"
        for s in mode_shorts:
            k = f"{s.lower()}_tok_s"
            line += f" {r[k]:>{col_w}.1f}"
        line += "  "
        for s in mode_shorts:
            k = f"{s.lower()}_match"
            line += f" {r[k]:>{col_w}.3f}"
        print(line)

    print("─" * 120)
    line = f"{'AVG':>2}  {'':32}  {avg('base_tok_s'):>{col_w}.1f}"
    for s in mode_shorts:
        line += f" {avg(f'{s.lower()}_tok_s'):>{col_w}.1f}"
    line += "  "
    for s in mode_shorts:
        line += f" {avg(f'{s.lower()}_match'):>{col_w}.3f}"
    print(line)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  AGGREGATE METRICS")
    print("=" * 72)

    base_avg = avg("base_tok_s")

    print(f"  Speed")
    print(f"    Baseline:              {base_avg:>6.1f} tok/s")
    for m in MODES:
        p = m["short"].lower()
        m_avg = avg(f"{p}_tok_s")
        print(f"    {m['label']:22s} {m_avg:>6.1f} tok/s  "
              f"({m_avg/base_avg:.2f}x)")

    for m in MODES:
        p = m["short"].lower()
        print(f"\n  Quality vs Baseline ({m['label']})")
        print(f"    Exact match rate: {avg(f'{p}_match'):>6.3f}")
        print(f"    BLEU-2:           {avg(f'{p}_bleu2'):>6.3f}")
        print(f"    Unigram F1:       {avg(f'{p}_f1'):>6.3f}")
        print(f"    Draft accept:     {avg(f'{p}_d_acc'):>6.1%}")
        print(f"    Qualifier accept: {avg(f'{p}_q_acc'):>6.1%}")
        ts = avg(f'{p}_tree_saves')
        dl = avg(f'{p}_avg_dl')
        if ts > 0:
            print(f"    Tree saves:       {ts:>6.1f} avg/prompt")
        if dl > 0:
            print(f"    Avg draft length: {dl:>6.1f}")

    print()


if __name__ == "__main__":
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_benchmark(max_tokens)
