#!/usr/bin/env python3
"""PyramidSD 3-model speculative decoding on Apple Silicon using MLX.

Simplified, correct KV-cached implementation.
No "peek" — cache state is managed explicitly.
"""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

def load_models():
    print("Loading all models...")
    d, t = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    q, _ = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    tgt, _ = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    return d, q, tgt, t

def trim_cache(c, n):
    if n > 0:
        for l in c: l.trim(n)

def generate_pyramid(d_model, q_model, t_model, tokenizer, prompt, max_tokens=100, l_d=3, l_q=6, tau_q=0.3, tau_t=0.4):
    prompt_ids = mx.array([tokenizer.encode(prompt)])
    d_cache, q_cache, t_cache = make_prompt_cache(d_model), make_prompt_cache(q_model), make_prompt_cache(t_model)
    d_model(prompt_ids, cache=d_cache); q_model(prompt_ids, cache=q_cache); t_model(prompt_ids, cache=t_cache)
    mx.eval(d_cache[0].keys)
    
    output_ids = []
    t0 = time.perf_counter()
    y = prompt_ids
    
    while len(output_ids) < max_tokens:
        q_buffer = []
        q_probs_buf = []
        
        while len(q_buffer) < l_q and len(output_ids) + len(q_buffer) < max_tokens:
            draft_toks, d_logits_vecs = [], []
            for _ in range(l_d):
                logits = d_model(y, cache=d_cache)
                y = mx.argmax(logits[:, -1, :], axis=-1)
                draft_toks.append(y.item())
                d_logits_vecs.append(logits[0, -1, :])

            stats_d_proposed += l_d
            q_logits = q_model(mx.array([draft_toks]), cache=q_cache)
            mx.eval(q_logits)

            n_acc = 0
            has_repl = False
            for i in range(l_d):
                d_probs, q_probs = mx.softmax(d_logits_vecs[i]), mx.softmax(q_logits[0, i, :])
                if abs(q_probs[draft_toks[i]].item() - d_probs[draft_toks[i]].item()) <= tau_q:
                    q_buffer.append(draft_toks[i]); q_probs_buf.append(q_probs); n_acc += 1
                else:
                    repl = mx.argmax(q_probs).item()
                    q_buffer.append(repl); q_probs_buf.append(q_probs); has_repl = True; break

            stats_d_accepted += n_acc
            tokens_kept = n_acc + (1 if has_repl else 0)
            trim_cache(d_cache, l_d - tokens_kept); trim_cache(q_cache, l_d - tokens_kept)
            y = mx.array([[q_buffer[-1]]])

        if not q_buffer: break
        
        stats_q_proposed += len(q_buffer)
        t_logits = t_model(mx.array([q_buffer]), cache=t_cache)
        mx.eval(t_logits)

        final = []
        for i in range(len(q_buffer)):
            t_probs = mx.softmax(t_logits[0, i, :])
            if abs(t_probs[q_buffer[i]].item() - q_probs_buf[i][q_buffer[i]].item()) <= tau_t:
                final.append(q_buffer[i])
            else:
                final.append(mx.argmax(t_probs).item()); break

        stats_q_accepted += len(final)
        trim_cache(d_cache, len(q_buffer) - len(final)); trim_cache(q_cache, len(q_buffer) - len(final)); trim_cache(t_cache, len(q_buffer) - len(final))
        
        output_ids.extend(final)
        y = mx.array([final])
        if tokenizer.eos_token_id in final: break
    
    return tokenizer.decode(output_ids)

# Dummy stats for now
stats_d_proposed, stats_d_accepted, stats_q_proposed, stats_q_accepted = 0, 0, 0, 0
if __name__ == "__main__":
    d, q, t, tok = load_models()
    print(f"Loaded. Mem: {mx.get_peak_memory()/1e9:.2f}GB")
    out = generate_pyramid(d, q, t, tok, "What is a hash table?")
    print(out)
