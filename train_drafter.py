#!/usr/bin/env python3
"""Train a ReDrafter RNN head for Qwen2.5-7B-Instruct.

Adapted from Apple's ml-recurrent-drafter training code.
Key differences:
  - Uses Qwen2.5 instead of Vicuna/Llama
  - Supports single-GPU training (no DDP required)
  - Uses ShareGPT_Vicuna_unfiltered dataset (same as Apple)
  - Outputs drafter weights compatible with MLX inference

Usage:
  # Single GPU (Colab/Lambda/local):
  python train_drafter.py --output_dir ./drafter_qwen7b

  # Multi-GPU:
  torchrun --nproc_per_node=N train_drafter.py --output_dir ./drafter_qwen7b

  # Local M4 Mac (CPU/MPS, slower):
  python train_drafter.py --output_dir ./drafter_qwen7b --bf16 False --fp16 False
"""

import json
import math
import multiprocessing
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ── Drafter Config ────────────────────────────────────────────────────────────

class DrafterConfig(transformers.PretrainedConfig):
    model_type = "recurrent_drafting_drafter"

    def __init__(
        self,
        vocab_size: int = -1,
        hidden_size: int = -1,
        exit_dim: int = -1,
        num_draft_layers: int = -1,
        rnn: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.exit_dim = exit_dim
        self.num_draft_layers = num_draft_layers
        self.rnn = rnn


# ── Drafter Model ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.linear = nn.Linear(cfg.exit_dim, cfg.exit_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.silu(self.linear(x))


class Drafter(transformers.PreTrainedModel):
    config_class = DrafterConfig

    def __init__(self, cfg: DrafterConfig):
        super().__init__(cfg)
        self.config = cfg

        input_dim = 2 * cfg.hidden_size
        if input_dim != cfg.exit_dim:
            self.input_proj = nn.Linear(input_dim, cfg.exit_dim, bias=True)

        self.lm_head = nn.Sequential(
            *([ResBlock(cfg) for _ in range(cfg.num_draft_layers)]),
            nn.Linear(in_features=cfg.exit_dim, out_features=cfg.vocab_size, bias=False),
        )

        if cfg.rnn:
            self.rnn_u = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
            self.rnn_w = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)


# ── Combined Model (LLM frozen + Drafter trainable) ──────────────────────────

class ReDrafter(nn.Module):
    def __init__(self, llm, drafter):
        super().__init__()
        self.llm = llm
        self.drafter = drafter

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, position_ids=None, next_n=1):
        with torch.inference_mode():
            outputs = self.llm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = outputs[0].clone()  # [batch_size, seq_len, hidden_dim]
        drafter_logits = []
        input_embs = self.llm.model.embed_tokens(input_ids)
        cumsum_input_embs = torch.zeros_like(input_embs)

        for _ in range(next_n):
            input_embs = torch.roll(input_embs, -1, dims=1)
            if self.drafter.config.rnn:
                o = self.drafter.rnn_u(cumsum_input_embs)
                cumsum_input_embs = nn.SiLU()(o + self.drafter.rnn_w(input_embs))
            else:
                cumsum_input_embs += input_embs
            h = torch.cat((hidden_states, cumsum_input_embs), -1)
            logits = self.drafter.lm_head(
                self.drafter.input_proj(h) if hasattr(self.drafter, 'input_proj') else h
            )
            drafter_logits.append(logits)

        return torch.stack(drafter_logits, dim=0)


# ── Loss ──────────────────────────────────────────────────────────────────────

IGNORE_TOKEN_ID = -100

def drafter_loss(logits, labels, next_n, top_k):
    loss, log, eval_log = 0, {}, {}
    for i in range(next_n):
        logits_i = logits[i, :, :-(i + 2)].contiguous().view(-1, logits.shape[-1])
        labels_i = labels[..., i + 2:].contiguous().view(-1).to(logits_i.device)
        loss_i = nn.CrossEntropyLoss()(logits_i, labels_i)
        loss += loss_i
        not_ignore = labels_i.ne(IGNORE_TOKEN_ID)
        labels_i = labels_i[not_ignore]
        for k in range(1, top_k + 1):
            topk = logits_i.topk(k, dim=-1)[-1][not_ignore]
            correct = topk.eq(labels_i.unsqueeze(-1)).any(-1)
            log[f"drafter{i}_top{k}"] = correct.float().mean().item()
            eval_log[f"drafter{i}_top{k}"] = correct.float().mean()
        log[f"drafter{i}_loss"] = loss_i.item()
    return loss, log, eval_log


# ── Data Processing ───────────────────────────────────────────────────────────

def process_sharegpt_for_qwen(example, tokenizer, max_length=2048):
    """Convert ShareGPT format to tokenized training data for Qwen."""
    conversations = example.get("conversations", [])
    if not conversations:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # Build chat from conversations
    messages = []
    for turn in conversations:
        role = turn.get("from", "")
        content = turn.get("value", "")
        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})

    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # Use Qwen's chat template
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, truncation=True, max_length=max_length,
                           padding="max_length", return_tensors="pt")
        input_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        # Labels = input_ids (shifted internally by the loss function)
        labels = [t if m == 1 else IGNORE_TOKEN_ID
                  for t, m in zip(input_ids, attention_mask)]
    except Exception:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Trainer ───────────────────────────────────────────────────────────────────

class ReDrafterTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        next_n = self.args.drafter_predict_n_tokens
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            next_n=next_n,
        )
        loss, log, eval_log = drafter_loss(
            logits, inputs["labels"], next_n, self.args.drafter_top_k
        )
        self.log(log)
        return (loss, eval_log) if return_outputs else loss


# ── Training Arguments ────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    llm_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct"
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    drafter_predict_n_tokens: int = field(default=5)
    drafter_top_k: int = field(default=5)
    drafter_num_layers: int = field(default=2)
    rnn: bool = field(default=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Loading tokenizer from {model_args.llm_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset (ShareGPT)...")
    raw_dataset = datasets.load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    train_dataset = raw_dataset.map(
        lambda x: process_sharegpt_for_qwen(x, tokenizer, training_args.model_max_length),
        num_proc=min(multiprocessing.cpu_count(), 8),
        remove_columns=raw_dataset.column_names,
    )
    # Filter empty examples
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"Training dataset: {len(train_dataset)} examples")

    print(f"Loading LLM from {model_args.llm_name_or_path}...")
    config = AutoConfig.from_pretrained(model_args.llm_name_or_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        model_args.llm_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Freeze all LLM parameters
    for param in llm.parameters():
        param.requires_grad = False
    print(f"LLM loaded. hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")

    # Create drafter
    drafter_config = DrafterConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        exit_dim=2 * config.hidden_size,
        num_draft_layers=training_args.drafter_num_layers,
        rnn=training_args.rnn,
    )
    drafter = Drafter(drafter_config)

    # Count trainable params
    total_params = sum(p.numel() for p in drafter.parameters())
    print(f"Drafter parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  hidden_size={drafter_config.hidden_size}")
    print(f"  exit_dim={drafter_config.exit_dim}")
    print(f"  num_draft_layers={drafter_config.num_draft_layers}")
    print(f"  rnn={drafter_config.rnn}")
    print(f"  vocab_size={drafter_config.vocab_size}")

    redrafter = ReDrafter(llm, drafter)

    # Output dir
    training_args.output_dir = (
        f"{training_args.output_dir}"
        f"_redrafter_qwen25-7b"
        f"_n_{training_args.drafter_predict_n_tokens}"
        f"_lr_{training_args.learning_rate}"
        f"_layers_{training_args.drafter_num_layers}"
    )
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Save drafter config
    drafter_config.save_pretrained(training_args.output_dir)

    trainer = ReDrafterTrainer(
        model=redrafter,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
    )

    print(f"\nStarting training...")
    print(f"  Output: {training_args.output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Predict N tokens: {training_args.drafter_predict_n_tokens}")

    trainer.train()

    # Save drafter weights
    print(f"\nSaving drafter to {training_args.output_dir}")
    drafter.save_pretrained(training_args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
