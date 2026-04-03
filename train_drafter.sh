#!/usr/bin/env bash
# Train ReDrafter RNN head for Qwen2.5-7B-Instruct
#
# GPU options:
#   1. Single GPU (Colab A100/T4, Lambda, RunPod):
#      bash train_drafter.sh
#
#   2. Multi-GPU (8xH100, etc):
#      NGPU=8 bash train_drafter.sh
#
#   3. Local M4 Mac (MPS, ~8-12 hours):
#      LOCAL=1 bash train_drafter.sh
#
# Expected training time:
#   8xH100: ~1.5 hours
#   1xA100: ~6-8 hours
#   1xT4:   ~12-16 hours
#   M4 Mac: ~8-12 hours (MPS)

set -e

LLM="Qwen/Qwen2.5-7B-Instruct"
N_LAYERS=2           # ResBlock layers in drafter head
N_EPOCHS=2
LR=0.001
BATCH_SIZE=4
GRAD_ACCUM=4
PREDICT_N=5          # Draft ahead 5 tokens
OUTPUT_DIR="./drafter_output"
NGPU=${NGPU:-1}

if [ "${LOCAL}" = "1" ]; then
    echo "=== Local M4 Mac training (no bf16, MPS backend) ==="
    python train_drafter.py \
        --llm_name_or_path "$LLM" \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs $N_EPOCHS \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate $LR \
        --weight_decay 0.0 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_strategy no \
        --evaluation_strategy no \
        --model_max_length 1024 \
        --drafter_predict_n_tokens $PREDICT_N \
        --drafter_num_layers $N_LAYERS \
        --rnn True \
        --dataloader_num_workers 0
elif [ "$NGPU" -gt 1 ]; then
    echo "=== Multi-GPU training ($NGPU GPUs) ==="
    torchrun --nproc_per_node=$NGPU train_drafter.py \
        --llm_name_or_path "$LLM" \
        --bf16 True \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs $N_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LR \
        --weight_decay 0.0 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --save_strategy no \
        --evaluation_strategy no \
        --tf32 True \
        --model_max_length 2048 \
        --drafter_predict_n_tokens $PREDICT_N \
        --drafter_num_layers $N_LAYERS \
        --rnn True
else
    echo "=== Single GPU training ==="
    python train_drafter.py \
        --llm_name_or_path "$LLM" \
        --bf16 True \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs $N_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LR \
        --weight_decay 0.0 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --save_strategy no \
        --evaluation_strategy no \
        --tf32 True \
        --model_max_length 2048 \
        --drafter_predict_n_tokens $PREDICT_N \
        --drafter_num_layers $N_LAYERS \
        --rnn True
fi

echo ""
echo "=== Training complete ==="
echo "Drafter saved to: ${OUTPUT_DIR}_redrafter_qwen25-7b_n_${PREDICT_N}_lr_${LR}_layers_${N_LAYERS}"
