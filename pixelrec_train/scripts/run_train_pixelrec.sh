#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Configuration
ITEM_MODEL_DIR="/llm-reco-ssd-share/zhangrongzhou/OneRec-Think/test_test/OneRec-Think/basemodel/Qwen3-1-7B"
USER_MODEL_DIR="/llm-reco-ssd-share/zhangrongzhou/OneRec-Think/test_test/OneRec-Think/basemodel/Qwen3-1-7B"

INTERACTION_DATA="../dataset/Pixel200K.csv"
TEXT_DATA="../information/Pixel200K.csv"

OUTPUT_DIR="./results/pixelrec"
LOGGING_DIR="./logs/pixelrec"

# Training parameters (matching original config)
BATCH_SIZE=8
EPOCHS=5
LEARNING_RATE=1e-4
MAX_SEQ_LENGTH=11          # MAX_ITEM_LIST_LENGTH (10) + 1
MAX_TEXT_LENGTH=256
TEXT_KEYS="title,tag,description"

# Launch training with DeepSpeed
nohup deepspeed --num_gpus=8 \
    ./scripts/train_pixelrec.py \
    --interaction_data "${INTERACTION_DATA}" \
    --text_data "${TEXT_DATA}" \
    --item_model_dir "${ITEM_MODEL_DIR}" \
    --user_model_dir "${USER_MODEL_DIR}" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_text_length ${MAX_TEXT_LENGTH} \
    --text_keys "${TEXT_KEYS}" \
    --item_emb_token_n 1 \
    --num_negatives None \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOGGING_DIR}" \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --bf16 True \
    --deepspeed ./configs/ds_config_zero2.json \
    --dataloader_num_workers 4 \
    >> pixelrec_train.log 2>&1 &

echo "Training started in background. Check pixelrec_train.log for progress."
echo "PID: $!"
