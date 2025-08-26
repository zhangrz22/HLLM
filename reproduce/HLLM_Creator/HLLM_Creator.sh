#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliate

# 1B + 8B: 64 A100s for ≈ 0.5days
cd code && python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM_Creator/HLLM_Creator.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--MAX_TEXT_LENGTH 64 \
--MAX_GEN_TEXT_LENGTH 4096 \
--MAX_USER_PROFILE_TEXT_LENGTH 1024 \
--epochs 1 \
--optim_args.learning_rate 2e-5 \
--checkpoint_dir checkpoint_dir \
--gradient_checkpointing True \
--train_batch_size 2 \
--item_pretrain_dir item_pretrain_dir \
--user_pretrain_dir user_pretrain_dir \
--creative_pretrain_dir creative_pretrain_dir \
--train_path train_path \
--stage 3