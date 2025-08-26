#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliate

cd code
model_path=$1

shift 1
if [ $# -gt 0 ]; then
    num_gpus=$1
else
    num_gpus=8 
fi
echo num_gpus=$num_gpus # use $num_gpus extract user emb

# fill in these params on your machine
item_pretrain_dir=''
user_pretrain_dir=''
creative_pretrain_dir=''
train_data_path=''
test_data_path=''

# 0. convert checkpoint to hf
if [ ! -e "$model_path/zero3_merge_states.pt" ]; then
    echo "merge zero3 ckpt"
    python3 $model_path/zero_to_fp32.py $model_path $model_path/zero3_merge_states.pt
fi
if [ ! -e "$model_path/pytorch_model.bin" ]; then
    echo "convert zero3 to hf"
    python3 HLLM_Creator_eval_scripts/convert2hf.py --src_path $model_path/zero3_merge_states.pt --trg_path $model_path/pytorch_model.bin
    cp $creative_pretrain_dir/*config* $model_path/
    cp $creative_pretrain_dir/*token* $model_path/
    cp $creative_pretrain_dir/*.txt $model_path/
    cp $creative_pretrain_dir/vocab.json $model_path/
fi

# 1. get user emb
if [ ! -e "$model_path/rank0_user_emb.pt" ]; then
    python3 HLLM_Creator_eval_scripts/generate_useremb.py \
    --model_path $model_path/zero3_merge_states.pt \
    --data_path $train_data_path \
    --output_path $model_path \
    --batch_size 4 \
    --num_gpus $num_gpus \
    --random_sample \
    --config_file overall/LLM_deepspeed.yaml HLLM_Creator/HLLM_Creator.yaml \
    --MAX_ITEM_LIST_LENGTH 50 \
    --MAX_TEXT_LENGTH 64 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --creative_pretrain_dir $creative_pretrain_dir \
    --limit 65536
fi

# 2. clustering
# pip3 install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
if [ ! -e "$model_path/cluster_256.pt" ]; then
    python3 HLLM_Creator_eval_scripts/cluster.py \
    --user_emb_path $model_path \
    --num_clusters 256
fi

# 3. generate
if [ ! -e $model_path/output_cluster.parquet ]; then
    python3 HLLM_Creator_eval_scripts/generate_usertitle.py \
    --ckpt_model_path $model_path/zero3_merge_states.pt \
    --creative_model_path $model_path \
    --data_path $test_data_path \
    --output_path $model_path/output_cluster.parquet \
    --batch_size 4 \
    --num_gpus 1 \
    --random_sample \
    --config_file overall/LLM_deepspeed.yaml HLLM_Creator/HLLM_Creator.yaml \
    --MAX_ITEM_LIST_LENGTH 50 \
    --MAX_TEXT_LENGTH 64 \
    --item_pretrain_dir $item_pretrain_dir \
    --user_pretrain_dir $user_pretrain_dir \
    --creative_pretrain_dir $creative_pretrain_dir \
    --cluster_path $model_path/cluster_256.pt
fi