# Item and User LLM are initialized by specific pretrain_dir.
python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--loss nce \
--epochs 5 \
--dataset Pixel200K \
--train_batch_size 4 \
--MAX_TEXT_LENGTH 256 \
--MAX_ITEM_LIST_LENGTH 10 \
--checkpoint_dir saved_path \
--optim_args.learning_rate 1e-4 \
--item_pretrain_dir /llm-reco-ssd-share/zhangrongzhou/OneRec-Think/test_test/OneRec-Think/basemodel/Qwen3-1-7B \
--user_pretrain_dir /llm-reco-ssd-share/zhangrongzhou/OneRec-Think/test_test/OneRec-Think/basemodel/Qwen3-1-7B \
--text_path /llm-reco-ssd-share/zhangrongzhou/HLLM/HLLM/information \
--text_keys '["title","description"]'