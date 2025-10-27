# PixelRec Training - DeepSpeed Implementation

重构后的PixelRec训练代码，使用DeepSpeed框架替代原始的torchrun。

## 项目结构

```
pixelrec_train/
├── data/
│   ├── __init__.py
│   └── dataset.py          # 数据加载模块
├── model/
│   ├── __init__.py
│   └── hllm_model.py       # 双层LLM模型
├── scripts/
│   ├── train_pixelrec.py   # 训练脚本
│   └── run_train_pixelrec.sh  # 启动脚本
├── configs/
│   ├── ds_config_zero2.json   # DeepSpeed ZeRO-2配置
│   └── ds_config_zero3.json   # DeepSpeed ZeRO-3配置
├── requirements.txt
└── README.md
```

## 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- torch >= 2.0.0
- transformers >= 4.30.0
- deepspeed >= 0.10.0
- pandas >= 1.5.0

## 数据准备

需要准备两个CSV文件：

1. **交互数据** (`Pixel200K.csv`): 用户-物品交互序列
   - 必需列: `user_id`, `item_id`
   - 可选列: `timestamp`

2. **文本特征数据** (`Pixel200K.csv`): 物品的文本描述
   - 必需列: `item_id`, `title`, `description`

## 训练配置

在 `scripts/run_train_pixelrec.sh` 中配置以下参数：

```bash
# 模型路径
ITEM_MODEL_DIR="path/to/item_llm"     # item LLM预训练模型
USER_MODEL_DIR="path/to/user_llm"     # user LLM预训练模型

# 数据路径
INTERACTION_DATA="../dataset/Pixel200K.csv"
TEXT_DATA="../information/Pixel200K.csv"

# 训练参数
BATCH_SIZE=4              # 每GPU的batch size
EPOCHS=5                  # 训练轮数
LEARNING_RATE=1e-4        # 学习率
MAX_SEQ_LENGTH=11         # 最大序列长度 (MAX_ITEM_LIST_LENGTH + 1)
MAX_TEXT_LENGTH=256       # 物品文本最大长度
TEXT_KEYS="title,description"  # 使用的文本特征
```

## 启动训练

### 单机多卡训练

```bash
cd /path/to/pixelrec_train
bash scripts/run_train_pixelrec.sh
```

训练日志会输出到 `pixelrec_train.log`

### 自定义启动

也可以直接使用deepspeed命令：

```bash
deepspeed --num_gpus=8 \
    scripts/train_pixelrec.py \
    --interaction_data ../dataset/Pixel200K.csv \
    --text_data ../information/Pixel200K.csv \
    --item_model_dir /path/to/item_llm \
    --user_model_dir /path/to/user_llm \
    --per_device_train_batch_size 4 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --deepspeed configs/ds_config_zero2.json
```

## DeepSpeed配置

项目提供了两个DeepSpeed配置：

- `ds_config_zero2.json`: ZeRO-2优化，适合内存充足的情况
- `ds_config_zero3.json`: ZeRO-3优化，可节省更多内存

如需切换，修改 `run_train_pixelrec.sh` 中的 `--deepspeed` 参数。

## 输出

训练过程中会保存：

1. **检查点**: `results/pixelrec/checkpoint-epoch-{N}/`
   - 每个epoch结束后保存

2. **最佳模型**: `results/pixelrec/best_model/`
   - 当验证loss下降时更新

3. **日志**: `logs/pixelrec/`
   - TensorBoard日志

## 训练流程说明

本重构保持了与原始HLLM代码完全一致的训练流程：

1. **数据处理**:
   - 从CSV加载用户交互序列
   - 排除最后2个item用于评估
   - 为每个序列位置采样负样本
   - 将item ID映射到文本，进行tokenization

2. **模型架构**:
   - Item LLM: 编码物品文本为embedding
   - User LLM: 编码用户序列为表示
   - NCE Loss: 对比学习损失函数

3. **训练优化**:
   - AdamW优化器
   - Warmup + Cosine学习率调度
   - BF16混合精度训练
   - Gradient Checkpointing节省内存

## 与原始代码的区别

- ✅ 使用DeepSpeed替代torchrun进行分布式训练
- ✅ 简化配置，移除不必要的参数
- ✅ 独立的数据加载模块，不依赖原始codebase
- ✅ 保持完全相同的训练逻辑和损失计算
- ✅ 支持相同的checkpoint保存策略

## 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `per_device_train_batch_size` | 4 | 每GPU的batch size |
| `num_train_epochs` | 5 | 训练轮数 |
| `learning_rate` | 1e-4 | 学习率 |
| `warmup_ratio` | 0.1 | Warmup比例 |
| `weight_decay` | 0.01 | 权重衰减 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `gradient_checkpointing` | True | 梯度检查点 |
| `max_seq_length` | 11 | 序列长度 |
| `max_text_length` | 256 | 文本长度 |
| `item_emb_token_n` | 1 | 每个item的embedding token数 |
| `num_negatives` | None | 负样本数（None表示序列级采样）|
| `bf16` | True | 使用BF16精度 |

## 监控训练

查看实时日志：
```bash
tail -f pixelrec_train.log
```

查看TensorBoard：
```bash
tensorboard --logdir=logs/tensorboard
```

## 常见问题

### OOM (Out of Memory)

如果遇到显存不足：
1. 减小 `per_device_train_batch_size`
2. 使用 `ds_config_zero3.json` 配置
3. 减小 `max_seq_length` 或 `max_text_length`

### 数据加载慢

如果数据加载慢：
1. 增加 `dataloader_num_workers`
2. 确保CSV文件在快速存储上

### 训练不稳定

如果训练loss波动大：
1. 减小学习率
2. 增加warmup步数
3. 检查数据质量
