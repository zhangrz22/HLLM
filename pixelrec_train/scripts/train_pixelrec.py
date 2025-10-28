#!/usr/bin/env python3
"""
PixelRec Training Script with DeepSpeed
Refactored from original HLLM code to use DeepSpeed framework
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import deepspeed
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import PixelRecDataset
from model.hllm_model import HLLMModel


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train PixelRec with DeepSpeed')

    # Custom type for optional int (allows None)
    def optional_int(value):
        if value.lower() == 'none':
            return None
        return int(value)

    # Custom type for bool (handles 'True'/'False' strings)
    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Data arguments
    parser.add_argument('--interaction_data', type=str, required=True,
                        help='Path to interaction CSV file')
    parser.add_argument('--text_data', type=str, required=True,
                        help='Path to text features CSV file')

    # Model arguments
    parser.add_argument('--item_model_dir', type=str, required=True,
                        help='Path to pretrained item LLM')
    parser.add_argument('--user_model_dir', type=str, required=True,
                        help='Path to pretrained user LLM')
    parser.add_argument('--item_emb_token_n', type=int, default=1,
                        help='Number of embedding tokens per item')
    parser.add_argument('--num_negatives', type=optional_int, default=None,
                        help='Number of negative samples (None for sequence-level)')

    # Training arguments
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                        help='Batch size per device')
    parser.add_argument('--num_train_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm')
    parser.add_argument('--gradient_checkpointing', type=str_to_bool, default=True,
                        help='Use gradient checkpointing')

    # Data arguments
    parser.add_argument('--max_seq_length', type=int, default=11,
                        help='Max sequence length (MAX_ITEM_LIST_LENGTH + 1)')
    parser.add_argument('--max_text_length', type=int, default=256,
                        help='Max text length')
    parser.add_argument('--text_keys', type=str, default='title,description',
                        help='Comma-separated text keys')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='Number of dataloader workers')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results/pixelrec',
                        help='Output directory')
    parser.add_argument('--logging_dir', type=str, default='./logs/pixelrec',
                        help='Logging directory')
    parser.add_argument('--logging_steps', type=int, default=20,
                        help='Log every N steps')
    parser.add_argument('--save_strategy', type=str, default='epoch',
                        help='Save strategy (epoch/steps)')
    parser.add_argument('--save_total_limit', type=int, default=5,
                        help='Max number of checkpoints to keep')

    # DeepSpeed arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--deepspeed', type=str, default=None,
                        help='DeepSpeed config file')
    parser.add_argument('--bf16', type=str_to_bool, default=True,
                        help='Use bf16 precision')

    # Parse args (allow unknown for DeepSpeed)
    args, unknown = parser.parse_known_args()

    return args


def create_dataloader(args, tokenizer_path):
    """Create training dataloader"""
    text_keys = args.text_keys.split(',')

    dataset = PixelRecDataset(
        interaction_file=args.interaction_data,
        text_file=args.text_data,
        tokenizer_path=tokenizer_path,
        max_seq_length=args.max_seq_length,
        max_text_length=args.max_text_length,
        text_keys=text_keys,
        item_emb_token_n=args.item_emb_token_n,
        num_negatives=args.num_negatives,
        use_nce=True,
    )

    # Custom collate function to handle variable length tensors
    def collate_fn(batch):
        """Collate function for batching"""
        batched = {}
        for key in batch[0].keys():
            if key in ['pos_input_ids', 'pos_cu_input_lens', 'pos_position_ids',
                       'neg_input_ids', 'neg_cu_input_lens', 'neg_position_ids']:
                # Concatenate these tensors
                batched[key] = torch.cat([item[key] for item in batch], dim=0)
            else:
                # Stack these tensors
                batched[key] = torch.stack([item[key] for item in batch], dim=0)
        return batched

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader, len(dataset)


def main():
    args = parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Get local rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    logger.info(f"Local rank: {local_rank}, World size: {world_size}")
    logger.info(f"Arguments: {args}")

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # Create model
    logger.info("Creating model...")
    model = HLLMModel(
        item_pretrain_dir=args.item_model_dir,
        user_pretrain_dir=args.user_model_dir,
        item_emb_token_n=args.item_emb_token_n,
        num_negatives=args.num_negatives,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Create dataloader
    logger.info("Creating dataloader...")
    train_dataloader, total_samples = create_dataloader(args, args.item_model_dir)

    # Calculate training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Set DeepSpeed auto parameters
    args.warmup_num_steps = warmup_steps
    args.total_num_steps = total_steps
    args.warmup_min_lr = 0.0
    args.warmup_max_lr = args.learning_rate

    # Create optimizer
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]

    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters,
        config=args.deepspeed,
    )

    # DeepSpeed manages scheduler internally, no need to create manually
    logger.info("DeepSpeed initialization complete")

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_loss = float('inf')

    for epoch in range(args.num_train_epochs):
        model_engine.train()
        epoch_loss = 0.0
        epoch_metrics = {}

        if local_rank == 0:
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        else:
            pbar = train_dataloader

        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            outputs = model_engine(batch)
            loss = outputs['loss']

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()

            # Update metrics
            epoch_loss += loss.item()
            for key, value in outputs.items():
                if key != 'loss':
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value.item())

            global_step += 1

            # Logging
            if local_rank == 0 and global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                # Get LR from DeepSpeed
                try:
                    lr = model_engine.get_lr()[0]
                except:
                    lr = args.learning_rate

                log_str = f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}"

                # Add metrics
                for key, values in epoch_metrics.items():
                    if values:
                        log_str += f" | {key}: {np.mean(values):.4f}"

                logger.info(log_str)
                if isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': f"{lr:.2e}"})

        # Epoch end
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if local_rank == 0:
            logger.info(f"\nEpoch {epoch + 1} finished | Average Loss: {avg_epoch_loss:.4f}")

            # Log metrics
            for key, values in epoch_metrics.items():
                logger.info(f"  {key}: {np.mean(values):.4f}")

        # Save checkpoint
        if args.save_strategy == 'epoch':
            if local_rank == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
                os.makedirs(save_dir, exist_ok=True)

                # Save model
                model_engine.save_checkpoint(save_dir)
                logger.info(f"Checkpoint saved to {save_dir}")

                # Save best model
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(best_dir, exist_ok=True)
                    model_engine.save_checkpoint(best_dir)
                    logger.info(f"Best model saved to {best_dir}")

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
