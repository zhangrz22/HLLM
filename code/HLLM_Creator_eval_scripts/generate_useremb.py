# Copyright 2025 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import copy

import numpy as np
import pandas as pd
import argparse
import json
import tqdm

import torch
import torch.multiprocessing as mp
from REC.config import Config
from REC.model.HLLM.hllm_creator import HLLM_Creator
from REC.data.dataset.trainset import CreatorProcessor
from REC.data.dataset.collate_fn import customize_rmpad_collate

parser = argparse.ArgumentParser(description="Run LLM with configurable parameters.")
parser.add_argument(
    "--model_path", type=str, default="model/", help="Path to the model"
)
parser.add_argument(
    "--data_path", type=str, default="model/", help="Path to the dataset"
)
parser.add_argument(
    "--output_path", type=str, default="model/", help="Path to the output emb"
)
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--num_gpus", type=int, default=8, help="Num gpus use")
parser.add_argument("--limit", type=int, default=5000000000)
parser.add_argument('--config_file', nargs='+', type=str)
parser.add_argument('--random_sample', action='store_true')
args, extra_args = parser.parse_known_args()


def forward(
    hllm,
    seq_item_input_ids=None,
    seq_mask=None,
    cu_input_lens=None,
    seq_item_position_ids=None,
    user_prompt_ids=None,
    user_attention_mask=None,
    input_id_list=None,
    **kwargs,
):
    N, S = user_attention_mask.size()
    if hllm.user_emb_type == 'text_seq':
        inputs_embeds = hllm.user_llm.get_input_embeddings()(user_prompt_ids)
        attention_mask = user_attention_mask
    else:
        if hllm.user_emb_type == 'id_seq':
            seq_item_emb = hllm.item_embedding(input_id_list)
        else:
            seq_item_emb = hllm.forward_item_emb(
                seq_item_input_ids,
                seq_item_position_ids,
                cu_input_lens,
                hllm.item_emb_token_n,
                hllm.item_emb_tokens,
                hllm.item_llm,
            )
            seq_item_emb = seq_item_emb.reshape(N, -1, hllm.item_llm.config.hidden_size)

        inputs_embeds = hllm.user_llm.get_input_embeddings()(user_prompt_ids)
        inputs_embeds = torch.cat((seq_item_emb, inputs_embeds), dim=1)
        attention_mask = torch.cat((seq_mask, user_attention_mask), dim=1)

    inputs_embeds = torch.cat((inputs_embeds, hllm.emb_tokens.expand(N, -1, -1)), dim=1)
    attention_mask = torch.cat(
        (attention_mask, attention_mask.new_ones(N, hllm.emb_token_n)), dim=1
    )
    seq_user_emb = hllm.user_llm(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask
    ).hidden_states[-1][:, -hllm.emb_token_n :]
    seq_user_emb = hllm.user_projector(seq_user_emb.squeeze(1)).reshape(
        -1, hllm.creative_llm.config.hidden_size
    )
    return seq_user_emb


def worker(gpu_id, df):
    config = Config(config_file_list=args.config_file)
    if len(extra_args):
        config.parse_extra_args(extra_args)

    processor = CreatorProcessor(config)

    hllm = HLLM_Creator(config, dataload=None)
    hllm_ckpt = torch.load(f'{args.model_path}')
    msg = hllm.load_state_dict(hllm_ckpt, strict=False)
    print(f"{msg.missing_keys = }")
    print(f"{msg.unexpected_keys = }")
    hllm.eval()
    hllm = hllm.to(f'cuda:{gpu_id}').bfloat16().eval()

    input_embeds, bs_data = [], []
    with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        for idx, row in tqdm.tqdm(
            df.iterrows(), total=len(df), position=gpu_id, leave=True, ncols=150
        ):
            row = row.to_dict()
            cur_data = processor.process(row)
            if cur_data is None:
                continue
            bs_data.append(cur_data)
            if len(bs_data) >= args.batch_size:
                bs_data = customize_rmpad_collate(bs_data)
                bs_data = {k: v.to(f'cuda:{gpu_id}') for k, v in bs_data.items()}
                seq_user_emb = forward(hllm, **bs_data)
                input_embeds.append(seq_user_emb.float())
                bs_data = []
        if len(bs_data) > 0:
            bs_data = customize_rmpad_collate(bs_data)
            bs_data = {k: v.to(f'cuda:{gpu_id}') for k, v in bs_data.items()}
            seq_user_emb = forward(hllm, **bs_data)
            input_embeds.append(seq_user_emb.float())
    input_embeds = torch.cat(input_embeds, dim=0)
    print(f"{input_embeds.size() = }")
    torch.save(input_embeds, f"{args.output_path}/rank{gpu_id}_user_emb.pt")


def distribute_work(df, num_gpus):
    df_split = np.array_split(df, num_gpus)

    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, df_split[gpu_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    df = pd.read_parquet(args.data_path)
    if args.random_sample:
        drop_df = df.drop_duplicates(subset=['user_profile']).sample(
            frac=1, random_state=42
        )
    else:
        drop_df = df
    head_df = drop_df.head(args.limit)
    print(f"{len(df) = } {len(drop_df) = } {len(head_df) = }")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    num_gpus = args.num_gpus
    mp.set_start_method('spawn')
    distribute_work(head_df, num_gpus)
