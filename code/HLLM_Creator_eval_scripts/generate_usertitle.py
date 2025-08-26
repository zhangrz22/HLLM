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

os.environ['VLLM_USE_V1'] = '0'

import pandas as pd
import argparse
import tqdm

import torch
from transformers import AutoTokenizer
from REC.config import Config
from REC.model.HLLM.hllm_creator import HLLM_Creator
from REC.data.dataset.trainset import CreatorProcessor
from REC.data.dataset.collate_fn import customize_rmpad_collate

parser = argparse.ArgumentParser(description="Run LLM with configurable parameters.")
parser.add_argument(
    "--ckpt_model_path", type=str, default="model/", help="Path to the model"
)
parser.add_argument("--creative_tokenizer_path", type=str)
parser.add_argument(
    "--data_path", type=str, default="model/", help="Path to the input dataset"
)
parser.add_argument(
    "--output_path", type=str, default="model/", help="Path to the output emb"
)
parser.add_argument(
    "--temperature", type=float, default=0.8, help="Sampling temperature"
)
parser.add_argument(
    "--top_p", type=float, default=0.95, help="Top-p sampling parameter"
)
parser.add_argument("--top_k", type=int, default=50, help="Top-p sampling parameter")
parser.add_argument(
    "--max_tokens", type=int, default=128, help="Maximum number of tokens to generate"
)
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--limit", type=int, default=5000000000)
parser.add_argument('--config_file', nargs='+', type=str)
parser.add_argument('--random_sample', action='store_true')
parser.add_argument('--cluster_path', type=str, default=None)
parser.add_argument('--cluster_norm', action='store_true')
args, extra_args = parser.parse_known_args()
if args.cluster_path is not None:
    cluster = torch.load(args.cluster_path).to('cuda:0').bfloat16()
    print(f"{cluster.size() = }")
    cluster_norm = torch.nn.functional.normalize(cluster, dim=-1)


def forward(
    hllm,
    seq_item_input_ids=None,
    seq_mask=None,
    cu_input_lens=None,
    seq_item_position_ids=None,
    user_prompt_ids=None,
    user_attention_mask=None,
    prompt_ids=None,
    user_pos=None,
    input_mask=None,
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

    if args.cluster_path is not None:
        if args.cluster_norm:
            seq_user_emb = torch.nn.functional.normalize(seq_user_emb, dim=-1)
            sim = torch.matmul(seq_user_emb, cluster_norm.t())
            print(f"{sim.size() = }")
            seq_user_emb = cluster[sim.argmax(dim=1)]
        else:
            sim = torch.matmul(seq_user_emb, cluster.t())
            print(f"{sim.size() = }")
            seq_user_emb = cluster[sim.argmax(dim=1)]

    inputs_embeds = hllm.creative_llm.get_input_embeddings()(prompt_ids)
    row_idx = torch.arange(N).repeat_interleave(hllm.emb_token_n)
    col_idx = user_pos.flatten()
    inputs_embeds[row_idx, col_idx] = seq_user_emb.reshape(
        -1, hllm.creative_llm.config.hidden_size
    )
    return inputs_embeds, input_mask


def main():
    llm_tokenizer = AutoTokenizer.from_pretrained(args.creative_tokenizer_path)

    config = Config(config_file_list=args.config_file)
    if len(extra_args):
        config.parse_extra_args(extra_args)

    processor = CreatorProcessor(config)

    hllm = HLLM_Creator(config, dataload=None)
    hllm_ckpt = torch.load(f'{args.ckpt_model_path}')
    msg = hllm.load_state_dict(hllm_ckpt, strict=False)
    print(f"{msg.missing_keys = }")
    print(f"{msg.unexpected_keys = }")
    hllm.eval()
    hllm = hllm.to(f'cuda:0').bfloat16().eval()

    df = pd.read_parquet(args.data_path).head(args.limit)
    input_embeds, input_mask, bs_data = [], [], []
    with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            row = row.to_dict()
            cur_data = processor.process(row, eval=True)
            if cur_data is None:
                continue
            bs_data.append(cur_data)
            if len(bs_data) >= args.batch_size:
                bs_data = customize_rmpad_collate(bs_data)
                bs_data = {k: v.to(f'cuda:0') for k, v in bs_data.items()}
                result, mask = forward(hllm, **bs_data)
                input_embeds.append(result)
                input_mask.append(mask)
                bs_data = []
        if len(bs_data) > 0:
            bs_data = customize_rmpad_collate(bs_data)
            bs_data = {k: v.to(f'cuda:0') for k, v in bs_data.items()}
            result, mask = forward(hllm, **bs_data)
            input_embeds.append(result)
            input_mask.append(mask)

    llm_output = []
    for embeds, mask in tqdm.tqdm(
        zip(input_embeds, input_mask), total=len(input_embeds)
    ):
        with torch.no_grad(), torch.amp.autocast(
            dtype=torch.bfloat16, device_type="cuda"
        ):
            output = hllm.creative_llm.generate(
                inputs_embeds=embeds.to('cuda:0'),
                attention_mask=mask.to('cuda:0'),
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            output = llm_tokenizer.batch_decode(output, skip_special_tokens=True)
        llm_output.extend(output)

    llm_output = pd.DataFrame({'llm_output': llm_output})
    df_combined = pd.concat([df, llm_output], axis=1)
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_combined.to_parquet(f'{args.output_path}', index=False)


if __name__ == "__main__":
    main()
