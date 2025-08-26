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
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from logging import getLogger

from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, all_gather, nce_loss
from REC.model.HLLM.modeling_llama import LlamaForCausalLM
from REC.model.HLLM.modeling_mistral import MistralForCausalLM
from REC.model.HLLM.modeling_bert import BertModel
from REC.model.HLLM.baichuan.modeling_baichuan import BaichuanForCausalLM
from REC.model.HLLM.mixer import MLPMixer


class HLLM_Creator(BaseModel):
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(HLLM_Creator, self).__init__()
        self.logger = getLogger()
        self.config = config
        self.user_emb_type = config['user_emb_type']

        self.gradient_checkpointing = config['gradient_checkpointing']
        self.use_ft_flash_attn = config['use_ft_flash_attn']

        self.logger.info(f"create user llm")
        self.user_pretrain_dir = config['user_pretrain_dir']
        self.user_llm = self.create_llm(self.user_pretrain_dir, config['user_llm_init'])

        if self.user_emb_type == 'id_seq':
            self.item_embedding = nn.Embedding(
                600000, self.user_llm.config.hidden_size, padding_idx=0
            )
            self.item_embedding.weight.data.normal_(mean=0.0, std=0.02)
        elif self.user_emb_type is None:
            self.logger.info(f"create item llm")
            self.item_pretrain_dir = config['item_pretrain_dir']
            self.item_llm = self.create_llm(
                self.item_pretrain_dir, config['item_llm_init']
            )

        self.logger.info(f"create creative llm")
        self.creative_pretrain_dir = config['creative_pretrain_dir']
        self.creative_llm = self.create_llm(
            self.creative_pretrain_dir, config['creative_llm_init']
        )
        self.logger.info(f"create decoder llm")
        self.decoder_pretrain_dir = config['decoder_pretrain_dir']
        if self.decoder_pretrain_dir:
            self.decoder_llm = self.create_llm(
                self.decoder_pretrain_dir, config['decoder_llm_init']
            )
        else:
            self.logger.info(f"decoder and creative share llm")
            self.decoder_llm = self.creative_llm

        self.item_emb_token_n = config['item_emb_token_n']
        self.emb_token_n = config['emb_token_n']
        if self.item_emb_token_n > 1 or self.emb_token_n > 1:
            raise NotImplementedError(
                f"Not support item_emb_token_n {self.item_emb_token_n} > 1 or emb_token_n {self.emb_token_n} > 1"
            )

        if self.item_emb_token_n > 0:
            self.item_emb_tokens = nn.Parameter(
                torch.zeros(1, self.item_emb_token_n, self.item_llm.config.hidden_size)
            )
            self.item_emb_tokens.data.normal_(mean=0.0, std=0.02)
        else:  # mean pooling
            self.item_emb_tokens = None

        if self.emb_token_n > 0:
            self.emb_tokens = nn.Parameter(
                torch.zeros(1, self.emb_token_n, self.user_llm.config.hidden_size)
            )
            self.emb_tokens.data.normal_(mean=0.0, std=0.02)
        else:  # mean pooling
            self.emb_tokens = None

        self.user_projector = nn.Linear(
            self.user_llm.config.hidden_size,
            self.creative_llm.config.hidden_size,
            bias=False,
        )

        if self.config['aux_loss'] is not None and self.config['aux_loss'] != '':
            self.user_profile_projector = nn.Linear(
                self.user_llm.config.hidden_size,
                self.decoder_llm.config.hidden_size,
                bias=False,
            )
            self.user_profile_align_projector = nn.Linear(
                self.user_llm.config.hidden_size,
                self.decoder_llm.config.hidden_size,
                bias=False,
            )
            if 'align' in self.config['aux_loss']:
                self.align_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.05))
                self.nce_loss_fn = nn.CrossEntropyLoss()
                self.dec_emb_tokens = nn.Parameter(
                    torch.zeros(
                        1, self.emb_token_n, self.decoder_llm.config.hidden_size
                    )
                )
                self.dec_emb_tokens.data.normal_(mean=0.0, std=0.02)
            if 'cls' in self.config['aux_loss']:
                self.cls_loss_fn = nn.BCEWithLogitsLoss()
                self.cls_target_dim = self.config['cls_dim']
                self.user_tower = nn.Linear(
                    self.creative_llm.config.hidden_size,
                    self.cls_target_dim,
                    bias=False,
                )
                self.item_tower = nn.Linear(
                    self.item_llm.config.hidden_size, self.cls_target_dim, bias=False
                )
                if self.config['cls_block_type'] == 'linear':
                    self.cls_tower = nn.Sequential(
                        nn.Linear(
                            self.cls_target_dim * 2, self.cls_target_dim, bias=True
                        ),
                        nn.ReLU(),
                        nn.Linear(self.cls_target_dim, 1, bias=True),
                    )
                elif self.config['cls_block_type'] == 'mixer':
                    self.cls_tower = MLPMixer(
                        dim=self.cls_target_dim,
                        seq_lens=2,
                        num_classes=1,
                        depth=self.config.get('cls_block_num', 1),
                        token_dim=8,
                        channel_dim=self.cls_target_dim * 3,
                    )
                else:
                    raise NotImplementedError(
                        f"Not support cls_block_type {self.config['cls_block_type']}"
                    )

    def create_llm(self, pretrain_dir, init=True):
        self.logger.info(f"******* create LLM {pretrain_dir} *******")
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        self.logger.info(f"hf_config: {hf_config}")
        hf_config.gradient_checkpointing = self.gradient_checkpointing
        hf_config.use_cache = False
        hf_config.output_hidden_states = True
        hf_config.return_dict = True

        self.logger.info("xxxxx starting loading checkpoint")
        if isinstance(hf_config, transformers.LlamaConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(
                f'Using flash attention {hf_config.use_ft_flash_attn} for llama'
            )
            self.logger.info(f'Init {init} for llama')
            if init:
                return LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return LlamaForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.MistralConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(
                f'Using flash attention {hf_config.use_ft_flash_attn} for mistral'
            )
            self.logger.info(f'Init {init} for mistral')
            if init:
                return MistralForCausalLM.from_pretrained(
                    pretrain_dir, config=hf_config
                )
            else:
                return MistralForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.BertConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(
                f'Using flash attention {hf_config.use_ft_flash_attn} for bert'
            )
            self.logger.info(f'Init {init} for bert')
            if init:
                return BertModel.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BertModel(config=hf_config).cuda()
        elif getattr(hf_config, "model_type", None) == "baichuan":
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(
                f'Using flash attention {hf_config.use_ft_flash_attn} for baichuan'
            )
            self.logger.info(f'Init {init} for baichuan')
            if init:
                return BaichuanForCausalLM.from_pretrained(
                    pretrain_dir, config=hf_config
                )
            else:
                return BaichuanForCausalLM(config=hf_config).cuda()
        else:
            return AutoModelForCausalLM.from_pretrained(pretrain_dir, config=hf_config)

    def forward_item_emb(
        self, input_ids, position_ids, cu_input_lens, emb_token_n, emb_tokens, llm
    ):
        inputs_embeds = llm.get_input_embeddings()(input_ids)
        emb_pos = cu_input_lens.cumsum(dim=0, dtype=torch.int32)
        if emb_token_n > 0:
            inputs_embeds[emb_pos - 1] = emb_tokens
        model_out = llm(
            inputs_embeds=inputs_embeds.unsqueeze(0),
            cu_input_lens=cu_input_lens,
            position_ids=position_ids.unsqueeze(0),
        )
        model_out = model_out.hidden_states[-1].squeeze(0)

        if emb_token_n > 0:
            emb = model_out[emb_pos - 1]
        else:
            max_len = cu_input_lens.max().item()
            cu_seqlens = F.pad(cu_input_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
            seqs = [
                model_out[start:end]
                for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
            ]
            padded_seqs = [
                F.pad(
                    seqs[i],
                    (0, 0) * (seqs[i].dim() - 1) + (0, max_len - cu_input_lens[i]),
                    value=0.0,
                )
                for i in range(cu_input_lens.size(0))
            ]
            out = torch.stack(padded_seqs)
            emb = out.sum(dim=1) / cu_input_lens.unsqueeze(1)

        return emb

    def forward(self, interaction):
        model_out = {}
        user_prompt_ids, attention_mask = (
            interaction['user_prompt_ids'],
            interaction['user_attention_mask'],
        )
        prompt_ids, user_pos, input_mask, target_mask = (
            interaction['prompt_ids'],
            interaction['user_pos'],
            interaction['input_mask'],
            interaction['target_mask'],
        )
        N, S = attention_mask.size()
        if self.user_emb_type == 'text_seq':
            user_inputs_embeds = self.user_llm.get_input_embeddings()(user_prompt_ids)
        else:
            if self.user_emb_type == 'id_seq':
                input_id_list, seq_mask = (
                    interaction['input_id_list'],
                    interaction['seq_mask'],
                )
                seq_item_emb = self.item_embedding(input_id_list)
            else:
                seq_item_input_ids, cu_input_lens, seq_item_position_ids, seq_mask = (
                    interaction['seq_item_input_ids'],
                    interaction['cu_input_lens'],
                    interaction['seq_item_position_ids'],
                    interaction['seq_mask'],
                )

                seq_item_emb = self.forward_item_emb(
                    seq_item_input_ids,
                    seq_item_position_ids,
                    cu_input_lens,
                    self.item_emb_token_n,
                    self.item_emb_tokens,
                    self.item_llm,
                )
                seq_item_emb = seq_item_emb.reshape(
                    N, -1, self.item_llm.config.hidden_size
                )

            user_inputs_embeds = self.user_llm.get_input_embeddings()(user_prompt_ids)
            user_inputs_embeds = torch.cat((seq_item_emb, user_inputs_embeds), dim=1)
            attention_mask = torch.cat((seq_mask, attention_mask), dim=1)

        user_inputs_embeds = torch.cat(
            (user_inputs_embeds, self.emb_tokens.expand(N, -1, -1)), dim=1
        )
        attention_mask = torch.cat(
            (attention_mask, attention_mask.new_ones(N, self.emb_token_n)),
            dim=1,
        )
        seq_user_emb = self.user_llm(
            inputs_embeds=user_inputs_embeds, attention_mask=attention_mask
        ).hidden_states[-1][:, -self.emb_token_n :]
        seq_user_emb_proj = self.user_projector(seq_user_emb)

        inputs_embeds = self.creative_llm.get_input_embeddings()(prompt_ids)
        row_idx = torch.arange(N).repeat_interleave(self.emb_token_n)
        col_idx = user_pos.flatten()
        inputs_embeds[row_idx, col_idx] = seq_user_emb_proj.reshape(
            -1, self.creative_llm.config.hidden_size
        )
        labels = prompt_ids * target_mask + (1 - target_mask) * (-100)

        creative_llm_out = self.creative_llm(
            inputs_embeds=inputs_embeds, attention_mask=input_mask, labels=labels
        )
        model_out['loss'] = creative_llm_out['loss']
        model_out['creative_loss'] = model_out['loss']

        if 'recon' in self.config['aux_loss']:
            user_profile_ids, user_profile_pos, user_profile_mask, user_profile_target_mask = (
                interaction['user_profile_ids'],
                interaction['user_profile_pos'],
                interaction['user_profile_mask'],
                interaction['user_profile_target_mask'],
            )
            user_profile_inputs_embeds = self.decoder_llm.get_input_embeddings()(
                user_profile_ids
            )
            col_idx = user_profile_pos.flatten()
            seq_user_emb_recon = self.user_profile_projector(seq_user_emb)
            user_profile_inputs_embeds[row_idx, col_idx] = seq_user_emb_recon.reshape(
                -1, self.decoder_llm.config.hidden_size
            )
            user_profile_labels = user_profile_ids * user_profile_target_mask + (
                1 - user_profile_target_mask
            ) * (-100)
            user_profile_model_out = self.decoder_llm(
                inputs_embeds=user_profile_inputs_embeds,
                attention_mask=user_profile_mask,
                labels=user_profile_labels,
            )
            model_out['recon_loss'] = user_profile_model_out['loss']
            model_out['loss'] = model_out['loss'] + model_out['recon_loss']
        if 'cls' in self.config['aux_loss']:
            user_profile_cls_ids, user_profile_cls_mask = (
                interaction['user_profile_cls_ids'],
                interaction['user_profile_cls_mask'],
            )
            user_profile_cls_inputs_embeds = self.item_llm.get_input_embeddings()(
                user_profile_cls_ids
            )
            user_profile_cls_inputs_embeds = torch.cat(
                (
                    user_profile_cls_inputs_embeds,
                    self.item_emb_tokens.expand(N, -1, -1),
                ),
                dim=1,
            )
            user_profile_cls_mask = torch.cat(
                (
                    user_profile_cls_mask,
                    user_profile_cls_mask.new_ones(N, self.item_emb_token_n),
                ),
                dim=1,
            )
            cls_item_emb = self.item_llm(
                inputs_embeds=user_profile_cls_inputs_embeds,
                attention_mask=user_profile_cls_mask,
            ).hidden_states[-1][:, -self.item_emb_token_n :]
            cls_user_emb = self.user_tower(seq_user_emb_proj).mean(dim=1)
            cls_item_emb = self.item_tower(cls_item_emb).mean(dim=1)
            cls_item_emb_broadcast = all_gather(cls_item_emb, sync_grads=True)
            mask = torch.arange(dist.get_world_size()) != dist.get_rank()
            item_emb_neg = cls_item_emb_broadcast[mask].reshape(-1, self.cls_target_dim)
            indices = torch.stack(
                [
                    torch.randperm(item_emb_neg.size(0))[: self.config['cls_neg_num']]
                    for _ in range(cls_item_emb.size(0))
                ],
                dim=0,
            )
            sampled_b = item_emb_neg[indices]
            cls_item_emb = cls_item_emb.unsqueeze(1)
            cls_item_emb = torch.cat([cls_item_emb, sampled_b], dim=1).reshape(
                -1, cls_item_emb.size(-1)
            )
            label = (
                cls_user_emb.new_tensor([1] + [0] * self.config['cls_neg_num'])
                .repeat(cls_user_emb.size(0))
                .reshape(-1, 1)
            )
            cls_user_emb = cls_user_emb.repeat_interleave(
                self.config['cls_neg_num'] + 1, dim=0
            )
            if self.config['cls_block_type'] == 'linear':
                logits = self.cls_tower(torch.cat((cls_user_emb, cls_item_emb), dim=-1))
            else:
                logits = self.cls_tower(
                    torch.cat(
                        [cls_user_emb.unsqueeze(1), cls_item_emb.unsqueeze(1)], dim=1
                    )
                ).reshape(-1, 1)

            model_out['cls_loss'] = self.cls_loss_fn(logits, label)
            model_out['logits'] = logits
            model_out['label'] = label
            model_out['loss'] = model_out['loss'] + model_out['cls_loss']

        if 'align' in self.config['aux_loss']:
            user_profile_align_ids, user_profile_align_mask = (
                interaction['user_profile_align_ids'],
                interaction['user_profile_align_mask'],
            )
            user_profile_inputs_embeds = self.decoder_llm.get_input_embeddings()(
                user_profile_align_ids
            )
            user_profile_inputs_embeds = torch.cat(
                (user_profile_inputs_embeds, self.dec_emb_tokens.expand(N, -1, -1)),
                dim=1,
            )
            user_profile_align_mask = torch.cat(
                (
                    user_profile_align_mask,
                    user_profile_align_mask.new_ones(N, self.emb_token_n),
                ),
                dim=1,
            )
            user_profile_emb = self.decoder_llm(
                inputs_embeds=user_profile_inputs_embeds,
                attention_mask=user_profile_align_mask,
            ).hidden_states[-1][
                :, -self.emb_token_n :
            ]  # (bs, dim)
            user_profile_emb = user_profile_emb / user_profile_emb.norm(
                dim=-1, keepdim=True
            )  # (bs, n, dim)
            user_profile_emb = user_profile_emb.reshape(
                -1, self.decoder_llm.config.hidden_size
            )
            seq_user_emb_align = self.user_profile_align_projector(
                seq_user_emb
            ).reshape(-1, self.decoder_llm.config.hidden_size)
            seq_user_emb_align = seq_user_emb_align / seq_user_emb_align.norm(
                dim=-1, keepdim=True
            )  # (bs, n, dim)

            align_loss1 = self.nce_loss_fn(
                *nce_loss(seq_user_emb_align, user_profile_emb, self.align_temp)
            )
            align_loss2 = self.nce_loss_fn(
                *nce_loss(user_profile_emb, seq_user_emb_align, self.align_temp)
            )
            model_out['align_loss'] = (align_loss1 + align_loss2) / 2
            model_out['loss'] = model_out['loss'] + model_out['align_loss']

        return model_out
