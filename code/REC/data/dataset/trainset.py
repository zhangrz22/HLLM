# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from asyncio.log import logger
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import random
import datetime
import pytz
import math
import torch.distributed as dist
import json

# 数据形式为 [[user_seq], [neg_item_seq]] , [mask]


class SEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        self.train_seq = dataload.train_feat['item_seq']

        self.length = len(self.train_seq)

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
        self.num_negatives = config['num_negatives']
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])
        logger.info(f"Use random sample {self.random_sample} for mask id")

    def __len__(self):
        return self.length

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        pad_len = max_length - len(sequence)
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        if self.num_negatives:
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample(item_seq))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
        return torch.as_tensor(item_seq, dtype=torch.int64), torch.as_tensor(neg_item, dtype=torch.int64), torch.as_tensor(masked_index, dtype=torch.int64)

    def __getitem__(self, index):
        # 最长长度为maxlen+1, 及若max_len是5
        # 则存在    1,2,3,4,5,6序列,
        # pos       2,3,4,5,6
        # neg       0,8,9,7,9,8
        # mask_index 1,1,1,1,1
        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)

        return item_seq, neg_item, masked_index


class TextSEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        self.train_seq = dataload.train_feat['item_seq']
        self.length = len(self.train_seq)
        self.train_time_seq = dataload.train_feat['time_seq']
        self.id2token = dataload.id2token['item_id']

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']

        self.text_path = config['text_path']
        self.text_keys = config['text_keys']
        self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)
        # self.pad_id = self.tokenizer.pad_token_id
        # assert self.pad_id is not None, f"pad_token_id can't be {self.pad_id}"
        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.num_negatives = config['num_negatives']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])  # for llm only
        logger.info(f"Use random sample {self.random_sample} for mask id")
        logger.info(f"Text path: {self.text_path}")
        logger.info(f"Text keys: {self.text_keys}")
        logger.info(f"Item prompt: {self.item_prompt}")
        self.load_content()

    def __len__(self):
        return self.length

    def load_content(self):
        self.env = pd.read_csv(self.text_path, delimiter=',', dtype={'item_id': str})
        self.env = self.env[self.text_keys + ['item_id']]
        self.env = self.env.set_index('item_id').T.to_dict()
        logger.info(f"Text Item num: {len(self.env)}")

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        pad_len = max_length - len(sequence)
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
        if self.num_negatives:
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample([]))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        return item_seq, neg_item, masked_index

    def _padding_time_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        vq_time = []
        for time in sequence:
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return torch.tensor(vq_time, dtype=torch.long)

    def __getitem__(self, index):

        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)
        time_seq = self.train_time_seq[index]
        time_seq = self._padding_time_sequence(list(time_seq), self.max_seq_length)
        item_seq_token = self.id2token[item_seq]
        neg_items_token = self.id2token[neg_item]
        pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
        neg_input_ids, neg_cu_input_lens, neg_position_ids = [], [], []

        def process_item(item):
            if item != self.id2token[0] and item not in self.env:
                # assert item in self.env, f"{item}"
                logger.info(f"{item} not in self.env")
            item_i = self.env.get(item, {})
            text_str = ""
            if len(item_i):
                text_str = f"{self.item_prompt}"
                for key in self.text_keys:
                    value = item_i[key]
                    if value and str(value) != 'nan':
                        text_str += f"{key}: {value}"

            ids = self.tokenizer.encode(text_str)
            ids = ids[:self.max_text_length]
            mask = [1] * len(ids)
            return ids, mask

        for item in item_seq_token:
            ids, _ = process_item(item)
            pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
            pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
            pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())

        for neg in neg_items_token:
            ids, _ = process_item(neg)
            neg_input_ids.extend(ids + [0] * self.item_emb_token_n)
            neg_cu_input_lens.append(len(ids) + self.item_emb_token_n)
            neg_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())

        outputs = {
            "pos_item_ids": torch.as_tensor(item_seq, dtype=torch.int64),
            "neg_item_ids": torch.as_tensor(neg_item, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
            "neg_input_ids": torch.as_tensor(neg_input_ids, dtype=torch.int64),
            "neg_cu_input_lens": torch.as_tensor(neg_cu_input_lens, dtype=torch.int64),
            "neg_position_ids": torch.as_tensor(neg_position_ids, dtype=torch.int64),
            "attention_mask": torch.as_tensor(masked_index, dtype=torch.int64),
            "time_ids": torch.as_tensor(time_seq, dtype=torch.int64),
        }
        return outputs
    
class CreatorProcessor:
    def __init__(self, config):
        self.config = config
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.max_gen_text_length = config.get('MAX_GEN_TEXT_LENGTH', 4096)
        self.max_user_profile_text_length = config.get('MAX_USER_PROFILE_TEXT_LENGTH', 1024)

        self.creative_tokenizer = AutoTokenizer.from_pretrained(config['creative_pretrain_dir'], trust_remote_code=True, add_bos_token=False)
        if config['item_pretrain_dir']:
            self.item_tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True, add_bos_token=False)
        if config['decoder_pretrain_dir']:
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(config['decoder_pretrain_dir'], trust_remote_code=True, add_bos_token=False)
        else:
            self.decoder_tokenizer = self.creative_tokenizer
        self.eos_id = self.creative_tokenizer.eos_token_id
        self.pad_id = self.creative_tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_id

        self.item_pad_id = self.item_tokenizer.pad_token_id
        if self.item_pad_id is None:
            self.item_pad_id = self.item_tokenizer.eos_token_id
        
        self.decoder_pad_id = self.decoder_tokenizer.pad_token_id
        if self.decoder_pad_id is None:
            self.decoder_pad_id = self.decoder_tokenizer.eos_token_id

        self.item_prompt = config['item_prompt']
        self.item_prompt_ids = self.item_tokenizer.encode(self.item_prompt)
        self.user_prompt = config['user_prompt']
        self.user_prompt_ids = self.item_tokenizer.encode(self.user_prompt)
        self.user_profile_prompt = config['user_profile_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.emb_token_n = config['emb_token_n']
        self.aux_loss = config['aux_loss']
        self.user_emb_type = config['user_emb_type']
        self.text_key = config.get('text_key', 'title_list')
        logger.info(f"Item prompt: {self.item_prompt}")
        logger.info(f"User prompt: {self.user_prompt}")
        logger.info(f"Text key: {self.text_key}")
    
    def process(self, data_dict, eval=False):
        input_strs = list(data_dict[self.text_key][-self.max_seq_length:])
        user_prompt_ids = self.user_prompt_ids
        user_attention_mask = [1] * len(self.user_prompt_ids)
        user_profile = json.dumps(
            json.loads(data_dict.get('user_profile', '{}')), ensure_ascii=False
        )
        if self.user_emb_type == 'text_seq':
            user_inputs_ids = []
            for input_str in input_strs:
                input_str_ids = self.item_tokenizer.encode(input_str + '\n')[:self.max_text_length]
                user_inputs_ids.extend(input_str_ids)
            user_inputs_ids = user_inputs_ids[:self.max_gen_text_length]
            user_ids_pad = self.max_gen_text_length - len(user_inputs_ids)
            user_prompt_ids = [self.item_pad_id] * user_ids_pad + user_inputs_ids + user_prompt_ids
            user_attention_mask = [0] * user_ids_pad + [1] * len(user_inputs_ids) + user_attention_mask
        elif self.user_emb_type == 'id_seq':
            input_id_list = list(data_dict['input_id_list'][-self.max_seq_length:])
            input_id_list = [x+1 for x in input_id_list]
            seq_mask = [0] * (self.max_seq_length - len(input_id_list)) + [1] * len(input_id_list)
            input_id_list = [0] * (self.max_seq_length - len(input_id_list)) + input_id_list
        else:
            seq_mask, item_input_ids = [1] * len(input_strs), []

            if len(input_strs) < self.max_seq_length:
                seq_pad_len = self.max_seq_length - len(input_strs)
                seq_mask = [0] * seq_pad_len + seq_mask
                input_strs = [""] * seq_pad_len + input_strs

            cu_input_lens, input_position_ids = [], []
            for input_str in input_strs:
                input_str_ids = self.item_tokenizer.encode(input_str)[:self.max_text_length]
                item_input_ids.extend(input_str_ids + [0] * self.item_emb_token_n)
                cu_input_lens.append(len(input_str_ids) + self.item_emb_token_n)
                input_position_ids.extend((torch.arange(len(input_str_ids) + self.item_emb_token_n) + (self.max_text_length - len(input_str_ids))).tolist())

        prompt1_ids = self.creative_tokenizer.encode(data_dict['prompt1']) + [0] * self.emb_token_n
        prompt2_ids = self.creative_tokenizer.encode(data_dict['prompt2'])

        input_mask, target_mask = [], []
        input_mask += [1] * (len(prompt1_ids) + len(prompt2_ids))
        target_mask += [0] * (len(prompt1_ids) + len(prompt2_ids))
        response_max_len = self.max_gen_text_length - len(input_mask)
        if response_max_len < 0:
            return None

        if eval:
            response_ids = []
        else:
            response_ids = self.creative_tokenizer.encode(data_dict['response'])
            response_ids = response_ids[: response_max_len - 1] + [self.eos_id]
        input_mask += [1] * len(response_ids)
        target_mask += [1] * len(response_ids)
        input_ids = prompt1_ids + prompt2_ids + response_ids

        pad_len = self.max_gen_text_length - len(input_mask)
        input_ids = [self.pad_id] * pad_len + input_ids
        input_mask = [0] * pad_len + input_mask
        target_mask = [0] * pad_len + target_mask
        user_pos = [pad_len + len(prompt1_ids) - i for i in range(1, self.emb_token_n + 1)]

        outputs = {
            "user_prompt_ids": torch.as_tensor(user_prompt_ids, dtype=torch.int64),
            "user_attention_mask": torch.as_tensor(user_attention_mask, dtype=torch.int64),
            "prompt_ids": torch.as_tensor(input_ids, dtype=torch.int64),
            "user_pos": torch.as_tensor(user_pos, dtype=torch.int64),
            "input_mask": torch.as_tensor(input_mask, dtype=torch.int64),
            "target_mask": torch.as_tensor(target_mask, dtype=torch.int64),
        }

        if self.user_emb_type == 'id_seq':
            outputs['input_id_list'] = torch.as_tensor(input_id_list, dtype=torch.int64)
            outputs['seq_mask'] = torch.as_tensor(seq_mask, dtype=torch.int64)
        elif self.user_emb_type is None:
            outputs['seq_item_input_ids'] = torch.as_tensor(item_input_ids, dtype=torch.int64)
            outputs['cu_input_lens'] = torch.as_tensor(cu_input_lens, dtype=torch.int64)
            outputs['seq_item_position_ids'] = torch.as_tensor(input_position_ids, dtype=torch.int64)
            outputs['seq_mask'] = torch.as_tensor(seq_mask, dtype=torch.int64)

        
        if not eval:
            if 'recon' in self.aux_loss:
                user_profile_prompt_ids = [0] * self.emb_token_n + self.decoder_tokenizer.encode(self.user_profile_prompt)
                user_profile_attention_mask = [1] * len(user_profile_prompt_ids)
                user_profile_target_mask = [0] * len(user_profile_prompt_ids)
                user_profile_response_max_len = self.max_user_profile_text_length - len(user_profile_attention_mask)

                user_profile_response_ids = self.decoder_tokenizer.encode(user_profile)[: user_profile_response_max_len-1] + [self.decoder_tokenizer.eos_token_id]
                user_profile_attention_mask += [1] * len(user_profile_response_ids)
                user_profile_target_mask += [1] * len(user_profile_response_ids)

                user_profile_ids = user_profile_prompt_ids + user_profile_response_ids
                pad_len = self.max_user_profile_text_length - len(user_profile_attention_mask) # hard code
                outputs['user_profile_ids'] = torch.as_tensor([self.decoder_pad_id] * pad_len + user_profile_ids, dtype=torch.int64)
                outputs['user_profile_pos'] = pad_len + torch.arange(self.emb_token_n, dtype=torch.int64)
                outputs['user_profile_mask'] = torch.as_tensor([0] * pad_len + user_profile_attention_mask, dtype=torch.int64)
                outputs['user_profile_target_mask'] = torch.as_tensor([0] * pad_len + user_profile_target_mask, dtype=torch.int64)
            if 'align' in self.aux_loss:
                user_profile_response_ids = self.decoder_tokenizer.encode(user_profile)[:self.max_user_profile_text_length]
                user_profile_attention_mask = [1] * len(user_profile_response_ids)
                pad_len = self.max_user_profile_text_length - len(user_profile_attention_mask) # hard code
                outputs['user_profile_align_ids'] = torch.as_tensor([self.decoder_pad_id] * pad_len + user_profile_response_ids, dtype=torch.int64)
                outputs['user_profile_align_mask'] = torch.as_tensor([0] * pad_len + user_profile_attention_mask, dtype=torch.int64)
            if 'cls' in self.aux_loss:
                user_profile_response_ids = self.item_tokenizer.encode(data_dict['original_title'])[:self.max_text_length]
                user_profile_attention_mask = [1] * len(user_profile_response_ids)
                pad_len = self.max_text_length - len(user_profile_attention_mask)
                outputs['user_profile_cls_ids'] = torch.as_tensor([self.item_pad_id] * pad_len + user_profile_response_ids, dtype=torch.int64)
                outputs['user_profile_cls_mask'] = torch.as_tensor([0] * pad_len + user_profile_attention_mask, dtype=torch.int64)

        return outputs
    
class TextSEQCreatorTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.train_path = config['train_path']
        df = pd.read_parquet(self.train_path)
        self.env = df.to_dict('records')
        logger.info(f"Train num: {len(self.env)}")
        self.length = len(self.env)
        self.device = config['device']
        self.processor = CreatorProcessor(config)

        logger.info(f"Train path: {self.train_path}")


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.processor.process(self.env[index])
