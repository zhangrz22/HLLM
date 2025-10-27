"""
HLLM Model for PixelRec
Extracted and refactored from original HLLM code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM
from logging import getLogger

logger = getLogger(__name__)


class HLLMModel(nn.Module):
    """
    Hierarchical LLM for Sequential Recommendation
    Uses two LLMs: item_llm for encoding items, user_llm for encoding sequences
    """

    def __init__(
        self,
        item_pretrain_dir,
        user_pretrain_dir,
        item_emb_token_n=1,
        num_negatives=None,
        nce_thres=0.99,
        gradient_checkpointing=True,
        use_ft_flash_attn=False,
    ):
        super(HLLMModel, self).__init__()
        self.logger = getLogger()

        self.item_pretrain_dir = item_pretrain_dir
        self.user_pretrain_dir = user_pretrain_dir
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ft_flash_attn = use_ft_flash_attn
        self.item_emb_token_n = item_emb_token_n
        self.num_negatives = num_negatives
        self.nce_thres = nce_thres

        # Create LLMs
        self.logger.info("Creating item LLM...")
        self.item_llm = self.create_llm(self.item_pretrain_dir, init=True)

        self.logger.info("Creating user LLM...")
        self.user_llm = self.create_llm(self.user_pretrain_dir, init=True)

        # Item embedding tokens
        if self.item_emb_token_n > 1:
            raise NotImplementedError(f"Not support item_emb_token_n {self.item_emb_token_n} > 1")

        if self.item_emb_token_n > 0:
            self.item_emb_tokens = nn.Parameter(
                torch.zeros(1, self.item_emb_token_n, self.item_llm.config.hidden_size)
            )
            self.item_emb_tokens.data.normal_(mean=0.0, std=0.02)
        else:
            self.item_emb_tokens = None

        # NCE loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logger.info(f"NCE threshold: {self.nce_thres}")
        self.logger.info(f"Model initialized successfully")

    def create_llm(self, pretrain_dir, init=True):
        """Create LLM from pretrained checkpoint"""
        self.logger.info(f"Loading LLM from {pretrain_dir}")

        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        hf_config.gradient_checkpointing = self.gradient_checkpointing
        hf_config.use_cache = False
        hf_config.output_hidden_states = True
        hf_config.return_dict = True

        self.logger.info(f"Config: {hf_config}")

        if init:
            return AutoModelForCausalLM.from_pretrained(
                pretrain_dir, config=hf_config, trust_remote_code=True
            )
        else:
            return AutoModelForCausalLM(config=hf_config)

    def forward_item_emb(self, input_ids, position_ids, cu_input_lens, emb_token_n, emb_tokens, llm):
        """
        Encode items using item LLM

        Args:
            input_ids: concatenated token ids for all items
            position_ids: position ids
            cu_input_lens: cumulative lengths of each item
            emb_token_n: number of embedding tokens
            emb_tokens: learnable embedding tokens
            llm: the LLM model

        Returns:
            item embeddings
        """
        # Get token embeddings
        inputs_embeds = llm.get_input_embeddings()(input_ids)

        # Insert learnable embedding tokens
        if emb_token_n > 0:
            emb_pos = []
            s = 0
            for l in cu_input_lens:
                emb_pos.extend(list(range(s + l.item() - emb_token_n, s + l.item())))
                s += l.item()
            inputs_embeds[emb_pos] = emb_tokens.expand(len(emb_pos), -1, -1).reshape(-1, emb_tokens.size(-1))

        # Process through LLM in batches
        model_out = llm(inputs_embeds=inputs_embeds.unsqueeze(0), position_ids=position_ids.unsqueeze(0))
        hidden_states = model_out.hidden_states[-1].squeeze(0)

        # Extract embeddings (mean pooling or last token)
        if emb_token_n > 0:
            # Use embedding token positions
            emb = hidden_states[emb_pos]
            if emb_token_n > 1:
                emb = emb.reshape(-1, emb_token_n, emb.size(-1)).mean(dim=1)
        else:
            # Mean pooling
            padded_seqs = []
            start = 0
            for length in cu_input_lens:
                end = start + length.item()
                padded_seqs.append(hidden_states[start:end])
                start = end

            # Pad and average
            max_len = max(len(seq) for seq in padded_seqs)
            out = torch.stack([
                F.pad(seq, (0, 0, 0, max_len - len(seq)))
                for seq in padded_seqs
            ])
            emb = out.sum(dim=1) / cu_input_lens.unsqueeze(1)

        return emb

    def nce_loss(self, cur_embs, target_pos, target_neg, user_attention_mask):
        """
        Compute NCE loss

        Args:
            cur_embs: user sequence embeddings [N, S, D]
            target_pos: positive target items [N, S, D]
            target_neg: negative target items [N, num_neg, D]
            user_attention_mask: attention mask [N, S]

        Returns:
            logits, labels
        """
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))

        N, S, D = cur_embs.shape

        # Normalize embeddings
        cur_embs = cur_embs / cur_embs.norm(dim=-1, keepdim=True)
        target_pos = target_pos / target_pos.norm(dim=-1, keepdim=True)
        target_neg = target_neg / target_neg.norm(dim=-1, keepdim=True)

        # Compute logits
        logits_pos = (cur_embs * target_pos).sum(dim=-1, keepdim=True) * self.logit_scale.exp()
        logits_neg = torch.matmul(cur_embs, target_neg.transpose(1, 2)) * self.logit_scale.exp()

        # Filter out similar negatives
        mask = (logits_neg < self.nce_thres * logits_pos).float()
        logits_neg = logits_neg * mask + (1 - mask) * torch.finfo(logits_neg.dtype).min

        # Concatenate logits
        logits = torch.cat([logits_pos, logits_neg], dim=-1)  # [N, S, 1 + num_neg]

        # Apply attention mask
        logits = logits[user_attention_mask == 1]  # [N*S_valid, 1 + num_neg]

        # Labels (positive is always at index 0)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return logits, labels

    def forward(self, batch):
        """
        Forward pass

        Args:
            batch: dict containing input data

        Returns:
            dict with loss and metrics
        """
        user_attention_mask = batch['attention_mask']
        N, S = user_attention_mask.shape

        pos_input_ids = batch['pos_input_ids']
        pos_cu_input_lens = batch['pos_cu_input_lens']
        pos_position_ids = batch['pos_position_ids']

        neg_input_ids = batch['neg_input_ids']
        neg_cu_input_lens = batch['neg_cu_input_lens']
        neg_position_ids = batch['neg_position_ids']

        # Encode items
        pos_embedding = self.forward_item_emb(
            pos_input_ids, pos_position_ids, pos_cu_input_lens,
            self.item_emb_token_n, self.item_emb_tokens, self.item_llm
        )
        pos_embedding = pos_embedding.reshape(N, S + 1, -1)

        neg_embedding = self.forward_item_emb(
            neg_input_ids, neg_position_ids, neg_cu_input_lens,
            self.item_emb_token_n, self.item_emb_tokens, self.item_llm
        )
        neg_embedding = neg_embedding.reshape(N, -1, self.item_llm.config.hidden_size)

        # Target embeddings
        target_pos_embs = pos_embedding[:, 1:]  # Predict next items
        target_neg_embs = neg_embedding

        # Encode user sequence
        user_embedding = self.user_llm(
            inputs_embeds=pos_embedding[:, :-1],
            attention_mask=user_attention_mask
        ).hidden_states[-1]

        # Compute NCE loss
        logits, labels = self.nce_loss(user_embedding, target_pos_embs, target_neg_embs, user_attention_mask)
        loss = F.cross_entropy(logits, labels)

        # Metrics
        model_out = {
            'loss': loss,
            'nce_samples': (logits > torch.finfo(logits.dtype).min / 100).sum(dim=1).float().mean(),
        }

        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(1):
                break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = labels.view(-1, 1).eq(indices).any(dim=1).float().mean()

        return model_out
