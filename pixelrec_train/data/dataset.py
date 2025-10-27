"""
PixelRec Dataset
Extracted and refactored from original HLLM TextSEQTrainDataset
"""

import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from logging import getLogger

logger = getLogger(__name__)


class PixelRecDataset(Dataset):
    """
    Dataset for PixelRec training

    Processes interaction sequences and text features for hierarchical LLM recommendation
    """

    def __init__(
        self,
        interaction_file,
        text_file,
        tokenizer_path,
        max_seq_length=11,
        max_text_length=256,
        text_keys=['title', 'description'],
        item_emb_token_n=1,
        num_negatives=None,
        item_prompt="",
        use_nce=True,
    ):
        """
        Args:
            interaction_file: path to interaction CSV (user_id, item_id, timestamp)
            text_file: path to text features CSV (item_id, title, description, etc.)
            tokenizer_path: path to pretrained tokenizer
            max_seq_length: maximum sequence length (MAX_ITEM_LIST_LENGTH + 1)
            max_text_length: maximum text length for item description
            text_keys: list of text feature keys to use
            item_emb_token_n: number of embedding tokens per item
            num_negatives: number of negative samples (None for sequence-level)
            item_prompt: prompt prefix for item text
            use_nce: whether to use NCE loss (affects negative sampling)
        """
        self.max_seq_length = max_seq_length
        self.max_text_length = max_text_length
        self.text_keys = text_keys
        self.item_emb_token_n = item_emb_token_n
        self.num_negatives = num_negatives
        self.item_prompt = item_prompt
        self.use_nce = use_nce

        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        logger.info(f"Loading interaction data from {interaction_file}")
        self._load_interaction_data(interaction_file)

        logger.info(f"Loading text data from {text_file}")
        self._load_text_data(text_file)

        logger.info(f"Building training sequences")
        self._build_sequences()

        logger.info(f"Dataset initialized: {len(self)} samples")
        logger.info(f"  - Item num: {self.item_num}")
        logger.info(f"  - Max seq length: {self.max_seq_length}")
        logger.info(f"  - Max text length: {self.max_text_length}")
        logger.info(f"  - Text keys: {self.text_keys}")
        logger.info(f"  - Num negatives: {self.num_negatives}")
        logger.info(f"  - Use NCE: {self.use_nce}")

    def _load_interaction_data(self, interaction_file):
        """Load interaction data and build mappings"""
        df = pd.read_csv(interaction_file)

        # Ensure required columns exist
        assert 'user_id' in df.columns, "Missing user_id column"
        assert 'item_id' in df.columns, "Missing item_id column"

        # Convert to string for consistency
        df['user_id'] = df['user_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)

        # Build item mapping (reserve 0 for padding)
        unique_items = sorted(df['item_id'].unique())
        self.item2id = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self.item2id['<PAD>'] = 0
        self.id2item = {idx: item for item, idx in self.item2id.items()}
        self.item_num = len(self.item2id)

        # Remap item IDs
        df['item_id_internal'] = df['item_id'].map(self.item2id)

        # Group by user
        self.user_sequences = df.groupby('user_id')['item_id_internal'].apply(list).to_dict()
        self.user_items_map = df.groupby('user_id')['item_id'].apply(list).to_dict()

        logger.info(f"Loaded {len(df)} interactions from {len(self.user_sequences)} users")

    def _load_text_data(self, text_file):
        """Load item text features"""
        df = pd.read_csv(text_file, dtype={'item_id': str})

        # Keep only required columns
        cols = ['item_id'] + [k for k in self.text_keys if k in df.columns]
        df = df[cols]

        # Convert to dict for fast lookup
        df = df.set_index('item_id')
        self.item_text = df.T.to_dict()

        logger.info(f"Loaded text features for {len(self.item_text)} items")

    def _build_sequences(self):
        """Build training sequences from user interactions"""
        self.sequences = []

        for user_id, item_seq in self.user_sequences.items():
            # Need at least 3 items (2 for history, 1 for target)
            if len(item_seq) >= 3:
                # Exclude last 2 items (for evaluation)
                train_seq = item_seq[:-2]
                if len(train_seq) >= 2:
                    self.sequences.append(train_seq)

    def __len__(self):
        return len(self.sequences)

    def _neg_sample(self, item_set):
        """Sample a negative item not in item_set"""
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        """Pad sequence to max_length"""
        pad_len = max_length - len(sequence)
        if random_sample and self.use_nce:
            # Use negative samples as padding for NCE loss
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return torch.tensor(sequence, dtype=torch.long)

    def _process_item_text(self, item_id):
        """
        Process item text into token IDs

        Args:
            item_id: internal item ID

        Returns:
            token IDs
        """
        # Get original item token
        item_token = self.id2item[item_id]

        # Skip padding token
        if item_token == '<PAD>':
            text_str = ""
        elif item_token not in self.item_text:
            logger.warning(f"Item {item_token} not found in text data")
            text_str = ""
        else:
            # Build text string from features
            item_data = self.item_text[item_token]
            text_str = self.item_prompt
            for key in self.text_keys:
                value = item_data.get(key, '')
                if value and str(value) != 'nan':
                    text_str += f"{key}: {value} "

        # Tokenize
        ids = self.tokenizer.encode(text_str)
        ids = ids[:self.max_text_length]

        return ids

    def __getitem__(self, index):
        """
        Get training sample

        Returns:
            dict with:
                - pos_input_ids: concatenated token IDs for positive items
                - pos_cu_input_lens: cumulative lengths of each item
                - pos_position_ids: position IDs
                - neg_input_ids: token IDs for negative items
                - neg_cu_input_lens: cumulative lengths
                - neg_position_ids: position IDs
                - attention_mask: mask for valid positions
        """
        item_seq = self.sequences[index]

        # Build training data
        # For each item in sequence, predict the next item
        item_seq_len = len(item_seq)

        # Sample negative items
        neg_items = []
        masked_index = []
        for i in range(item_seq_len - 1):
            neg_items.append(self._neg_sample(item_seq))
            masked_index.append(1)

        # Pad sequences
        item_seq_padded = self._padding_sequence(
            list(item_seq),
            self.max_seq_length,
            random_sample=self.use_nce
        )
        masked_index = self._padding_sequence(masked_index, self.max_seq_length - 1)

        # Handle negative sampling strategy
        if self.num_negatives:
            # Item-level negatives (fixed number per batch)
            neg_items = []
            for _ in range(self.num_negatives):
                neg_items.append(self._neg_sample([]))
        else:
            # Sequence-level negatives (one per position)
            neg_items = self._padding_sequence(
                neg_items,
                self.max_seq_length,
                random_sample=self.use_nce
            )

        # Process item text into token IDs
        pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []

        for item_id in item_seq_padded:
            ids = self._process_item_text(item_id.item())

            # Append embedding token placeholders
            pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
            pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)

            # Position IDs (right-aligned to max_text_length)
            pos_ids = (torch.arange(len(ids) + self.item_emb_token_n) +
                      (self.max_text_length - len(ids))).tolist()
            pos_position_ids.extend(pos_ids)

        # Process negative items
        neg_input_ids, neg_cu_input_lens, neg_position_ids = [], [], []

        for item_id in neg_items:
            if isinstance(item_id, torch.Tensor):
                item_id = item_id.item()

            ids = self._process_item_text(item_id)

            neg_input_ids.extend(ids + [0] * self.item_emb_token_n)
            neg_cu_input_lens.append(len(ids) + self.item_emb_token_n)

            neg_pos_ids = (torch.arange(len(ids) + self.item_emb_token_n) +
                          (self.max_text_length - len(ids))).tolist()
            neg_position_ids.extend(neg_pos_ids)

        return {
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
            "neg_input_ids": torch.as_tensor(neg_input_ids, dtype=torch.int64),
            "neg_cu_input_lens": torch.as_tensor(neg_cu_input_lens, dtype=torch.int64),
            "neg_position_ids": torch.as_tensor(neg_position_ids, dtype=torch.int64),
            "attention_mask": torch.as_tensor(masked_index, dtype=torch.int64),
        }
