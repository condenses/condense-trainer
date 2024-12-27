from torch.utils.data import Dataset
from datasets import Dataset as HuggingFaceDataset
from transformers import LlamaTokenizer
import torch
import traceback
from datasets import load_dataset


class SubnetSyntheticDataset(Dataset):

    def __init__(
        self,
        dataset,
        tokenizer: LlamaTokenizer,
        separate_tokenizer: LlamaTokenizer,
        num_condense_tokens=512,
        max_characters=10000,
        max_length=2048,
        split="train"
    ):
        # Load full training dataset since only train split exists
        self.full_dataset = dataset
        # Split into train/test based on split parameter
        if split == "train":
            self.dataset = self.full_dataset.select(range(0, int(0.9 * len(self.full_dataset))))
        else:
            self.dataset = self.full_dataset.select(range(int(0.1 * len(self.full_dataset)), len(self.full_dataset)))
            
        self.tokenizer = tokenizer
        self.num_condense_tokens = num_condense_tokens
        self.max_characters = max_characters
        self.max_length = max_length
        self.separate_tokenizer = separate_tokenizer
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        context = item["text"]
        output = self.tokenizer(
            context,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        context_ids = output.input_ids
        context_mask = output.attention_mask
        # Remove bos token from labels

        return {
            "input_ids": context_ids.squeeze(0),
            "attention_mask": context_mask.squeeze(0),
            "str_context": context,
            "str_uncondensed": self.separate_tokenizer.decode(context_ids.squeeze(0)),
        }
