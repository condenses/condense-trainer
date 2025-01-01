from datasets import Dataset
import torch


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, tokenizer, max_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.dataset[index]
        text = item["text"]
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __len__(self):
        return len(self.dataset)
