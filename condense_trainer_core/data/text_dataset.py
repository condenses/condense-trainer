from torch.utils.data import Dataset
from transformers import LlamaTokenizer


class TextDataset(Dataset):
    """
    A dataset that wraps text samples, tokenizing them for
    usage by a condense mechanism and a separate target model.
    """

    def __init__(
        self,
        dataset,
        tokenizer: LlamaTokenizer,
        max_length=2048,
    ):
        """
        Args:
            dataset: An iterable or HuggingFace dataset of text samples.
            tokenizer: Tokenizer for the base model.
            max_length: Max token length after tokenization.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        return {
            "input_ids": context_ids.squeeze(0),
            "attention_mask": context_mask.squeeze(0),
        }
