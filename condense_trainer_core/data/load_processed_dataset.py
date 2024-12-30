from datasets import load_dataset
from transformers import AutoTokenizer


def load_redpajama_dataset(
    num_proc: int = 8,
    seed: int = 42,
    min_characters: int = 2000,
):
    """
    Loads a RedPajama dataset from Hugging Face.
    """
    full_dataset = load_dataset(
        "DKYoon/SlimPajama-6B", split="train", num_proc=num_proc
    ).shuffle(seed=seed)
    # Some optional filtering
    full_dataset = full_dataset.filter(
        lambda x: len(x["text"]) > min_characters, num_proc=num_proc
    )
    return full_dataset


def load_instruction_dataset(
    tokenizer: AutoTokenizer,
    num_proc: int = 8,
    seed: int = 42,
    min_characters: int = 2000,
):
    """
    Loads an instruction dataset from Hugging Face.
    """
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", num_proc=num_proc
    )
    ds = ds.map(
        lambda x: {
            "text": tokenizer.apply_chat_template(
                x["messages"], tokenize=False, add_special_tokens=False
            )
        },
        num_proc=num_proc,
    )
    ds = ds.filter(lambda x: len(x["text"]) > min_characters, num_proc=num_proc)
    return ds
