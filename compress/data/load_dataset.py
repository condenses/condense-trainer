from datasets import load_dataset, concatenate_datasets
import os
import random

MAX_PROC = os.cpu_count() * 2


def _format_conversations_to_messages(x):
    """Convert conversation format to messages format."""
    messages = []
    for conversation in x["conversations"]:
        role = "user" if conversation["from"] == "human" else "assistant"
        messages.append({"role": role, "content": conversation["value"]})
    return {"messages": messages}


def _augment_with_math(messages1, math_dataset):
    """Randomly combine messages from two datasets."""
    if random.random() < 0.5:
        # Get a random index within the math dataset size
        random_idx = random.randint(0, len(math_dataset) - 1)
        # Get single item from math dataset
        math_message = math_dataset[random_idx]["messages"]
        return messages1 + math_message
    return messages1


def _process_dataset(dataset, tokenizer, augment_with=None):
    """Process a dataset with optional augmentation."""
    if augment_with is not None:
        dataset = dataset.map(
            lambda x: {"messages": _augment_with_math(x["messages"], augment_with)},
            num_proc=MAX_PROC,
        )

    dataset = dataset.map(
        lambda x: {
            "text": tokenizer.apply_chat_template(x["messages"], tokenize=False)
        },
        num_proc=MAX_PROC,
    )
    return dataset.filter(lambda x: len(x["text"]) > 2000, num_proc=MAX_PROC)


def load_processed_dataset(tokenizer):
    """Load and process multiple datasets."""
    # Load UltraChat dataset
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")
    ultrachat = ultrachat.shuffle(seed=42)
    ultrachat = _process_dataset(ultrachat, tokenizer)

    # Load Infinity Instruct dataset
    infinity = load_dataset(
        "manifoldlabs/Infinity-Instruct", "0625", split="train", num_proc=MAX_PROC
    )
    infinity = infinity.map(_format_conversations_to_messages, num_proc=MAX_PROC)

    # Prepare math dataset without loading all messages at once
    math_dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M")
    math_dataset = math_dataset.map(
        lambda x: {
            "messages": [
                {"role": "user", "content": x["problem"]},
                {"role": "assistant", "content": x["generated_solution"]},
            ]
        },
        num_proc=MAX_PROC,
    )
    math_dataset = math_dataset.shuffle(seed=42)

    print("Mixing math dataset with infinity dataset")
    # Pass the entire dataset instead of just messages
    infinity = _process_dataset(infinity, tokenizer, augment_with=math_dataset)

    # Combine and shuffle datasets
    combined_dataset = concatenate_datasets([ultrachat, infinity])
    return combined_dataset.shuffle(seed=42)
