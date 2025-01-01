from datasets import load_dataset, concatenate_datasets


def format_conversations_to_messages(x):
    messages = []
    for conversation in x["conversations"]:
        role = "user" if conversation["from"] == "human" else "assistant"
        content = conversation["value"]
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def load_processed_dataset(tokenizer):
    full_dataset_1 = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")
    full_dataset_1 = full_dataset_1.shuffle(seed=42)
    full_dataset_1 = full_dataset_1.map(
        lambda x: {
            "text": tokenizer.apply_chat_template(x["messages"], tokenize=False)
        },
        num_proc=8,
    )
    full_dataset_1 = full_dataset_1.filter(lambda x: len(x["text"]) > 2000)

    full_dataset_2 = load_dataset(
        "manifoldlabs/Infinity-Instruct", "0625", split="train", num_proc=16
    )
    full_dataset_2 = full_dataset_2.map(
        format_conversations_to_messages,
        num_proc=8,
    )
    full_dataset_2 = full_dataset_2.map(
        lambda x: {
            "text": tokenizer.apply_chat_template(x["messages"], tokenize=False)
        },
        num_proc=16,
    )
    full_dataset_2 = full_dataset_2.filter(lambda x: len(x["text"]) > 2000, num_proc=16)
    full_dataset = concatenate_datasets([full_dataset_1, full_dataset_2])
    full_dataset = full_dataset.shuffle(seed=42)
    return full_dataset
