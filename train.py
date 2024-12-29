from condense_trainer_core import LitCondenseLLM, SubnetSyntheticDataset
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import argparse
from condense_trainer_core import SaveModelHuggingface
from datasets import load_dataset

wandb_logger = WandbLogger(project="Condense")

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Use smaller test models")
parser.add_argument(
    "--pretrained_id",
    type=str,
    default=None,
    help="HuggingFace repo ID of pretrained model",
)
parser.add_argument(
    "--num_condense_tokens", type=int, default=512, help="Number of condense tokens"
)
parser.add_argument(
    "--max_tokens", type=int, default=4096, help="Maximum number of tokens"
)
parser.add_argument(
    "--max_characters", type=int, default=10000, help="Maximum number of characters"
)
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument(
    "--num_workers", type=int, default=8, help="Number of dataloader workers"
)
parser.add_argument(
    "--dataset_id",
    type=str,
    default="Condense-AI/benchmark-condense-v0.1",
    help="Dataset to use",
)
parser.add_argument(
    "--model_id",
    type=str,
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="Model ID to use",
)
parser.add_argument(
    "--target_model_id",
    type=str,
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="Target model ID to use",
)
parser.add_argument("--devices", type=int, default=-1, help="Number of devices to use")
args = parser.parse_args()

num_condense_tokens = args.num_condense_tokens
max_tokens = args.max_tokens
max_characters = args.max_characters

dataset_id = args.dataset_id
if args.test:
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    target_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
else:
    model_id = args.model_id
    target_model_id = args.target_model_id

print(f"Model ID: {model_id}")
print(f"Target Model ID: {target_model_id}")
print(f"Pretrained ID: {args.pretrained_id}")
lit_model = LitCondenseLLM(
    model_id=model_id,
    pretrained_id=args.pretrained_id,
    target_model_id=target_model_id,
    num_condense_tokens=num_condense_tokens,
    lora_r=512,
    lora_alpha=512,
    lora_dropout=0.0,
    mean_compression_ratio=4,
)

tokenizer = lit_model.tokenizer
target_tokenizer = lit_model.target_tokenizer

# full_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", num_proc=8)
full_dataset = load_dataset("DKYoon/SlimPajama-6B", split="train", num_proc=16)
full_dataset = full_dataset.shuffle(seed=100)

# full_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")
# full_dataset = full_dataset.shuffle(seed=42)
# full_dataset = full_dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)}, num_proc=8)
# full_dataset = full_dataset.filter(lambda x: len(x["text"]) > 2000)
# print(len(full_dataset))
# print(full_dataset[0]["text"])
# Split into train/test based on split parameter

train_dataset = full_dataset.select(range(0, int(0.9 * len(full_dataset))))
validation_dataset = full_dataset.select(
    range(int(0.9 * len(full_dataset)), len(full_dataset))
)

train_dataset = SubnetSyntheticDataset(
    train_dataset,
    tokenizer,
    target_tokenizer,
    num_condense_tokens,
    max_characters,
    max_length=max_tokens,
)
validation_dataset = SubnetSyntheticDataset(
    validation_dataset,
    tokenizer,
    target_tokenizer,
    num_condense_tokens,
    max_characters,
    max_length=max_tokens,
)

trainer = Trainer(
    max_epochs=10,
    precision="bf16-true",
    gradient_clip_val=1.0,
    log_every_n_steps=5,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
    val_check_interval=500,
    limit_val_batches=100,
    devices=args.devices,
    strategy=DDPStrategy(find_unused_parameters=False),
    callbacks=[SaveModelHuggingface()],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)
validation_loader = DataLoader(
    validation_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
)

trainer.fit(lit_model, train_loader, validation_loader)
