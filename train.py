from condense_trainer_core import LitCondenseLLM, SubnetSyntheticDataset
from lightning import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import argparse
wandb_logger = WandbLogger(project="Condense")

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Use smaller test models")
parser.add_argument("--pretrained_id", type=str, default=None, help="HuggingFace repo ID of pretrained model")
parser.add_argument("--num_condense_tokens", type=int, default=512, help="Number of condense tokens")
parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens")
parser.add_argument("--max_characters", type=int, default=10000, help="Maximum number of characters")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
parser.add_argument("--dataset_id", type=str, default="Condense-AI/benchmark-condense-v0.1", help="Dataset to use")
parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model ID to use")
parser.add_argument("--target_model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Target model ID to use")
parser.add_argument("--devices", type=int, default=-1, help="Number of devices to use")
args = parser.parse_args()

num_condense_tokens = args.num_condense_tokens
max_tokens = args.max_tokens
max_characters = args.max_characters

dataset_id = args.dataset_id
if args.test:
    model_id = "HuggingFaceTB/SmolLM2-135M"
    target_model_id = "HuggingFaceTB/SmolLM2-135M"
else:
    model_id = args.model_id
    target_model_id = args.target_model_id

print(f"Model ID: {model_id}")
print(f"Target Model ID: {target_model_id}")
print(f"Pretrained ID: {args.pretrained_id}")

if args.pretrained_id is not None:
    lit_model = LitCondenseLLM.from_pretrained(model_id, target_model_id, args.pretrained_id)
else:
    lit_model = LitCondenseLLM(
        model_id=model_id,
        target_model_id=target_model_id,
        num_condense_tokens=num_condense_tokens,
        lora_r=128,
        lora_alpha=128,
        lora_dropout=0.1,
    )

tokenizer = lit_model.tokenizer
target_tokenizer = lit_model.target_tokenizer

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if target_tokenizer.pad_token is None:
    target_tokenizer.pad_token = target_tokenizer.eos_token

train_dataset = SubnetSyntheticDataset(
    dataset_id, tokenizer, target_tokenizer, num_condense_tokens, max_characters, max_length=max_tokens
)
validation_dataset = SubnetSyntheticDataset(
    dataset_id, tokenizer, target_tokenizer, num_condense_tokens, max_characters, max_length=max_tokens, split="test"
)

trainer = Trainer(
    max_epochs=10,
    precision="bf16",
    gradient_clip_val=1.0,
    log_every_n_steps=5,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
    val_check_interval=500,
    limit_val_batches=100,
    devices=args.devices,
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

trainer.fit(lit_model, train_loader, validation_loader)
