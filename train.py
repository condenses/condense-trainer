import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse
import condense_trainer_core as ct


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a CondenseTrainer model")

    # Model arguments
    parser.add_argument(
        "--model-id", default="meta-llama/Llama-3.2-3B-Instruct", help="Base model ID"
    )
    parser.add_argument(
        "--target-model-id",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Target model ID",
    )
    parser.add_argument(
        "--pretrained-id",
        default=None,
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--num-condense-tokens", type=int, default=128, help="Number of condense tokens"
    )
    parser.add_argument("--compress-rate", type=int, default=4, help="Compression rate")
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Maximum sequence length"
    )
    parser.add_argument("--lora-r", type=int, default=512, help="LoRA r parameter")
    parser.add_argument(
        "--lora-alpha", type=int, default=512, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.0, help="LoRA dropout rate"
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of data loading workers"
    )
    parser.add_argument(
        "--devices", type=int, default=-1, help="Number of devices to use"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument("--precision", default="bf16-true", help="Training precision")
    parser.add_argument(
        "--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--log-every-n-steps", type=int, default=5, help="Log frequency"
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=1,
        help="Validation check frequency",
    )
    parser.add_argument(
        "--val-check-interval", type=int, default=500, help="Validation check interval"
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=100,
        help="Number of validation batches",
    )

    # Dataset arguments
    parser.add_argument(
        "--min-characters", type=int, default=2000, help="Minimum text length"
    )

    # Logging arguments
    parser.add_argument(
        "--wandb-project", default="Condense", help="Weights & Biases project name"
    )

    parser.add_argument("--test", action="store_true", help="Run a test run")

    return parser.parse_args()


def main():
    """
    Trains a CondenseTrainer module on a chosen dataset
    with autoencoding and continuation objectives.
    """
    # Parse arguments
    args = parse_args()
    if args.test:
        args.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        args.target_model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    print(args)

    wandb_logger = WandbLogger(project=args.wandb_project)

    # Initialize final trainer with objectives
    model = ct.condense_trainer.CondenseTrainer(
        model_id=args.model_id,
        target_model_id=args.target_model_id,
        pretrained_id=args.pretrained_id,
        num_condense_tokens=args.num_condense_tokens,
        max_seq_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        compress_rate=args.compress_rate,
    )

    full_dataset = ct.data.processed_dataset_loader.load_instruction_dataset(
        tokenizer=model.target_tokenizer,
        num_proc=16,
        seed=42,
        min_characters=args.min_characters,
    )
    split = full_dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    valid_dataset = split["test"]
    train_dataset = ct.data.TextDataset(
        train_dataset,
        tokenizer=model.target_tokenizer,
        max_length=args.max_length,
    )
    valid_dataset = ct.data.TextDataset(
        valid_dataset,
        tokenizer=model.target_tokenizer,
        max_length=args.max_length,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        logger=wandb_logger,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[ct.callbacks.SaveModelHuggingface()],
    )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
