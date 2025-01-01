from compress.lit import LitModel
from compress.data import PretrainDataset, load_processed_dataset
from loguru import logger
from lightning import Trainer as LTrainer
from omegaconf import OmegaConf
import argparse
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

logger.add("logs/train.log")

wandb_logger = WandbLogger(project="Condense")


def get_args():
    # Load default config
    config = OmegaConf.load("config/test.yaml")

    # Create parser with arguments matching config structure
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test.yaml")

    # Convert config to CLI arguments automatically
    cli_config = OmegaConf.from_cli()

    # Merge configs: CLI values override YAML config
    config = OmegaConf.merge(config, cli_config)

    return config


def main():
    config = get_args()
    logger.info(f"Config: {config}")
    logger.info("Initializing model...")
    lit_model = LitModel(config)

    logger.info("Loading dataset...")
    dataset = load_processed_dataset(lit_model.tokenizer)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    validation_dataset = split["test"]

    train_dataset = PretrainDataset(
        train_dataset,
        lit_model.tokenizer,
        config.trainer_config.data_config.max_length,
    )
    validation_dataset = PretrainDataset(
        validation_dataset,
        lit_model.tokenizer,
        config.trainer_config.data_config.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer_config.data_config.batch_size,
        num_workers=config.trainer_config.data_config.num_workers,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        num_workers=config.trainer_config.data_config.num_workers,
        shuffle=False,
    )

    logger.info("Initializing trainer...")
    trainer = LTrainer(
        **config.trainer_config.lightning_trainer_config,
        logger=wandb_logger,
    )

    logger.info("Training model...")
    trainer.fit(lit_model, train_loader, validation_loader)


if __name__ == "__main__":
    main()
