import os
import time
import torch
from huggingface_hub import HfApi
from lightning.pytorch.callbacks import Callback


class SaveModelHuggingface(Callback):
    """
    A Lightning callback to periodically save
    learned modules to Hugging Face Hub.
    """

    def __init__(
        self,
        output_dir: str = "checkpoints",
    ):
        super().__init__()
        self.output_dir = output_dir
        self.hf_api = HfApi()

    def setup(self, trainer, pl_module, stage: str) -> None:
        """
        Creates a Hugging Face repository to store
        model artifacts if it does not already exist.
        """
        try:
            repo_short_name = pl_module.model_id.split("/")[-1]
            time_str = time.strftime("%Y%m%d-%H%M%S")
            self.hf_save_repo = f"Condense-AI/Condenser-{repo_short_name}-{time_str}"

            lora_r = pl_module.lora_config.get("r", "unknown")
            lora_alpha = pl_module.lora_config.get("lora_alpha", "unknown")
            self.commit_description = (
                f"Condenser-{repo_short_name}, "
                f"{pl_module.target_model_id.split('/')[-1]}, "
                f"LoRA r={lora_r}, LoRA alpha={lora_alpha}"
            )

            self.hf_api.create_repo(
                repo_id=self.hf_save_repo,
                repo_type="model",
                exist_ok=True,
            )
        except Exception as e:
            print(f"Error creating HF repo: {e}")

    def on_validation_end(self, trainer, pl_module) -> None:
        """
        Saves learned tokens and embeddings to disk and
        then uploads them to the HF repository.
        """
        checkpoint = {
            "modules": {
                "condense_tokens": pl_module.compressor.condense_tokens.detach().cpu(),
                "ae_embedding": pl_module.compressor.ae_embedding.detach().cpu(),
                "lm_embedding": pl_module.compressor.lm_embedding.detach().cpu(),
            },
        }

        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.output_dir, "modules.pt")
        torch.save(checkpoint, checkpoint_path)

        self.hf_api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=checkpoint_path,
            repo_id=self.hf_save_repo,
        )

        pl_module.compressor.base_model.push_to_hub(self.hf_save_repo)
