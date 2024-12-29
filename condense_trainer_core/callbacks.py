from lightning.pytorch.callbacks import Callback
from huggingface_hub import HfApi
import os
import torch
import time


class SaveModelHuggingface(Callback):
    def __init__(
        self,
        output_dir: str = "checkpoints",
    ):
        super().__init__()
        self.output_dir = output_dir
        self.hf_api = HfApi()
        
    def setup(self, trainer, pl_module, stage: str) -> None:
        """Initialize HF repo on setup."""
        try:
            self.hf_save_repo = f"Condense-AI/Condenser-{pl_module.model_id.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}"
            self.commit_description = (f"Condenser-{pl_module.model_id.split('/')[-1]}, {pl_module.target_model_id.split('/')[-1]}, "
                                    f"LoRA r={pl_module.lora_config['r']}, LoRA alpha={pl_module.lora_config['lora_alpha']}")
            
            # Create the repo
            self.hf_api.create_repo(
                repo_id=self.hf_save_repo,
                repo_type="model",
                exist_ok=True,
            )
        except Exception as e:
            print(f"Error creating HF repo: {e}")

    def on_validation_end(self, trainer, pl_module) -> None:
        checkpoint = {
            "modules": {
                "condense_tokens": pl_module.condense_tokens.detach().cpu(),
                "ae_embedding": pl_module.ae_embedding.detach().cpu(),
                "bos_embedding": pl_module.bos_embedding.detach().cpu(),
            },
        }

        # Save locally
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.output_dir, "modules.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Upload to HF
        self.hf_api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=checkpoint_path,
            repo_id=self.hf_save_repo,
            commit_message=f"Epoch {trainer.current_epoch}, Loss: {trainer.callback_metrics.get('val_loss'):.4f}",
        )
        
        # Save the LoRA model
        pl_module.base_model.push_to_hub(
            self.hf_save_repo,
            commit_message=f"Epoch {trainer.current_epoch}, Loss: {trainer.callback_metrics.get('val_loss'):.4f}\n{self.commit_description}"
        ) 