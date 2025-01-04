from lightning.pytorch.callbacks import Callback
from huggingface_hub import HfApi
import os


class SaveModelHuggingface(Callback):
    def __init__(
        self,
        output_dir: str = "checkpoints",
    ):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.hf_api = HfApi()

    def setup(self, trainer, pl_module, stage: str) -> None:
        """Initialize HF repo on setup."""
        try:
            model_config = pl_module.model_config
            self.hf_save_repo = f"Condense-AI/Condenser_{model_config.llm_model_id.split('/')[-1]}_{model_config.num_gist_tokens}_{model_config.max_length}_{model_config.num_auto_encoding_flag}_{model_config.num_complete_flag}"
            # Create the repo
            self.hf_api.create_repo(
                repo_id=self.hf_save_repo,
                repo_type="model",
                exist_ok=True,
            )
        except Exception as e:
            print(f"Error creating HF repo: {e}")

    def on_validation_end(self, trainer, pl_module) -> None:
        pl_module.model.push_to_hub(self.hf_save_repo, self.output_dir, self.hf_api)
