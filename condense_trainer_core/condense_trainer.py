import torch
import lightning as L
from transformers import AutoModelForCausalLM, AutoTokenizer
from .objectives import AutoencodeObjective, ContinuationObjective
from .compressor import Compressor


class CondenseTrainer(L.LightningModule):
    """
    A LightningModule that orchestrates training
    by applying one or more objectives to the data.
    """

    def __init__(
        self,
        model_id: str,
        target_model_id: str,
        tokenizer_name: str = None,
        pretrained_id: str = None,
        num_condense_tokens: int = 512,
        max_seq_length: int = 4096,
        lora_r: int = 128,
        lora_alpha: int = 128,
        lora_dropout: float = 0.0,
        objectives=None,
        compress_rate: int = 4,
    ):
        """
        Args:
            model_id: Base model name or path for the compressor.
            target_model_id: A second model used as the "target"
                             for reconstruction or continuation.
            tokenizer_name: Name or path to the tokenizer
                            (defaults to model_id).
            pretrained_id: Hugging Face repo ID for loading
                           pretrained compressor weights.
            num_condense_tokens: Number of learned condense tokens.
            max_seq_length: Maximum sequence length for tokenization.
            lora_r, lora_alpha, lora_dropout: LoRA hyperparameters.
            objectives: A list of objective instances
                        (e.g., [AutoencodeObjective, ContinuationObjective]).
        """
        super().__init__()
        self.model_id = model_id
        self.target_model_id = target_model_id
        if tokenizer_name is None:
            tokenizer_name = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        self.target_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.model_id = model_id
        self.pretrained_id = pretrained_id
        self.compress_rate = compress_rate
        self.num_condense_tokens = num_condense_tokens
        self.max_seq_length = max_seq_length
        self.lora_config = {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }

    def configure_model(self):
        self.compressor = Compressor(
            model_id=self.model_id,
            tokenizer=self.tokenizer,
            lora_config=self.lora_config,
            num_condense_tokens=self.num_condense_tokens,
            pretrained_id=self.pretrained_id,
            compress_rate=self.compress_rate,
        )
        self.compressor.base_model.resize_token_embeddings(len(self.tokenizer))

        self.target_model = AutoModelForCausalLM.from_pretrained(self.target_model_id)
        self.target_model.resize_token_embeddings(len(self.target_tokenizer))
        for _, param in self.target_model.named_parameters():
            param.requires_grad = False

        self.max_seq_length = self.max_seq_length
        self.objectives = {
            "autoencoding": AutoencodeObjective(
                self.target_model, self.target_tokenizer
            ),
            "continuation": ContinuationObjective(
                self.target_model, self.target_tokenizer
            ),
        }

    def forward(self, batch):
        """
        Not used for direct forward.
        Instead objectives call self.compressor internally.
        """
        return {}

    def training_step(self, batch, batch_idx):
        """
        Computes the sum of all enabled objectives on this batch.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        for obj in self.objectives:
            loss_val = self.objectives[obj](self.compressor, batch)
            self.log(
                f"{obj}_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True
            )
            total_loss = total_loss + loss_val

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Same as training_step but logs separate metrics for validation.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        for obj in self.objectives:
            loss_val = self.objectives[obj](self.compressor, batch)
            self.log(
                f"{obj}_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True
            )
            total_loss = total_loss + loss_val

        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        """
        AdamW for both the compressor's base model and
        learnable tokens.
        """
        param_groups = [
            {
                "params": [
                    self.compressor.condense_tokens,
                    self.compressor.ae_embedding,
                    self.compressor.lm_embedding,
                ],
                "lr": 1e-4,
            },
            {"params": self.compressor.base_model.parameters(), "lr": 1e-4},
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)
        return {"optimizer": optimizer}
