import lightning as L
from .modeling import MultiSpanGistCausalLM, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from loguru import logger


class LitModel(L.LightningModule):
    def __init__(
        self,
        config: OmegaConf,
    ):
        super().__init__()
        self.model_config = config.model_config
        self.optimizer_config = config.trainer_config.optimizer_config
        self.configure_tokenizers()

    def configure_tokenizers(self):
        # Automatically called when starting training
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.llm_model_id)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def configure_model(self):
        logger.info("Loading model...")
        self.model = MultiSpanGistCausalLM(
            **self.model_config,
        )
        logger.info("Loading target model...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.llm_model_id
        )
        logger.info("Resizing token embeddings...")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.target_model.resize_token_embeddings(len(self.tokenizer))
        logger.info("Freezing target model...")
        self.freeze_model(self.target_model)

    def freeze_model(self, model: torch.nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def ce_loss(self, logits, labels):
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def forward(self, batch: dict):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        multi_span_gist_features, multi_span_context_ids, multi_span_attention_mask = (
            self.model.forward(input_ids, attention_mask)
        )
        multi_span_context_embeds = []
        for context_ids in multi_span_context_ids:
            context_embeds = self.target_model.get_input_embeddings()(context_ids)
            multi_span_context_embeds.append(context_embeds)

        losses = {}
        for task in self.model_config.objectives:
            task_losses = []
            multi_inputs_embeds, multi_labels, multi_position_ids = (
                self.model.forward_objective(
                    multi_span_gist_features,
                    multi_span_context_ids,
                    multi_span_context_embeds,
                    task,
                    device=self.device,
                )
            )
            for inputs_embeds, labels, position_ids in zip(
                multi_inputs_embeds, multi_labels, multi_position_ids
            ):
                loss = self.get_loss(inputs_embeds, labels, position_ids)
                task_losses.append(loss)
            loss = sum(task_losses) / len(task_losses)
            losses[task] = loss
        return losses

    def get_loss(self, inputs_embeds, labels, position_ids):
        labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = torch.ones_like(labels).to(self.device)
        outputs = self.target_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        losses = self.forward(batch)
        for task, loss in losses.items():
            self.log(f"train/{task}_loss", loss, on_step=True, on_epoch=True)
        loss = sum(losses.values())
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.forward(batch)
        for task, loss in losses.items():
            self.log(f"val/{task}_loss", loss, on_step=True, on_epoch=True)
        loss = sum(losses.values())
        self.log("val/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Returns AdamW optimizer for both LoRA parameters and the learnable tokens.
        """
        logger.info(f"Configuring optimizer...: {self.optimizer_config}")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, **self.optimizer_config)
        return {"optimizer": optimizer}
