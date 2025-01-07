import lightning as L
from .modeling import MultiSpanGistCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from loguru import logger
from cut_cross_entropy.transformers import cce_patch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


class LitModel(L.LightningModule):
    def __init__(
        self,
        config: OmegaConf,
    ):
        super().__init__()
        self.model_config = config.model_config
        self.optimizer_config = config.trainer_config.optimizer_config
        self.hf_api = HfApi()
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
        logger.info(
            "Loading target model...: {}".format(self.model_config.llm_model_id)
        )
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.llm_model_id
        )
        logger.info("Resizing token embeddings...")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.target_model.resize_token_embeddings(len(self.tokenizer))
        if self.model_config.pretrained_id:
            logger.info("Loading pretrained model...")
            self.model.load_pretrained(
                self.model_config.pretrained_id,
                self.hf_api,
            )
        logger.info("Freezing target model...")
        self.freeze_model(self.target_model)
        cce_patch(self.target_model)

    def configure_optimizers(self):
        """
        Returns AdamW optimizer for both LoRA parameters and the learnable tokens.
        """
        logger.info(f"Configuring optimizer...: {self.optimizer_config}")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # optimizer = torch.optim.AdamW(trainable_params, **self.optimizer_config)
        optimizer = DeepSpeedCPUAdam(
            trainable_params,
            **self.optimizer_config,
        )
        # optimizer = FusedAdam(
        #     trainable_params,
        #     **self.optimizer_config,
        # )
        return {"optimizer": optimizer}

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
            self.model.forward(
                input_ids, attention_mask, padding_id=self.tokenizer.pad_token_id
            )
        )
        multi_span_context_embeds = []
        for context_ids in multi_span_context_ids:
            context_embeds = self.target_model.get_input_embeddings()(context_ids)
            multi_span_context_embeds.append(context_embeds)

        losses = {}
        generate_inputs = []
        for task in self.model_config.objectives:
            task_losses = []
            (
                multi_inputs_embeds,
                multi_labels,
                multi_position_ids,
                multi_attention_masks,
                multi_inputs_embeds_for_generate,
                multi_gisted_context_ids,
                multi_gisted_position_ids,
            ) = self.model.forward_objective(
                multi_span_gist_features,
                multi_span_context_ids,
                multi_span_context_embeds,
                multi_span_attention_mask,
                task,
                device=self.device,
            )
            for inputs_embeds, labels, position_ids, attention_mask in zip(
                multi_inputs_embeds,
                multi_labels,
                multi_position_ids,
                multi_attention_masks,
            ):
                loss = self.get_loss(
                    inputs_embeds, labels, position_ids, attention_mask
                )
                task_losses.append(loss)
            loss = sum(task_losses) / len(task_losses)
            losses[task] = loss
            if multi_inputs_embeds_for_generate is not None:
                generate_inputs.append(
                    {
                        "list_inputs_embeds": multi_inputs_embeds_for_generate,
                        "list_gisted_context_ids": multi_gisted_context_ids,
                        "list_position_ids": multi_gisted_position_ids,
                        "list_labels": multi_labels,
                    }
                )
        return losses, generate_inputs

    def get_loss(self, inputs_embeds, labels, position_ids, attention_mask):
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.target_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        losses, _ = self.forward(batch)
        for task, loss in losses.items():
            self.log(f"train/{task}_loss", loss, on_step=True, on_epoch=True)
        loss = sum(losses.values())
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        losses, generate_inputs = self.forward(batch)
        for task, loss in losses.items():
            self.log(f"val/{task}_loss", loss, on_step=True, on_epoch=True)
        loss = sum(losses.values())
        self.log("val/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx <= 5:
            generated_samples = self.generate(generate_inputs)
            for gisted_text, target_text, generated_text in generated_samples:
                print("*" * 100)
                print(gisted_text)
                print("-" * 100)
                print(target_text)
                print("-" * 100)
                print(generated_text)
                print("*" * 100)
                self.text_samples.append((gisted_text, generated_text, target_text))
        return loss

    def on_validation_start(self):
        self.text_samples = []

    def on_validation_end(self):
        self.logger.log_table(
            "generated_samples",
            columns=[
                "gisted_text",
                "generated_text",
                "target_text",
            ],
            data=self.text_samples,
        )

    def generate(self, generate_inputs, max_length=64):
        """
        Generates text continuation from condensed embeddings.

        Args:
            generate_inputs (list): List of dicts containing inputs for generation
            max_length (int): Maximum number of tokens to generate

        Returns:
            list: List of tuples containing (gisted_text, target_text, generated_text)
        """
        generated_samples = []

        for inputs in generate_inputs:
            list_inputs_embeds = inputs["list_inputs_embeds"]
            list_gisted_context_ids = inputs["list_gisted_context_ids"]
            list_position_ids = inputs["list_position_ids"]
            list_labels = inputs["list_labels"]

            for inputs_embeds, position_ids, labels, gisted_context_ids in zip(
                list_inputs_embeds,
                list_position_ids,
                list_labels,
                list_gisted_context_ids,
            ):
                # Initial forward pass with condensed embeddings
                if inputs_embeds is None:
                    continue
                print(position_ids.shape)
                print(inputs_embeds.shape)
                out = self.target_model(
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )

                past_key_values = out.past_key_values
                logits = out.logits
                next_token_id = torch.argmax(logits[:, -1], dim=-1)

                generated_ids = [next_token_id.item()]
                next_position_ids = position_ids[:, -1:] + 1

                # Get embeddings for the predicted token
                next_inputs_embeds = (
                    self.target_model.get_input_embeddings()(next_token_id)
                    .unsqueeze(1)
                    .to(inputs_embeds.device)
                )

                # Auto-regressive generation
                for _ in range(max_length):
                    out = self.target_model(
                        position_ids=next_position_ids,
                        inputs_embeds=next_inputs_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    logits = out.logits[:, -1]
                    past_key_values = out.past_key_values
                    next_token_id = torch.argmax(logits, dim=-1)
                    generated_ids.append(next_token_id.item())

                    # Stop if we hit the end token
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break

                    # Update for next iteration
                    next_inputs_embeds = (
                        self.target_model.get_input_embeddings()(next_token_id)
                        .unsqueeze(1)
                        .to(inputs_embeds.device)
                    )
                    next_position_ids = next_position_ids + 1

                # remove -100 from labels
                labels = labels[labels != -100]
                target_text = self.tokenizer.decode(labels, skip_special_tokens=False)

                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=False
                )
                gisted_text = self.tokenizer.decode(
                    gisted_context_ids[0], skip_special_tokens=False
                )

                generated_samples.append((gisted_text, target_text, generated_text))

        return generated_samples
