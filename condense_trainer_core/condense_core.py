import torch
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from huggingface_hub import HfApi
from peft import get_peft_model, LoraConfig
import os
import traceback
import time
import huggingface_hub
import math
import schedulefree
from peft import PeftModel


class LitCondenseLLM(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        target_model_id: str,
        pretrained_id: str = None,
        num_condense_tokens: int = 386,
        max_seq_length: int = 4096,
        output_dir: str = "checkpoints",
        lora_r: int = 128,
        lora_alpha: int = 128,
        lora_dropout: float = 0,
        mean_compression_ratio: float = 2,
    ):
        super().__init__()
        self.lora_config = {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        self.pretrained_id = pretrained_id
        self.is_pretrained = pretrained_id is not None
        self.model_id = model_id
        self.target_model_id = target_model_id
        self.max_seq_length = max_seq_length
        self.num_condense_tokens = num_condense_tokens
        self.mean_compression_ratio = mean_compression_ratio
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.best_val_loss = float("inf")
        self.hf_api = HfApi()
        self.hf_save_repo = f"Condense-AI/Condenser-{model_id.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.commit_description = (f"Condenser-{model_id.split('/')[-1]}, {target_model_id.split('/')[-1]}, "
                                   f"LoRA r={lora_r}, LoRA alpha={lora_alpha}, LoRA dropout={lora_dropout}")
        self.output_dir = output_dir

    def configure_model(self):
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="flash_attention_2")
        for _, param in self.base_model.named_parameters():
            param.requires_grad = True
        if not self.is_pretrained:    
            self.base_model = get_peft_model(self.base_model, peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                **self.lora_config
            ))
            self.base_model.print_trainable_parameters()
        else:
            lora_model = PeftModel.from_pretrained(self.base_model, self.pretrained_id, is_trainable=True)
            self.base_model = lora_model
            self.base_model.print_trainable_parameters()
        self.base_model.gradient_checkpointing_enable()
        self.target_model = self._initialize_target_model(self.target_model_id)
        self.hidden_size = self.base_model.config.hidden_size
        self.target_hidden_size = self.target_model.config.hidden_size
        # Initialize learnable parameters
        self.condense_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size)
        )
        self.bos_embedding = self.target_model.get_input_embeddings()(torch.tensor(self.target_tokenizer.bos_token_id).unsqueeze(0).unsqueeze(0))
        self.bos_embedding.requires_grad = False
        self.ae_embedding = nn.Parameter(
            torch.randn(1, 1, self.hidden_size)
        )
        self.lm_embedding = nn.Parameter(
            torch.randn(1, 1, self.hidden_size)
        )
        self.span_concat_embedding = nn.Parameter(
            torch.randn(1, 1, self.hidden_size)
        )
        self._initialize_embeddings()
        if self.is_pretrained:
            self.load_pretrained(self.pretrained_id)

    def _initialize_embeddings(self):
        for param in [self.condense_tokens, self.ae_embedding]:
            if isinstance(param, nn.Parameter):
                torch.nn.init.xavier_uniform_(param)

    def forward(self, input_ids, attention_mask) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        prompt_embeds = self.base_model.get_input_embeddings()(input_ids)
        total_length = prompt_embeds.size(1)
        num_segments = math.ceil(total_length / (self.num_condense_tokens * self.mean_compression_ratio))
        segment_length = math.ceil(total_length / num_segments)
        
        all_condensed_tokens = []
        level_condensed_tokens = []
        segment_input_ids = []
        level_segment_labels = []
        pre_condense_tokens = []

        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            
            segment_embeds = prompt_embeds[:, start_idx:end_idx, :]
            input_validation_ids = input_ids[:, :end_idx]
            segment_labels = input_ids[:, end_idx:]
            if torch.all(segment_labels == self.tokenizer.pad_token_id) or segment_labels.size(1) == 0:
                continue
            # Skip if segment_labels is full of padding tokens
            segment_input_ids.append(input_ids[:, start_idx:end_idx])
            
            segment_mask = attention_mask[:, start_idx:end_idx]
            batch_size = segment_embeds.shape[0]
            condense_embeds = self.condense_tokens.repeat(batch_size, 1, 1)
            segment_embeds = torch.cat([segment_embeds, condense_embeds], dim=1)
            
            condense_mask = torch.ones((batch_size, self.num_condense_tokens), device=self.device, dtype=torch.bool)
            segment_mask = torch.cat([segment_mask, condense_mask], dim=1)
            
            output = self.base_model(
                inputs_embeds=segment_embeds, 
                output_hidden_states=True, 
                attention_mask=segment_mask
            )
            segment_condensed = output.hidden_states[-1][:, -self.num_condense_tokens:, :]
            segment_condensed = torch.cat([segment_condensed, self.span_concat_embedding.repeat(batch_size, 1, 1)], dim=1)
            all_condensed_tokens.append(segment_condensed)
            current_level_condensed = torch.cat(all_condensed_tokens, dim=1)
            current_level_condensed = torch.cat([current_level_condensed, self.lm_embedding.repeat(batch_size, 1, 1)], dim=1)
            level_condensed_tokens.append(current_level_condensed)
            level_segment_labels.append(segment_labels)
            pre_condense_tokens.append(self.tokenizer.decode(input_validation_ids[0], skip_special_tokens=False))
        return level_condensed_tokens, level_segment_labels, pre_condense_tokens

    def loss_fn(self, logits, labels):
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        pad_token_id = self.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        labels = labels.long()
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss

    def _process_batch(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.shape[0]
        level_condensed_tokens, level_segment_labels, pre_condense_tokens = self.forward(input_ids, attention_mask=attention_mask)
        all_level_input_embeds = []
        all_level_labels = []
        original_label_length = input_ids.size(1)

        for level_tokens, level_labels in zip(level_condensed_tokens, level_segment_labels):
            condensed_tokens = torch.cat([
                self.bos_embedding.repeat(batch_size, 1, 1),
                level_tokens,
            ], dim=1)
            target_embeds = self.target_model.get_input_embeddings()(level_labels)
            ae_labels = self._pad_labels(level_labels, condensed_tokens.size(1))
            inputs_embeds = torch.cat([condensed_tokens, target_embeds], dim=1)
            all_level_input_embeds.append(inputs_embeds)
            all_level_labels.append(ae_labels)

        return all_level_input_embeds, all_level_labels, original_label_length, pre_condense_tokens

    def _pad_labels(self, labels, total_condensed_length):
        batch_size = labels.shape[0]
        padding_labels = torch.full((batch_size, total_condensed_length), -100, device=self.device)
        labels = torch.cat((padding_labels, labels), dim=1)
        return labels

    def training_step(self, batch):
        all_level_input_embeds, all_level_labels, original_label_length, pre_condense_tokens = self._process_batch(batch)
        losses = []
        for input_embeds, labels in zip(all_level_input_embeds, all_level_labels):
            output = self.target_model(inputs_embeds=input_embeds)
            logits = output.logits
            loss = self.loss_fn(logits, labels)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_start(self):
        self.text_samples = []

    def validation_step(self, batch, batch_idx):
        all_level_input_embeds, all_level_labels, original_label_length, pre_condense_tokens = self._process_batch(batch)
        losses = []
        for input_embeds, labels in zip(all_level_input_embeds, all_level_labels):
            output = self.target_model(inputs_embeds=input_embeds)
            logits = output.logits
            loss = self.loss_fn(logits, labels)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx < 5:
            for i in range(len(all_level_input_embeds)):
                max_length = original_label_length
                input_embeds = all_level_input_embeds[i]
                labels = all_level_labels[i]
                input_text = pre_condense_tokens[i]
                labels[labels==-100] = self.tokenizer.pad_token_id
                generated_text = self.generate_text(input_embeds, max_length=max_length)
                target_text = self.tokenizer.decode(labels[0], skip_special_tokens=False)
                validation_sample = [
                    input_text,
                    generated_text,
                    target_text,
                ]
                self.text_samples.append(validation_sample)
        return loss

    def generate_text(self, inputs_embeds, max_length=50):
        try:
            generated_ids = self.target_model.generate(
                inputs_embeds=inputs_embeds.to("cuda").to(self.target_model.dtype),
                max_new_tokens=max_length,
                do_sample=False,
            )
            return self.target_tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        except Exception as e:
            traceback.print_exc()
            return "Error generating text"

    def on_validation_epoch_end(self):
        try:
            val_loss = self.trainer.callback_metrics["val_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(val_loss)
        except Exception as e:
            traceback.print_exc()
    
    def _save_checkpoint(self, val_loss):
        checkpoint = {
            "modules": {
                "condense_tokens": self.condense_tokens.detach().cpu(),
                "ae_embedding": self.ae_embedding.detach().cpu(),
                "bos_embedding": self.bos_embedding.detach().cpu(),
                "span_concat_embedding": self.span_concat_embedding.detach().cpu(),
            },
        }

        checkpoint_path = os.path.join(self.output_dir, "modules.pt")
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        self.hf_api.create_repo(
            repo_id=self.hf_save_repo, repo_type="model", exist_ok=True,
        )
        self.hf_api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=checkpoint_path,
            repo_id=self.hf_save_repo,
            run_as_future=True,
            commit_description=self.commit_description + f", Val Loss: {val_loss:.6f}",
        )
        self.base_model.push_to_hub(self.hf_save_repo, commit_message=self.commit_description + f", Val Loss: {val_loss:.6f}")

    def on_validation_end(self):
        self.logger.log_table("generated_samples", data=self.text_samples)

    def configure_optimizers(self):
        param_groups = [
            {'params': [self.condense_tokens, self.ae_embedding, self.span_concat_embedding, self.lm_embedding], 'lr': 1e-4},
            {'params': self.base_model.parameters(), 'lr': 1e-4}
        ]
        return {"optimizer": torch.optim.AdamW(param_groups)}
    
    def _initialize_target_model(self, model_name_or_pretrained_path, **kwargs):
        target_model = AutoModelForCausalLM.from_pretrained(model_name_or_pretrained_path, attn_implementation="flash_attention_2")

        for _, param in target_model.named_parameters():
            param.requires_grad = False
        target_model.gradient_checkpointing_enable({"use_reentrant": False})
        return target_model

    def load_pretrained(self, pretrained_id: str):
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id=pretrained_id, 
            filename="checkpoints/modules.pt"
        )
        state_dict = torch.load(checkpoint_path)

        device = self.device
        dtype = self.dtype
        self.condense_tokens.data = state_dict["modules"]["condense_tokens"].to(
            dtype=dtype,
            device=device
        )
        self.ae_embedding.data = state_dict["modules"]["ae_embedding"].to(
            dtype=dtype,
            device=device
        )
        self.span_concat_embedding.data = state_dict["modules"]["span_concat_embedding"].to(
            dtype=dtype,
            device=device
        )
        self.lm_embedding.data = state_dict["modules"]["lm_embedding"].to(
            dtype=dtype,
            device=device
        )
