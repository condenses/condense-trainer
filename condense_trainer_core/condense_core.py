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
        mean_compression_ratio: float = 4,
        is_pretraining: bool = False,
    ):
        super().__init__()
        self.is_pretraining = is_pretraining
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
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.target_tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def configure_model(self):
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="flash_attention_2")
        # Resize token embddings
        self.base_model.resize_token_embeddings(len(self.tokenizer))
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
        self.target_model.resize_token_embeddings(len(self.target_tokenizer))
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
        original_segment_length = self.mean_compression_ratio * self.num_condense_tokens
        num_segments = math.floor(total_length / original_segment_length)
        

        input_ids = input_ids[:, :num_segments * original_segment_length]
        segments_input_ids: list[torch.Tensor] = torch.split(input_ids, original_segment_length, dim=1)
        attention_mask = attention_mask[:, :num_segments * original_segment_length]
        segments_attention_mask: list[torch.Tensor] = torch.split(attention_mask, original_segment_length, dim=1)
        condensed_tokens = []
        for segment_input_ids, segment_attention_mask in zip(segments_input_ids, segments_attention_mask):
            segment_prompt_embeds = self.base_model.get_input_embeddings()(segment_input_ids)
            condense_input_embeds = torch.cat(
                [
                    segment_prompt_embeds,
                    self.condense_tokens.repeat(segment_prompt_embeds.size(0), 1, 1),
                ],
                dim=1
            )
            condense_attention_mask = torch.cat(
                [
                    segment_attention_mask,
                    torch.ones(segment_attention_mask.size(0), self.num_condense_tokens, device=segment_attention_mask.device),
                ],
                dim=1
            )
            output = self.base_model(
                inputs_embeds=condense_input_embeds,
                attention_mask=condense_attention_mask,
                output_hidden_states=True,
            )
            condensed_tokens.append(output.hidden_states[-1][:, -self.num_condense_tokens:, :])
        condensed_tokens = torch.cat(condensed_tokens, dim=1)
        condensed_tokens = torch.cat(
            [
                self.bos_embedding.repeat(condensed_tokens.size(0), 1, 1),
                condensed_tokens,
                self.ae_embedding.repeat(condensed_tokens.size(0), 1, 1),
            ],
            dim=1
        )
        return condensed_tokens, input_ids
    def loss_fn(self, logits, labels):
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        pad_token_id = self.target_tokenizer.pad_token_id
        labels = labels.long()
        loss = F.cross_entropy(logits, labels, ignore_index=pad_token_id)
        return loss

    def _process_batch(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        condensed_tokens, input_ids = self.forward(input_ids, attention_mask=attention_mask)
        labels_embeds = self.target_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([condensed_tokens, labels_embeds], dim=1)
        labels = torch.cat(
            [
                torch.full((condensed_tokens.size(0), condensed_tokens.size(1)), self.target_tokenizer.pad_token_id, device=input_ids.device),
                input_ids,
            ],
            dim=1
        )
        return inputs_embeds, labels, condensed_tokens
    
    def on_validation_start(self):
        self.text_samples = []

    def training_step(self, batch, batch_idx):
        inputs_embeds, labels, _ = self._process_batch(batch)
        outputs = self.target_model(inputs_embeds=inputs_embeds)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs_embeds, labels, condensed_tokens = self._process_batch(batch)
        outputs = self.target_model(inputs_embeds=inputs_embeds)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # if batch_idx < 1:
        #     generated_text = self.generate_text(condensed_tokens[:1,:, :])
        #     generated_text = generated_text.replace("<pad>", "")
        #     ground_truth_text = self.target_tokenizer.decode(labels[0, :], skip_special_tokens=False)
        #     ground_truth_text = ground_truth_text.replace("<pad>", "")
        #     self.text_samples.append((generated_text, ground_truth_text))
        return loss

    def generate_text(self, inputs_embeds, max_length=50):
        try:
            generated_ids = self.target_model.generate(
                inputs_embeds=inputs_embeds.to("cuda").to(self.target_model.dtype),
                max_new_tokens=max_length,
                do_sample=False,
                use_cache=False,
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

    def _switch_state(self, state: str):
        if state == "train":
            self.base_model.train()
            self.condense_tokens.requires_grad = True
            self.ae_embedding.requires_grad = True
            self.span_concat_embedding.requires_grad = True
            self.lm_embedding.requires_grad = True
            self.optimizers().train()
        elif state == "eval":
            self.base_model.eval()
            self.condense_tokens.requires_grad = False
            self.ae_embedding.requires_grad = False
            self.span_concat_embedding.requires_grad = False
            self.lm_embedding.requires_grad = False
            self.optimizers().eval()
    
    def _save_checkpoint(self, val_loss):
        checkpoint = {
            "modules": {
                "condense_tokens": self.condense_tokens.detach().cpu(),
                "ae_embedding": self.ae_embedding.detach().cpu(),
                "bos_embedding": self.bos_embedding.detach().cpu(),
                "span_concat_embedding": self.span_concat_embedding.detach().cpu(),
                "lm_embedding": self.lm_embedding.detach().cpu(),
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

    # def on_validation_end(self):
    #     self.logger.log_table("Validation Samples", data=self.text_samples, columns=["Generated Text", "Ground Truth Text"])

    def configure_optimizers(self):
        param_groups = [
            {'params': [self.condense_tokens, self.ae_embedding, self.span_concat_embedding, self.lm_embedding], 'lr': 1e-4},
            {'params': self.base_model.parameters(), 'lr': 1e-4}
        ]
        return {"optimizer": torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)}
    
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
        try:
            self.lm_embedding.data = state_dict["modules"]["lm_embedding"].to(
                dtype=dtype,
                device=device
            )
        except Exception as e:
            print(e)
