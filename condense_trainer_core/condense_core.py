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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

nltk.download('punkt_tab')
# Download required NLTK data
nltk.download('punkt')

class LitCondenseLLM(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        target_model_id: str,
        pretrained_id: str = None,
        num_condense_tokens: int = 386,
        max_seq_length: int = 4096,
        n_last_hidden_states: int = 2,
        output_dir: str = "checkpoints",
        lora_r: int = 128,
        lora_alpha: int = 128,
        lora_dropout: float = 0,
        mean_compression_rate: int = 4,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_id or model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
        self.model = get_peft_model(self.model, peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        self.model.print_trainable_parameters()
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.num_condense_tokens = num_condense_tokens
        self.n_last_hidden_states = n_last_hidden_states
        self.hidden_size = self.model.config.hidden_size
        self.target_decoder = self.create_target_decoder(target_model_id)
        self.separate_tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        self.base_model_hidden_size = self.target_decoder.config.hidden_size
        # Initialize learnable parameters
        self.norm = nn.LayerNorm(self.hidden_size * self.n_last_hidden_states)
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size)
        )
        self.bos_token = self.target_decoder.get_input_embeddings()(torch.tensor(self.separate_tokenizer.bos_token_id).to(self.target_decoder.device)).unsqueeze(0).unsqueeze(0)
        print(self.bos_token.shape)
        self.ae_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_size)
        )
        self.multi_span_concat_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_size)
        )
        self.linear = nn.Linear(self.hidden_size * self.n_last_hidden_states, self.base_model_hidden_size, bias=True)
        self._init_weights(self.linear)
        self._init_weights(self.norm)
        self._init_weights(self.pre_condensed_tokens)
        self._init_weights(self.ae_token)
        self.best_val_loss = float("inf")
        self.best_checkpoints = []
        self.hf_api = HfApi()
        self.hf_save_repo = f"Condense-AI/Condenser-{model_id.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.commit_description = (f"Condenser-{model_id.split('/')[-1]}, {target_model_id.split('/')[-1]}, "
                                   f"LoRA r={lora_r}, LoRA alpha={lora_alpha}, LoRA dropout={lora_dropout}")
        self.output_dir = output_dir
        self.mean_compression_rate = mean_compression_rate
        
        # Add scorer initialization
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, prompt_embeds, attention_mask) -> torch.Tensor:
        # Calculate number of segments based on input length
        total_length = prompt_embeds.size(1)
        num_segments = math.ceil(total_length / (self.num_condense_tokens * self.mean_compression_rate))
        segment_length = math.ceil(total_length / num_segments)
        
        all_condensed_tokens = []
        
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            
            # Get segment embeddings and attention mask
            segment_embeds = prompt_embeds[:, start_idx:end_idx, :]
            segment_mask = attention_mask[:, start_idx:end_idx]
            # Add pre-condensed tokens to segment
            n_batch = segment_embeds.shape[0]
            pre_condensed_embeds = self.pre_condensed_tokens.repeat(n_batch, 1, 1)
            segment_embeds = torch.cat([segment_embeds, pre_condensed_embeds], dim=1)
            
            # Create attention mask where pre-condensed tokens can only attend to non-padding tokens
            pre_condensed_mask = torch.ones((n_batch, self.num_condense_tokens), device=segment_mask.device, dtype=torch.bool)
            segment_mask = torch.cat([segment_mask, pre_condensed_mask], dim=1)
            
            # Modified attention mask creation - keep it 2D for the model
            # The model will handle the proper attention pattern internally
            segment_mask = segment_mask.bool()
            # Process segment through model
            output = self.model(
                inputs_embeds=segment_embeds, 
                output_hidden_states=True, 
                attention_mask=segment_mask  # Now passing 2D mask
            )
            hidden_states = output.hidden_states[-self.n_last_hidden_states:]
            concated_hidden_states = torch.cat(hidden_states, dim=-1)
            
            # Get condensed tokens for this segment
            segment_condensed = concated_hidden_states[:, -self.num_condense_tokens:, :]
            segment_condensed = self.linear(self.norm(segment_condensed))
            segment_condensed = torch.cat([segment_condensed, self.multi_span_concat_token.repeat(n_batch, 1, 1)], dim=1)
            all_condensed_tokens.append(segment_condensed)

        # Concatenate all segments
        condensed_tokens = torch.cat(all_condensed_tokens, dim=1)
        return condensed_tokens

    def loss_fn(self, logits, labels):
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        pad_token_id = self.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        labels = labels.long()
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss

    def _process_batch(self, batch):
        context_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        n_batch = context_ids.shape[0]
        
        # Get embeddings
        context_embeds = self.model.get_input_embeddings()(context_ids)
        uncondensed_embeds = self.target_decoder.get_input_embeddings()(labels)
        
        # Get condensed tokens for all segments
        condensed_tokens = self.forward(context_embeds, attention_mask=attention_mask)
        
        # Add BOS and AE tokens
        condensed_tokens = torch.cat([
            self.bos_token.repeat(n_batch, 1, 1),
            condensed_tokens,
            self.ae_token.repeat(n_batch, 1, 1)
        ], dim=1)

        original_label_length = labels.size(1)
        # Prepare labels with padding
        total_condensed_length = condensed_tokens.size(1)
        padding_labels = torch.full((n_batch, total_condensed_length), -100, device=context_ids.device)
        labels = torch.cat((padding_labels, labels), dim=1)
        
        # Combine embeddings
        inputs_embeds = torch.cat([condensed_tokens, uncondensed_embeds], dim=1)
        return inputs_embeds, labels, condensed_tokens, original_label_length

    def training_step(self, batch):
        inputs_embeds, labels, condensed_tokens, original_label_length = self._process_batch(batch)
        output = self.target_decoder(inputs_embeds=inputs_embeds)
        logits = output.logits
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_start(self):
        self.text_samples = []

    def validation_step(self, batch, batch_idx):
        inputs_embeds, labels, condensed_tokens, original_label_length = self._process_batch(batch)
        output = self.target_decoder(inputs_embeds=inputs_embeds)
        logits = output.logits
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Generate text and calculate metrics
        if batch_idx < 5:  # Keep the sample limit
            max_length = original_label_length
            print("Generating text... max_length:", max_length)
            generated_text = self.generate_text(condensed_tokens[0].unsqueeze(0), max_length=max_length)
            input_text = batch["str_context"][0]
            target_text = batch["str_uncondensed"][0]
            
            # Calculate BLEU-4
            reference_tokens = nltk.word_tokenize(target_text)
            candidate_tokens = nltk.word_tokenize(generated_text)
            bleu_score = sentence_bleu([reference_tokens], candidate_tokens, 
                                     weights=(0.25, 0.25, 0.25, 0.25),
                                     smoothing_function=self.smoothing)
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(target_text, generated_text)
            
            # Log metrics
            self.log("val_bleu4", bleu_score, on_step=False, on_epoch=True)
            self.log("val_rouge1", rouge_scores['rouge1'].fmeasure, on_step=False, on_epoch=True)
            self.log("val_rouge2", rouge_scores['rouge2'].fmeasure, on_step=False, on_epoch=True)
            self.log("val_rougeL", rouge_scores['rougeL'].fmeasure, on_step=False, on_epoch=True)
            
            self.text_samples.append([
                input_text,
                generated_text,
                target_text,
            ])
        return loss

    def generate_text(self, inputs_embeds, max_length=50):
        """
        Generate text based on the given inputs_embeds.
        """
        try:
            generated_ids = self.target_decoder.generate(
                inputs_embeds=inputs_embeds.to("cuda").to(self.target_decoder.dtype),
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.1,
            )
            generated_text = self.separate_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            traceback.print_exc()
            print(f"Error during text generation: {e}")
            return "Error generating text"


    
    def on_validation_epoch_end(self):
        try:
            val_loss = self.trainer.callback_metrics["val_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Save only the main model state dict
                checkpoint = {
                    "modules": {
                        "pre_condensed_tokens": self.pre_condensed_tokens.detach().cpu(),
                        "linear_state_dict": self.linear.state_dict(),
                        "norm_state_dict": self.norm.state_dict(),
                        "ae_token": self.ae_token.detach().cpu(),
                        "bos_token": self.bos_token.detach().cpu(),
                        "multi_span_concat_token": self.multi_span_concat_token.detach().cpu(),
                    },
                }

                checkpoint_path = os.path.join(self.output_dir, "modules.pt")
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                # Push to HuggingFace Hub
                self.hf_api.create_repo(
                    repo_id=self.hf_save_repo, repo_type="model", exist_ok=True,
                )
                self.hf_api.upload_file(
                    path_or_fileobj=checkpoint_path,
                    path_in_repo=checkpoint_path,
                    repo_id=self.hf_save_repo,
                    run_as_future=True,
                    commit_description=self.commit_description + f", Val Loss: {val_loss:.4f}",
                )
                self.model.push_to_hub(self.hf_save_repo)
        except Exception as e:
            traceback.print_exc()
            print(f"Error in on_validation_epoch_end: {e}")
    
    def on_validation_end(self):
        self.logger.log_table("generated_samples", data=self.text_samples)

    def configure_optimizers(self):
        # Define parameter groups with different learning rates
        group_lr = [
            {
                'params': self.pre_condensed_tokens,
                'lr': 1e-4
            },
            {
                'params': self.linear.parameters(),
                'lr': 1e-4  # Lower learning rate for linear layer
            },
            {
                'params': self.norm.parameters(), 
                'lr': 1e-4  # Lower learning rate for norm layer
            },
            {
                'params': self.model.parameters(),
                'lr': 1e-4  # Lower learning rate for base model
            },
            {
                "params": self.ae_token,
                "lr": 1e-4
            }
        ]
        optimizer = torch.optim.AdamW(
            group_lr, weight_decay=1e-5
        )
        return {
            "optimizer": optimizer,
        }
    
    def create_target_decoder(self, model_name_or_pretrained_path, **kwargs):
        target_decoder = AutoModelForCausalLM.from_pretrained(model_name_or_pretrained_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")

        for _, param in target_decoder.named_parameters():
            param.requires_grad = False
        # Enable gradient checkpointing to reduce memory usage during training
        # Setting use_reentrant=False avoids potential issues with backward pass recomputation
        target_decoder.gradient_checkpointing_enable({"use_reentrant": False})
        return target_decoder

    @classmethod
    def from_pretrained(cls, condense_model_id: str, target_model_id: str, pretrained_id: str, checkpoint_path: str = None):
        """Load a pretrained Condenser model."""
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id=pretrained_id, 
            filename="checkpoints/modules.pt"
        )
        
        state_dict = torch.load(checkpoint_path)
        num_condense_tokens = state_dict["modules"]["pre_condensed_tokens"].shape[1]
        n_last_hidden_states = 2  # This is hardcoded in inference.py
        
        # Initialize model
        model = cls(
            model_id=condense_model_id,
            target_model_id=target_model_id,
            pretrained_id=pretrained_id,
            num_condense_tokens=num_condense_tokens,
            n_last_hidden_states=n_last_hidden_states
        )
        
        # Load state dict for the learnable parameters
        model.pre_condensed_tokens.data = state_dict["modules"]["pre_condensed_tokens"].to(
            dtype=model.pre_condensed_tokens.dtype,
            device=model.pre_condensed_tokens.device
        )
        model.linear.load_state_dict(state_dict["modules"]["linear_state_dict"]).to(model.linear.device)
        model.norm.load_state_dict(state_dict["modules"]["norm_state_dict"]).to(model.norm.device)
        model.ae_token.data = state_dict["modules"]["ae_token"].to(
            dtype=model.ae_token.dtype,
            device=model.ae_token.device
        )
        model.multi_span_concat_token.data = state_dict["modules"]["multi_span_concat_token"].to(
            dtype=model.multi_span_concat_token.dtype,
            device=model.multi_span_concat_token.device
        )
        
        return model