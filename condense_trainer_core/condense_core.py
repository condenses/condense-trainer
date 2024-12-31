import os
import time
import math
import traceback
import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
import huggingface_hub
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import get_peft_model, LoraConfig, PeftModel
from cut_cross_entropy.transformers import cce_patch


class LitCondenseLLM(L.LightningModule):
    """
    A LightningModule for training a "condensing" mechanism using LoRA + GPT-style models.
    """

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
        lora_dropout: float = 0.0,
        mean_compression_ratio: float = 2.0,
        is_pretraining: bool = False,
        continuation_weight: float = 1.0,
    ):
        super().__init__()
        self.is_pretraining = is_pretraining
        self.pretrained_id = pretrained_id
        self.is_pretrained = pretrained_id is not None
        self.continuation_weight = continuation_weight

        # Model IDs
        self.model_id = model_id
        self.target_model_id = target_model_id

        # Hyperparameters
        self.max_seq_length = max_seq_length
        self.num_condense_tokens = num_condense_tokens
        self.mean_compression_ratio = mean_compression_ratio
        self.original_segment_length = (
            self.mean_compression_ratio * self.num_condense_tokens
        )

        # Directories & HF Repo
        self.output_dir = output_dir
        self.hf_api = HfApi()
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        self.hf_save_repo = (
            f"Condense-AI/Condenser-{model_id.split('/')[-1]}-{time_tag}"
        )
        self.commit_description = (
            f"Condenser-{model_id.split('/')[-1]}, {target_model_id.split('/')[-1]}, "
            f"LoRA r={lora_r}, LoRA alpha={lora_alpha}, LoRA dropout={lora_dropout}"
        )

        # Tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_id)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.target_tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # LoRA config
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

        # Track best validation loss (optional if you want to monitor improvements)
        self.best_val_loss = float("inf")

    def configure_model(self):
        """
        Sets up the base model (with or without LoRA), target model, and embeddings.
        """
        # Base model
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Enable gradient on all parameters (which later might be masked by LoRA)
        for _, param in self.base_model.named_parameters():
            param.requires_grad = True

        # Wrap base model in LoRA, or load pretrained LoRA weights
        if not self.is_pretrained:
            self.base_model = get_peft_model(
                self.base_model,
                peft_config=LoraConfig(task_type="CAUSAL_LM", **self.lora_config),
            )
        else:
            self.base_model = PeftModel.from_pretrained(
                self.base_model, self.pretrained_id, is_trainable=True
            )
        self.base_model.print_trainable_parameters()
        self.base_model.gradient_checkpointing_enable()

        # Target model (frozen)
        self.target_model = self._initialize_target_model(self.target_model_id)
        self.target_model.resize_token_embeddings(len(self.target_tokenizer))

        # Hidden sizes
        self.hidden_size = self.base_model.config.hidden_size
        self.target_hidden_size = self.target_model.config.hidden_size

        # Initialize learnable parameters
        self.condense_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size)
        )
        self.ae_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.lm_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        # Initialize parameter weights
        self._initialize_embeddings()

        # If LoRA pretrained weights exist, load them
        if self.is_pretrained:
            self.load_pretrained(self.pretrained_id)

    def _initialize_embeddings(self):
        """
        Xavier initialization for learnable embeddings.
        """
        torch.nn.init.xavier_uniform_(self.condense_tokens)
        torch.nn.init.xavier_uniform_(self.ae_embedding)

    def _initialize_target_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        """
        Loads the target model and freezes its parameters.
        """
        target_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        for _, param in target_model.named_parameters():
            param.requires_grad = False

        target_model = cce_patch(target_model)

        return target_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass to transform input tokens into condensed representations.

        Returns:
            condensed_states: [batch_size, total_condensed_tokens, hidden_size]
            clipped_input_ids: truncated input IDs so they align with the segmented logic
            condensed_position_ids: position IDs for the condensed tokens
            condensed_outputs: list of per-segment condensed outputs
            segment_input_ids: list of per-segment input IDs
            segment_attn_masks: list of per-segment attention masks
        """
        # Embeddings
        prompt_embeds = self.base_model.get_input_embeddings()(input_ids)
        batch_size, total_length, _ = prompt_embeds.shape

        # Number of segments we can form from the input
        num_segments = math.floor(total_length / self.original_segment_length)

        # Truncate input to fit evenly into segments
        clipped_input_ids = input_ids[
            :, : num_segments * int(self.original_segment_length)
        ]
        clipped_attn_mask = attention_mask[
            :, : num_segments * int(self.original_segment_length)
        ]

        # Split into segments
        segment_input_ids = torch.split(
            clipped_input_ids, int(self.original_segment_length), dim=1
        )
        segment_attn_masks = torch.split(
            clipped_attn_mask, int(self.original_segment_length), dim=1
        )

        # Position IDs
        segment_position_ids = torch.arange(
            1, self.original_segment_length + 1, device=self.device
        )
        tokens_per_condense = int(
            self.original_segment_length // self.num_condense_tokens
        )
        condense_position_ids = torch.arange(
            0,
            tokens_per_condense * self.num_condense_tokens,
            step=tokens_per_condense,
            device=self.device,
        )
        combined_position_ids = torch.cat(
            [segment_position_ids, condense_position_ids], dim=0
        )
        combined_position_ids = combined_position_ids.unsqueeze(0).repeat(batch_size, 1)

        # Collect condensed tokens from each segment
        condensed_outputs = []
        for seg_ids, seg_mask in zip(segment_input_ids, segment_attn_masks):
            seg_embeds = self.base_model.get_input_embeddings()(seg_ids)
            # Append pre-initialized condense_tokens to each segment
            condense_input_embeds = torch.cat(
                [seg_embeds, self.condense_tokens.repeat(seg_embeds.size(0), 1, 1)],
                dim=1,
            )
            condense_mask = torch.cat(
                [
                    seg_mask,
                    torch.ones(
                        seg_mask.size(0),
                        self.num_condense_tokens,
                        device=seg_mask.device,
                    ),
                ],
                dim=1,
            )

            # Forward pass
            outputs = self.base_model(
                inputs_embeds=condense_input_embeds,
                attention_mask=condense_mask,
                position_ids=combined_position_ids,
                output_hidden_states=True,
            )
            # Extract the newly generated condensed tokens
            last_layer = outputs.hidden_states[-1]
            condensed_outputs.append(last_layer[:, -self.num_condense_tokens :, :])

        # Concatenate condensed tokens from all segments
        condensed_states = torch.cat(condensed_outputs, dim=1)

        # Position IDs for all condensed segments across the entire input
        all_condensed_pos_ids = []
        for i in range(len(condensed_outputs)):
            offset = i * int(self.original_segment_length)
            all_condensed_pos_ids.append(condense_position_ids + offset)
        all_condensed_pos_ids = torch.cat(all_condensed_pos_ids, dim=0).unsqueeze(0)
        all_condensed_pos_ids = all_condensed_pos_ids.repeat(batch_size, 1)

        return (
            condensed_states,
            clipped_input_ids,
            all_condensed_pos_ids,
            condensed_outputs,
            segment_input_ids,
            segment_attn_masks,
        )

    def _process_batch(self, batch):
        """
        Transform a batch of input IDs and attention masks into embeddings suitable for the target model.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Condensed representations
        (
            condensed_tokens,
            clipped_input_ids,
            condensed_pos_ids,
            condensed_outputs_list,
            segment_input_ids,
            segment_attn_masks,
        ) = self.forward(input_ids, attention_mask)

        # Prepare final position_ids for AE step
        seq_len = clipped_input_ids.size(1)
        batch_size = clipped_input_ids.size(0)
        seq_position_ids = torch.arange(0, seq_len + 1, device=clipped_input_ids.device)
        seq_position_ids = seq_position_ids.unsqueeze(0).repeat(batch_size, 1)
        ae_position_ids = torch.cat([condensed_pos_ids, seq_position_ids], dim=1)

        # Append AE embedding token to condensed tokens
        condensed_tokens_with_ae = torch.cat(
            [condensed_tokens, self.ae_embedding.repeat(batch_size, 1, 1)], dim=1
        )

        # Create embeddings for the ground truth tokens
        target_embeds = self.target_model.get_input_embeddings()(clipped_input_ids)
        inputs_embeds = torch.cat([condensed_tokens_with_ae, target_embeds], dim=1)

        # Labels: pad the condensed tokens portion so that we only predict the original text tokens
        labels = torch.cat(
            [
                torch.full(
                    (batch_size, condensed_tokens_with_ae.size(1)),
                    self.target_tokenizer.pad_token_id,
                    device=clipped_input_ids.device,
                ),
                clipped_input_ids,
            ],
            dim=1,
        )

        return (
            inputs_embeds,
            labels,
            condensed_tokens_with_ae,
            ae_position_ids,
            condensed_outputs_list,
            segment_input_ids,
            segment_attn_masks,
        )

    def _compute_continuation_loss(
        self,
        condensed_outputs_list,
        segment_input_ids,
        batch_size: int,
    ):
        """
        Compute the multi-level continuation loss.
        """
        all_losses = []
        M = len(condensed_outputs_list)

        all_level_inputs_to_generate = []

        for i in range(1, M):
            # 1) Prompt: cat condensed outputs from segments [0..i-1]
            level_i_condensed_tokens = torch.cat(condensed_outputs_list[:i], dim=1)
            max_effective_length = 256
            # Append AE token
            level_i_condensed_tokens = torch.cat(
                [level_i_condensed_tokens, self.lm_embedding.repeat(batch_size, 1, 1)],
                dim=1,
            )
            level_i_lm_inputs = torch.cat(segment_input_ids[:i], dim=1)

            # 2) LM targets: cat segment_input_ids from [i..M-1]
            level_i_lm_targets = torch.cat(segment_input_ids[i:], dim=1)
            if torch.all(level_i_lm_targets == self.target_tokenizer.pad_token_id):
                continue

            # 3) Convert to inputs_embeds for target model
            level_i_inputs_embeds = self.target_model.get_input_embeddings()(
                level_i_lm_targets
            )

            level_i_inputs_embeds = level_i_inputs_embeds[:, :max_effective_length, :]
            level_i_lm_targets = level_i_lm_targets[:, :max_effective_length]
            # Cat condensed prompt with embeddings
            final_inputs_embeds = torch.cat(
                [level_i_condensed_tokens, level_i_inputs_embeds], dim=1
            )

            # 4) Build labels
            prompt_length = level_i_condensed_tokens.size(1)
            labels = torch.cat(
                [
                    torch.full(
                        (batch_size, prompt_length),
                        self.target_tokenizer.pad_token_id,
                        device=level_i_lm_targets.device,
                    ),
                    level_i_lm_targets,
                ],
                dim=1,
            )

            # 5) Position IDs
            seq_len = level_i_lm_targets.size(1) + prompt_length
            position_ids = torch.arange(0, seq_len, device=level_i_lm_targets.device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

            labels[labels == self.target_tokenizer.pad_token_id] = -100
            # 6) Forward pass
            outputs = self.target_model(
                inputs_embeds=final_inputs_embeds,
                position_ids=position_ids,
                labels=labels,
            )

            all_level_inputs_to_generate.append(
                (
                    level_i_condensed_tokens,
                    position_ids,
                    level_i_lm_inputs,
                    level_i_lm_targets,
                )
            )

            # 7) Compute loss
            loss_i = outputs.loss
            if torch.isnan(loss_i):
                print(f"Loss is NaN at level {i}")
                print(f"Labels: {labels}")
                print(f"Position IDs: {position_ids}")
            all_losses.append(loss_i)

        # Average across levels
        if len(all_losses) > 0:
            continuation_loss = torch.mean(torch.stack(all_losses))
        else:
            continuation_loss = torch.tensor(0.0, device=self.device)

        return (
            continuation_loss,
            all_level_inputs_to_generate,
        )

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss with the target model's pad token ignored.
        """
        # Shift so next token predictions are measured
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        pad_token_id = self.target_tokenizer.pad_token_id

        return F.cross_entropy(logits, labels.long(), ignore_index=pad_token_id)

    def training_step(self, batch, batch_idx):
        (
            inputs_embeds,
            labels,
            _,
            position_ids,
            condensed_outputs_list,
            segment_input_ids,
            _,
        ) = self._process_batch(batch)

        # Reconstruction loss
        labels[labels == self.target_tokenizer.pad_token_id] = -100
        outputs = self.target_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, labels=labels
        )
        reconstruction_loss = outputs.loss

        # Continuation loss
        batch_size = labels.size(0)
        continuation_loss, _ = self._compute_continuation_loss(
            condensed_outputs_list, segment_input_ids, batch_size
        )

        # Combined loss
        total_loss = reconstruction_loss + self.continuation_weight * continuation_loss
        # total_loss = reconstruction_loss
        # Log losses
        self.log(
            "train_loss/reconstruction",
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_loss/continuation",
            continuation_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_loss/total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def on_validation_start(self):
        self.text_samples = []

    def validation_step(self, batch, batch_idx):
        (
            inputs_embeds,
            labels,
            condensed_tokens_with_ae,
            position_ids,
            condensed_outputs_list,
            segment_input_ids,
            _,
        ) = self._process_batch(batch)

        # Reconstruction loss
        original_labels = labels.clone()
        labels[labels == self.target_tokenizer.pad_token_id] = -100
        outputs = self.target_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, labels=labels
        )
        reconstruction_loss = outputs.loss

        # Continuation loss
        batch_size = labels.size(0)
        continuation_loss, all_level_inputs_to_generate = (
            self._compute_continuation_loss(
                condensed_outputs_list, segment_input_ids, batch_size
            )
        )

        # # Combined loss
        total_loss = reconstruction_loss + self.continuation_weight * continuation_loss
        # total_loss = reconstruction_loss
        # Log losses
        self.log(
            "val_loss/reconstruction",
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_loss/continuation",
            continuation_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_loss/total",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Generate some text for the first few batches
        max_generate_batch_size = 1
        n_neightbor_token = 16
        if batch_idx < 10:
            generated_text = self.generate_ae(
                condensed_tokens_with_ae[:max_generate_batch_size, :, :],
                position_ids[:max_generate_batch_size, :],
                max_length=min(128, labels.size(1)),
            )
            generated_text = generated_text.replace("<pad>", "")
            ground_truth_text = self.target_tokenizer.decode(
                original_labels[0, :], skip_special_tokens=False
            )
            ground_truth_text = ground_truth_text.replace("<pad>", "")
            # Generate continuation
            continuation_texts = []
            for level_inputs_to_generate in all_level_inputs_to_generate:
                (
                    condensed_tokens,
                    position_ids,
                    level_i_lm_inputs,
                    level_i_lm_targets,
                ) = level_inputs_to_generate
                neightbor_tokens = self.target_model.get_input_embeddings()(
                    level_i_lm_targets[:max_generate_batch_size, :n_neightbor_token]
                )
                condensed_tokens = condensed_tokens[:max_generate_batch_size, :, :]
                condensed_tokens = torch.cat(
                    [condensed_tokens, neightbor_tokens], dim=1
                )
                position_ids = position_ids[
                    :max_generate_batch_size, : condensed_tokens.size(1)
                ]
                print(condensed_tokens.shape)
                continuation = self.generate_continuation(
                    condensed_tokens[:max_generate_batch_size, :, :],
                    position_ids[:max_generate_batch_size, :],
                    max_length=min(128, level_i_lm_targets.size(1)),
                )
                continuation_ground_truth = self.target_tokenizer.decode(
                    level_i_lm_targets[:max_generate_batch_size, :][0],
                    skip_special_tokens=False,
                )
                input_text = self.target_tokenizer.decode(
                    level_i_lm_inputs[:max_generate_batch_size, :][0],
                    skip_special_tokens=False,
                )
                seed_text = self.target_tokenizer.decode(
                    level_i_lm_targets[:max_generate_batch_size, :n_neightbor_token][0],
                    skip_special_tokens=False,
                )
                full_text = (
                    input_text.replace("<pad>", "")
                    + "\n===>"
                    + seed_text
                    + "\n===>"
                    + continuation
                    + f"\n###({continuation_ground_truth.replace('<pad>', '')})"
                )
                print("-" * 100)
                print(str(full_text))
                print("-" * 100)
                continuation_texts.append(full_text)
            self.text_samples.append(
                (
                    generated_text,
                    ground_truth_text,
                    "\n||||||\n".join(continuation_texts),
                )
            )
        return total_loss

    def on_validation_end(self):
        """
        Logs a small table of generated vs ground-truth text at the end of validation.
        """
        self.logger.log_table(
            "text_samples",
            data=self.text_samples,
            columns=["Generated Text", "Ground Truth Text", "Continuation"],
        )

    def generate_continuation(self, inputs_embeds, position_ids, max_length=128):
        """
        Generates text continuation from condensed embeddings.

        Args:
            inputs_embeds (torch.Tensor): Condensed token embeddings [batch_size, seq_len, hidden_dim]
            position_ids (torch.Tensor): Position IDs for the input embeddings [batch_size, seq_len]
            max_length (int): Maximum number of tokens to generate

        Returns:
            str: Generated text continuation
        """
        try:
            # Initial forward pass with condensed embeddings
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
                if next_token_id.item() == self.target_tokenizer.eos_token_id:
                    break

                # Update for next iteration
                next_inputs_embeds = (
                    self.target_model.get_input_embeddings()(next_token_id)
                    .unsqueeze(1)
                    .to(inputs_embeds.device)
                )
                next_position_ids = next_position_ids + 1

            return self.target_tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            traceback.print_exc()
            return "Error generating continuation"

    def generate_ae(self, inputs_embeds, position_ids, max_length=128):
        """
        Greedy decoding for demonstration. Replace with your preferred generation method.
        """
        try:
            batch_size = inputs_embeds.size(0)

            # Position IDs for the initial condensed tokens
            condensed_pos_ids = position_ids[:, : inputs_embeds.size(1)]
            out = self.target_model(
                position_ids=condensed_pos_ids,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )

            past_key_values = out.past_key_values
            logits = out.logits
            next_token_id = torch.argmax(logits[:, -1], dim=-1)

            generated_ids = [next_token_id.item()]
            next_inputs_embeds = (
                self.target_model.get_input_embeddings()(next_token_id)
                .unsqueeze(1)
                .to(inputs_embeds.device)
            )
            next_position_ids = torch.tensor([1], device=inputs_embeds.device).repeat(
                batch_size, 1
            )

            for _ in range(max_length):
                out = self.target_model(
                    position_ids=next_position_ids,
                    inputs_embeds=next_inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logit = out.logits[:, -1]
                past_key_values = out.past_key_values
                next_token_id = torch.argmax(logit, dim=-1)
                generated_ids.append(next_token_id.item())

                # Update for next iteration
                next_inputs_embeds = (
                    self.target_model.get_input_embeddings()(next_token_id)
                    .unsqueeze(1)
                    .to(inputs_embeds.device)
                )
                next_position_ids = next_position_ids + 1
            return self.target_tokenizer.decode(
                generated_ids, skip_special_tokens=False
            )
        except Exception as e:
            traceback.print_exc()
            return "Error generating text"

    def configure_optimizers(self):
        """
        Returns AdamW optimizer for both LoRA parameters and the learnable tokens.
        """
        param_groups = [
            {
                "params": [self.condense_tokens, self.ae_embedding, self.lm_embedding],
                "lr": 1e-4,
            },
            {"params": self.base_model.parameters(), "lr": 1e-4},
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)
        return {"optimizer": optimizer}

    def load_pretrained(self, pretrained_id: str):
        """
        Loads pretrained LoRA + condense token weights from a Hugging Face Hub repository.
        """
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id=pretrained_id, filename="checkpoints/modules.pt"
        )
        state_dict = torch.load(checkpoint_path)

        self.condense_tokens.data = state_dict["modules"]["condense_tokens"].to(
            dtype=self.dtype, device=self.device
        )
        self.ae_embedding.data = state_dict["modules"]["ae_embedding"].to(
            dtype=self.dtype, device=self.device
        )
        try:
            self.lm_embedding.data = state_dict["modules"]["lm_embedding"].to(
                dtype=self.dtype, device=self.device
            )
        except Exception as e:
            print(e)
