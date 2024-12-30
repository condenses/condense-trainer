import torch
import torch.nn as nn
import math
import huggingface_hub
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, PeftModel


class Compressor(nn.Module):
    """
    A module that encapsulates:
      - A base GPT-style model with LoRA
      - Learnable condense tokens
      - Specialized embeddings (autoencoder token, LM token)
      - Methods to compress contexts and generate answers
    """

    def __init__(
        self,
        model_id: str,
        tokenizer,
        lora_config: dict,
        num_condense_tokens: int,
        pretrained_id: str = None,
        compress_rate: int = 4,
    ):
        """
        Args:
            model_id: HuggingFace model name or path.
            tokenizer: Corresponding tokenizer object.
            lora_config: Dictionary with LoRA hyperparameters.
            num_condense_tokens: Number of special tokens for compression.
            pretrained_id: An optional HF repo ID from which
                           to load pretrained LoRA + compressor weights.
        """
        super().__init__()
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.lora_config = lora_config
        self.num_condense_tokens = num_condense_tokens
        self.pretrained_id = pretrained_id
        self.compress_rate = compress_rate
        self.segment_len = compress_rate * num_condense_tokens

        self._init_base_model()
        self._init_learnable_parameters()

        if self.pretrained_id is not None:
            self._load_pretrained_weights()

    def _init_base_model(self):
        """
        Initializes the base GPT-style model
        with or without pretrained LoRA weights.
        """
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        for _, param in self.base_model.named_parameters():
            param.requires_grad = True

        lora = LoraConfig(task_type="CAUSAL_LM", **self.lora_config)
        self.base_model = get_peft_model(self.base_model, lora)
        self.base_model.print_trainable_parameters()
        self.base_model.gradient_checkpointing_enable()

    def _init_learnable_parameters(self):
        """
        Initializes learnable condense tokens,
        plus specialized embeddings for autoencoder and LM tasks.
        """
        hidden_size = self.base_model.config.hidden_size

        self.condense_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, hidden_size)
        )
        self.ae_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.lm_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))

        torch.nn.init.xavier_uniform_(self.condense_tokens)
        torch.nn.init.xavier_uniform_(self.ae_embedding)
        torch.nn.init.xavier_uniform_(self.lm_embedding)

    def _load_pretrained_weights(self):
        """
        Loads previously saved condense tokens and embeddings
        from a Hugging Face repository.
        """
        print(f"Loading pretrained weights from {self.pretrained_id}")
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id=self.pretrained_id, filename="checkpoints/modules.pt"
        )
        state_dict = torch.load(checkpoint_path)
        modules = state_dict["modules"]

        self.condense_tokens.data = modules["condense_tokens"].to(
            dtype=self.condense_tokens.dtype, device=self.condense_tokens.device
        )
        self.ae_embedding.data = modules["ae_embedding"].to(
            dtype=self.ae_embedding.dtype, device=self.ae_embedding.device
        )
        self.lm_embedding.data = modules["lm_embedding"].to(
            dtype=self.lm_embedding.dtype, device=self.lm_embedding.device
        )

        self.base_model = PeftModel.from_pretrained(
            self.base_model, self.pretrained_id, is_trainable=True
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor],
        tuple[torch.Tensor],
        tuple[torch.Tensor],
    ]:
        """
        Compresses a batch of input sequences into condense tokens.
        Pads inputs to be divisible by segment_len instead of clipping.
        """
        # input_embeds = self.base_model.get_input_embeddings()(input_ids)
        # batch_size, total_length, _ = input_embeds.shape

        # # Calculate padding needed to make length divisible by segment_len
        # pad_length = (
        #     self.segment_len - (total_length % self.segment_len)
        # ) % self.segment_len
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        batch_size, total_length, _ = input_embeds.shape

        num_segments = total_length // self.segment_len
        padded_input_ids = input_ids[:, : num_segments * self.segment_len]
        padded_mask = attention_mask[:, : num_segments * self.segment_len]

        # if pad_length > 0:
        #     print(f"Padding input_ids with padding token {self.tokenizer.pad_token_id}")
        #     print(f"Pad length: {pad_length}")
        #     # Pad input_ids with padding token
        #     padding = torch.full(
        #         (batch_size, pad_length),
        #         self.tokenizer.pad_token_id,
        #         dtype=input_ids.dtype,
        #         device=input_ids.device,
        #     )
        #     padded_input_ids = torch.cat([input_ids, padding], dim=1)

        #     # Pad attention mask with zeros
        #     mask_padding = torch.zeros(
        #         batch_size,
        #         pad_length,
        #         dtype=attention_mask.dtype,
        #         device=attention_mask.device,
        #     )
        #     padded_mask = torch.cat([attention_mask, mask_padding], dim=1)
        # else:
        #     padded_input_ids = input_ids
        #     padded_mask = attention_mask

        segment_position_ids = torch.arange(
            1, self.segment_len + 1, device=input_ids.device
        ).repeat(batch_size, 1)
        tokens_per_condense = int(self.segment_len // self.num_condense_tokens)
        condense_position_ids = torch.arange(
            0,
            tokens_per_condense * self.num_condense_tokens,
            step=tokens_per_condense,
            device=input_ids.device,
        ).repeat(batch_size, 1)
        combined_position_ids = torch.cat(
            [segment_position_ids, condense_position_ids], dim=1
        )
        # Split into segments
        segments_input_ids = torch.split(padded_input_ids, self.segment_len, dim=1)
        segments_mask = torch.split(padded_mask, self.segment_len, dim=1)
        condense_segments = []
        for seg_ids, seg_m in zip(segments_input_ids, segments_mask):
            seg_embeds = self.base_model.get_input_embeddings()(seg_ids)
            seg_batch = seg_embeds.size(0)

            seg_embeds_plus_condense = torch.cat(
                [seg_embeds, self.condense_tokens.repeat(seg_batch, 1, 1)], dim=1
            )
            seg_mask_plus_condense = torch.cat(
                [
                    seg_m,
                    torch.ones(
                        seg_batch, self.num_condense_tokens, device=seg_m.device
                    ),
                ],
                dim=1,
            )

            out = self.base_model(
                inputs_embeds=seg_embeds_plus_condense,
                attention_mask=seg_mask_plus_condense,
                position_ids=combined_position_ids,
                output_hidden_states=True,
            )
            last_hidden = out.hidden_states[-1]
            condensed_part = last_hidden[:, -self.num_condense_tokens :, :]
            condense_segments.append(condensed_part)
        segments_position_ids = []
        for i in range(len(condense_segments)):
            offset_position_ids = condense_position_ids + i * self.segment_len
            segments_position_ids.append(offset_position_ids)
        # convert list to tuple
        condense_segments = tuple(condense_segments)
        segments_position_ids = tuple(segments_position_ids)
        condensed_states = torch.cat(condense_segments, dim=1)
        return (
            condensed_states,
            segments_input_ids,
            condense_segments,
            segments_position_ids,
        )

    def compress_context(self, context_ids: torch.Tensor, context_mask: torch.Tensor):
        """
        High-level API to compress an entire context into
        condensation embeddings with an appended LM embedding.

        Args:
            context_ids: [batch_size, seq_len]
            context_mask: [batch_size, seq_len]
            segment_len: Segment length for chunking the context.

        Returns:
            [batch_size, total_condensed_tokens+1, hidden_size]
        """
        condensed_states, _, _ = self.forward(context_ids, context_mask)
        bsz = condensed_states.size(0)
        condensed_with_lm = torch.cat(
            [condensed_states, self.lm_embedding.repeat(bsz, 1, 1)], dim=1
        )
        return condensed_with_lm

    def encode_question(self, question_ids: torch.Tensor):
        """
        Retrieves embeddings for question tokens.
        By default, we reuse the base model's embeddings here,
        or you may integrate a separate target model's embeddings.
        """
        return self.base_model.get_input_embeddings()(question_ids)

    def generate_from_embeds(self, inputs_embeds: torch.Tensor, max_length: int = 128):
        """
        Greedy generation starting from custom inputs_embeds.

        Args:
            inputs_embeds: [batch_size, seq_len, hidden_dim]
            max_length: How many new tokens to generate.

        Returns:
            Generated token IDs.
        """
        device = inputs_embeds.device
        batch_size, init_seq_len, _ = inputs_embeds.size()

        out = self.base_model(inputs_embeds=inputs_embeds, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated = [next_token]

        for _ in range(max_length - 1):
            next_input_embeds = self.base_model.get_input_embeddings()(next_token).to(
                device
            )
            out = self.base_model(
                inputs_embeds=next_input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = out.logits
            past_key_values = out.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated.append(next_token)

        generated_ids = torch.cat(generated, dim=1)
        return generated_ids

    def answer_context_question(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        segment_len: int,
        max_new_tokens: int = 128,
    ):
        """
        Compresses the context, obtains question embeddings,
        concatenates, and then generates an answer.

        Args:
            context_ids: [batch_size, ctx_len]
            context_mask: [batch_size, ctx_len]
            question_ids: [batch_size, q_len]
            segment_len: The chunk length for context.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated answer token IDs.
        """
        c_with_lm = self.compress_context(context_ids, context_mask, segment_len)
        q_embeds = self.encode_question(question_ids)
        full_input_embeds = torch.cat([c_with_lm, q_embeds], dim=1)
        generated_ids = self.generate_from_embeds(full_input_embeds, max_new_tokens)
        return generated_ids
