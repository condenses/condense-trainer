import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig
from .objectives import auto_encoding
from peft import get_peft_model, LoraConfig


class ModelConfig(PretrainedConfig):
    llm_model_id: str
    num_gist_tokens: int
    max_length: int
    num_auto_encoding_flag: int
    num_complete_flag: int


class GistCausalLM(nn.Module):
    """
    A wrapper for adding 'gist token' functionality to a preloaded causal language model,
    optimized to reduce overhead in the forward pass.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        num_gist_tokens: int,
        max_length: int,
    ):
        super().__init__()
        assert num_gist_tokens > 0, "num_gist_tokens must be > 0."
        assert num_gist_tokens % 2 == 0, "num_gist_tokens must be even."
        assert max_length % 2 == 0, "max_length must be even."
        assert (
            max_length % num_gist_tokens == 0
        ), "max_length must be divisible by num_gist_tokens."
        self.model = model
        self.num_gist_tokens = num_gist_tokens
        self.max_length = max_length

        # This ratio is how many original tokens each gist token represents.
        self.compress_ratio = max_length // num_gist_tokens

        # Register gist tokens as a learnable parameter (one for each chunk).
        hidden_size = self.model.config.hidden_size
        self.gist_tokens = nn.Parameter(torch.randn(num_gist_tokens, hidden_size))

        # Precompute and register index maps for attention mask expansion
        # and gist extraction. We'll store these as buffers so they're
        # moved automatically to the correct device.
        self._register_expansion_indices()
        self._register_gist_indices()

    def _register_expansion_indices(self):
        """
        Precompute the index mapping needed to expand the attention mask
        by inserting gist positions after each chunk.
        """
        # Example:
        #   T = self.max_length
        #   chunk_size = self.compress_ratio
        #   total_chunks = T // chunk_size == self.num_gist_tokens
        chunk_size = self.compress_ratio
        total_chunks = self.num_gist_tokens  # same as T // chunk_size

        idx_map = []
        gist_positions = []
        offset = 0
        for chunk_idx in range(total_chunks):
            start_pos = chunk_idx * chunk_size
            end_pos = start_pos + chunk_size
            # Original token positions
            idx_map.extend(range(start_pos, end_pos))
            # Gist position comes after each chunk
            gist_positions.append(end_pos + offset)
            offset += 1

        idx_map = torch.tensor(idx_map, dtype=torch.long)
        gist_positions = torch.tensor(gist_positions, dtype=torch.long)

        self.register_buffer("attention_idx_map", idx_map)  # [T]
        self.register_buffer(
            "attention_gist_positions", gist_positions
        )  # [num_gist_tokens]

    def _register_gist_indices(self):
        """
        Precompute the positions (in the final hidden state) where
        the gist tokens live (for extraction).
        """
        # After expansion, we have T + num_gist_tokens tokens total.
        # Gist tokens appear at indices:
        #   [chunk_size, chunk_size+(chunk_size+1), ...] with step=(chunk_size+1)
        gist_indices = torch.arange(
            self.compress_ratio,
            self.max_length + self.num_gist_tokens,
            step=self.compress_ratio + 1,
            dtype=torch.long,
        )
        self.register_buffer("gist_indices", gist_indices)

    def _expand_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Use precomputed indices to expand the attention mask in one shot.
        """
        B, T = attention_mask.shape
        assert (
            T == self.max_length
        ), f"Attention mask length {T} must match max_length {self.max_length}"

        new_length = T + self.num_gist_tokens
        # Initialize all to 0, then fill in:
        expanded = torch.zeros(
            (B, new_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # Place original mask
        expanded[:, self.attention_idx_map] = attention_mask

        # Place gist token positions to 1
        expanded[:, self.attention_gist_positions] = 1

        return expanded

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Processes input through the model, inserting gist tokens, and returns gist features.
        """
        B, T = input_ids.shape
        assert (
            T == self.max_length
        ), f"Input length must be equal to max_length: {T} != {self.max_length}"

        # Get input embeddings directly
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # [B, T, H] -> [B, num_chunks, chunk_size, H]
        chunk_size = self.compress_ratio
        chunked = inputs_embeds.reshape(
            B, -1, chunk_size, self.model.config.hidden_size
        )
        # ^ shape: (B, num_gist_tokens, chunk_size, H)

        # Expand gist tokens to [B, num_gist_tokens, H], then unsqueeze dim=2
        # to match the chunk dimension for concatenation.
        gist_tokens_expanded = self.gist_tokens.unsqueeze(0).expand(B, -1, -1)
        gist_tokens_expanded = gist_tokens_expanded.unsqueeze(2)
        # shape: (B, num_gist_tokens, 1, H)

        # Concatenate each chunk with the corresponding gist token along dim=2
        combined = torch.cat([chunked, gist_tokens_expanded], dim=2)
        # shape: (B, num_gist_tokens, chunk_size + 1, H)

        # Flatten back to [B, T + num_gist_tokens, H]
        inputs_embeds = combined.reshape(B, -1, self.model.config.hidden_size)

        # Expand attention mask (now T + num_gist_tokens in length)
        expanded_attention_mask = self._expand_attention_mask(attention_mask)

        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_attention_mask,
            output_hidden_states=True,
        )

        # Extract gist features from the last hidden state
        last_state = outputs.hidden_states[-1]  # shape: (B, T + num_gist_tokens, H)
        gist_features = last_state[:, self.gist_indices, :]
        # shape: (B, num_gist_tokens, H)

        return gist_features


class MultiSpanGistCausalLM(nn.Module):
    """
    A wrapper for gisting multiple spans of text.
    """

    def __init__(
        self,
        llm_model_id: str,
        num_gist_tokens: int,
        max_length: int,
        num_auto_encoding_flag: int = 1,
        num_complete_flag: int = 1,
        peft_config: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_id)
        self.num_gist_tokens = num_gist_tokens
        self.max_length = max_length
        self.gist_model = GistCausalLM(self.model, num_gist_tokens, max_length)

        if peft_config is not None:
            self.model = get_peft_model(self.model, LoraConfig(**peft_config))
            self.model.print_trainable_parameters()

        # Extra learnable flags
        hidden_size = self.model.config.hidden_size
        self.auto_encoding_embedding = nn.Parameter(
            torch.randn(num_auto_encoding_flag, hidden_size)
        )
        self.lm_embedding = nn.Parameter(torch.randn(num_complete_flag, hidden_size))
        self.num_auto_encoding_flag = num_auto_encoding_flag
        self.num_complete_flag = num_complete_flag

    def resize_token_embeddings(self, new_num_tokens: int):
        self.model.resize_token_embeddings(new_num_tokens)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Gist each of the multiple spans (split by max_length).
        """
        device = input_ids.device
        span_size = self.max_length
        assert (
            input_ids.shape[1] % span_size == 0
        ), "Input length must be a multiple of max_length."

        multi_span_input_ids = input_ids.split(span_size, dim=1)
        multi_span_attention_mask = attention_mask.split(span_size, dim=1)

        # Collect gist features for each chunk
        multi_span_gist_features = []
        for span_input_ids, span_attention_mask in zip(
            multi_span_input_ids, multi_span_attention_mask
        ):
            gist_feats = self.gist_model(span_input_ids, span_attention_mask)
            multi_span_gist_features.append(gist_feats)

        return (
            multi_span_gist_features,
            multi_span_input_ids,
            multi_span_attention_mask,
        )

    def forward_objective(
        self,
        multi_span_gist_features: list[torch.Tensor],
        multi_span_context_ids: list[torch.Tensor],
        multi_span_context_embeds: list[torch.Tensor],
        task: str,
        device: torch.device,
    ):
        if task == "auto_encoding":
            return auto_encoding.forward_auto_encoding(
                multi_span_gist_features,
                multi_span_context_ids,
                multi_span_context_embeds,
                self.num_gist_tokens,
                self.max_length,
                self.num_auto_encoding_flag,
                self.auto_encoding_embedding,
                device=device,
            )
        else:
            raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    # Test parameters
    batch_size = 2
    num_gist_tokens = 128
    max_length = 512
    num_spans = 3

    model_config = ModelConfig(
        llm_model_id="gpt2",
        num_gist_tokens=num_gist_tokens,
        max_length=max_length,
        num_auto_encoding_flag=2,
        num_complete_flag=1,
    )

    print("=== Testing GistCausalLM ===")
    # Initialize base model and gist model
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    gist_model = GistCausalLM(
        base_model, num_gist_tokens=num_gist_tokens, max_length=max_length
    )

    # Test single span processing
    input_ids = torch.randint(0, 50000, (batch_size, max_length))
    attention_mask = torch.ones(batch_size, max_length)

    gist_features = gist_model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"Single span gist features shape: {gist_features.shape}")
    print(
        f"Expected: (batch_size={batch_size}, num_gist_tokens={num_gist_tokens}, hidden_size={base_model.config.hidden_size})"
    )

    print("\n=== Testing MultiSpanGistCausalLM ===")
    # Initialize multi-span model
    multi_span_model = MultiSpanGistCausalLM(**model_config.to_dict())

    # Test multi-span processing
    multi_span_input_ids = torch.randint(0, 50000, (batch_size, max_length * num_spans))
    multi_span_attention_mask = torch.ones(batch_size, max_length * num_spans)

    # Forward pass through multi-span model
    multi_span_features = multi_span_model(
        input_ids=multi_span_input_ids, attention_mask=multi_span_attention_mask
    )
    print(f"Number of spans processed: {len(multi_span_features)}")
    print(f"Each span features shape: {multi_span_features[0].shape}")

    # Test auto-encoding preparation
    print("\n=== Testing Auto-Encoding Preparation ===")
    context_length = 256
    multi_span_context_ids = [
        torch.randint(0, 50000, (batch_size, context_length)) for _ in range(num_spans)
    ]
    multi_span_context_embeds = [
        torch.randn(batch_size, context_length, base_model.config.hidden_size)
        for _ in range(num_spans)
    ]

    multi_inputs_embeds, multi_labels, multi_position_ids = (
        multi_span_model.forward_objective(
            multi_span_features,
            multi_span_context_ids,
            multi_span_context_embeds,
            "auto_encoding",
        )
    )

    print("Auto-encoding outputs:")
    print(f"Number of prepared spans: {len(multi_inputs_embeds)}")
    print(f"Inputs embeds shape: {multi_inputs_embeds[0].shape}")
    print(f"Labels shape: {multi_labels[0].shape}")
    print(f"Position ids shape: {multi_position_ids[0].shape}")

    # Verify shapes and content
    expected_embed_length = (
        num_gist_tokens + multi_span_model.num_auto_encoding_flag + context_length
    )
    assert multi_inputs_embeds[0].shape == (
        batch_size,
        expected_embed_length,
        base_model.config.hidden_size,
    ), "Unexpected inputs_embeds shape"
    assert multi_labels[0].shape == (
        batch_size,
        expected_embed_length,
    ), "Unexpected labels shape"
    assert multi_position_ids[0].shape == (
        batch_size,
        expected_embed_length,
    ), "Unexpected position_ids shape"

    print("\nAll tests passed successfully!")
