import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig
from .objectives import auto_encoding, completing
from peft import get_peft_model, LoraConfig, PeftModel
import os
from omegaconf import OmegaConf
from huggingface_hub import HfApi


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
        self._register_buffer()

    def _register_buffer(self):
        # Register position IDs
        position_ids_with_gist = torch.cat(
            [
                torch.arange(0, self.max_length),
                torch.arange(0, self.max_length, step=self.compress_ratio),
            ],
            dim=0,
        )
        gist_attention_mask = torch.ones(self.num_gist_tokens)
        self.register_buffer("position_ids", position_ids_with_gist)
        self.register_buffer("gist_attention_mask", gist_attention_mask)

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
        inputs_embeds_with_gist = torch.cat(
            [inputs_embeds, self.gist_tokens.unsqueeze(0).expand(B, -1, -1)], dim=1
        )
        attention_mask_with_gist = torch.cat(
            [attention_mask, self.gist_attention_mask.repeat(B, 1)], dim=1
        )
        position_ids_with_gist = self.position_ids.repeat(B, 1)
        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds_with_gist,
            attention_mask=attention_mask_with_gist,
            position_ids=position_ids_with_gist,
            output_hidden_states=True,
        )

        # Extract gist features from the last hidden state
        last_state = outputs.hidden_states[-1]  # shape: (B, T + num_gist_tokens, H)
        gist_features = last_state[:, -self.num_gist_tokens :, :]

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
            print(f"Applying PEFT config: {peft_config}")
            peft_config = OmegaConf.to_container(peft_config)
            print(type(peft_config))
            print(type(peft_config["target_modules"]))
            self.model = get_peft_model(self.model, LoraConfig(**dict(peft_config)))
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
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, padding_id: int
    ) -> torch.Tensor:
        """
        Gist each of the multiple spans (split by max_length).
        """
        span_size = self.max_length
        assert (
            input_ids.shape[1] % span_size == 0
        ), "Input length must be a multiple of max_length."

        multi_span_input_ids = input_ids.split(span_size, dim=1)
        multi_span_attention_mask = attention_mask.split(span_size, dim=1)
        for i, (span_input_ids, span_attention_mask) in enumerate(
            zip(multi_span_input_ids, multi_span_attention_mask)
        ):
            if span_input_ids.all() == padding_id:
                # Drop this span
                multi_span_input_ids.pop(i)
                multi_span_attention_mask.pop(i)

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
        multi_span_attention_mask: list[torch.Tensor],
        task: str,
        device: torch.device,
    ):
        if task == "auto_encoding":
            return auto_encoding.forward_auto_encoding(
                multi_span_gist_features,
                multi_span_context_ids,
                multi_span_context_embeds,
                multi_span_attention_mask,
                self.num_gist_tokens,
                self.max_length,
                self.num_auto_encoding_flag,
                self.auto_encoding_embedding,
                device=device,
            )
        elif task == "completing":
            return completing.forward_completing(
                multi_span_gist_features,
                multi_span_context_ids,
                multi_span_context_embeds,
                multi_span_attention_mask,
                self.num_gist_tokens,
                self.num_complete_flag,
                self.lm_embedding,
                device=device,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def push_to_hub(self, repo_id: str, output_dir: str, hf_api: HfApi):
        self.model.push_to_hub(repo_id)
        checkpoint = {
            "lm_embedding": self.lm_embedding,
            "auto_encoding_embedding": self.auto_encoding_embedding,
            "gist_tokens": self.gist_model.gist_tokens,
        }
        checkpoint_path = os.path.join(output_dir, "modules.pt")
        torch.save(checkpoint, checkpoint_path)
        hf_api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=checkpoint_path,
            repo_id=repo_id,
        )

    def load_pretrained(self, repo_id: str, hf_api: HfApi):
        checkpoint_path = hf_api.hf_hub_download(
            repo_id=repo_id, filename="checkpoints/modules.pt"
        )

        checkpoint = torch.load(
            checkpoint_path, map_location=self.lm_embedding.device, weights_only=True
        )

        self.lm_embedding.data = checkpoint["lm_embedding"]
        self.auto_encoding_embedding.data = checkpoint["auto_encoding_embedding"]
        self.gist_model.gist_tokens.data = checkpoint["gist_tokens"]

        print("Loading llm model...")
        self.gist_model.model = PeftModel.from_pretrained(
            self.gist_model.model, repo_id, is_trainable=True, device_map="auto"
        )

        self.eval()


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
