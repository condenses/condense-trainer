import torch


def forward_auto_encoding(
    multi_span_gist_features: list[torch.Tensor],
    multi_span_context_ids: list[torch.Tensor],
    multi_span_context_embeds: list[torch.Tensor],
    multi_span_attention_mask: list[torch.Tensor],
    num_gist_tokens: int,
    max_length: int,
    num_auto_encoding_flag: int,
    auto_encoding_embedding: torch.Tensor,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Example: building inputs_embeds & labels that incorporate gist tokens + flags + context.
    """
    multi_inputs_embeds = []
    multi_labels = []
    multi_attention_masks = []
    batch_size, context_length = multi_span_context_ids[0].shape
    position_ids = _prepare_ae_position_ids(
        batch_size, context_length, num_gist_tokens, max_length, num_auto_encoding_flag
    ).to(device)
    multi_position_ids = [position_ids] * len(multi_span_gist_features)

    # Prepare AE labels for each chunk
    for context_ids in multi_span_context_ids:
        labels = _prepare_ae_labels(
            batch_size,
            context_ids,
            num_gist_tokens,
            num_auto_encoding_flag,
        ).to(device)
        multi_labels.append(labels)

    # Prepare AE inputs_embeds for each chunk
    for span_gist_features, context_embeds, attention_mask in zip(
        multi_span_gist_features, multi_span_context_embeds, multi_span_attention_mask
    ):
        # gist + flags + context
        inputs_embeds = torch.cat(
            [
                span_gist_features,
                auto_encoding_embedding.repeat(batch_size, 1, 1),
                context_embeds,
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                torch.ones(batch_size, num_gist_tokens + num_auto_encoding_flag).to(
                    device
                ),
                attention_mask,
            ],
            dim=1,
        )
        multi_inputs_embeds.append(inputs_embeds)
        multi_attention_masks.append(attention_mask)
    return multi_inputs_embeds, multi_labels, multi_position_ids, multi_attention_masks


def _prepare_ae_labels(
    batch_size: int,
    input_ids: torch.Tensor,
    num_gist_tokens: int,
    num_auto_encoding_flag: int,
) -> torch.Tensor:
    """
    In auto-encoding, gist tokens and flags typically won't be predicted (=-100).
    """
    labels = input_ids.clone()
    blank_prefix = torch.full(
        (batch_size, num_gist_tokens + num_auto_encoding_flag), -100
    ).to(input_ids.device)
    labels = torch.cat([blank_prefix, labels], dim=1)
    return labels


def _prepare_ae_position_ids(
    batch_size: int,
    context_length: int,
    num_gist_tokens: int,
    max_length: int,
    num_auto_encoding_flag: int,
) -> torch.Tensor:
    """
    Example of building position IDs (if needed).
    """
    context_position_ids = (
        torch.arange(1, context_length + 1).unsqueeze(0).expand(batch_size, -1)
    )
    auto_encoding_position_ids = torch.zeros(
        (batch_size, num_auto_encoding_flag), dtype=torch.long
    )
    gist_step = max_length // num_gist_tokens
    gist_position_ids = (
        torch.arange(0, max_length, gist_step).unsqueeze(0).expand(batch_size, -1)
    )

    position_ids = torch.cat(
        [gist_position_ids, auto_encoding_position_ids, context_position_ids], dim=1
    )
    return position_ids
