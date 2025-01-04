import torch


def forward_completing(
    multi_span_gist_features: list[torch.Tensor],
    multi_span_context_ids: list[torch.Tensor],
    multi_span_context_embeds: list[torch.Tensor],
    multi_span_attention_mask: list[torch.Tensor],
    num_gist_tokens: int,
    num_auto_encoding_flag: int,
    lm_embedding: torch.Tensor,
    device: torch.device,
    **kwargs,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Example: building inputs_embeds & labels that incorporate gist tokens + flags + context.
    """
    multi_gisted_context_ids = []
    multi_inputs_embeds = []
    multi_labels = []
    multi_attention_masks = []
    multi_inputs_embeds_for_generate = []
    multi_gisted_position_ids = []
    batch_size, context_length = multi_span_context_ids[0].shape
    total_span = len(multi_span_context_ids)

    multi_position_ids = []
    for i in range(1, total_span):
        n_seed_token = 8
        span_gist_features = torch.cat(multi_span_gist_features[:i], dim=1)
        label_ids = torch.cat(multi_span_context_ids[i:], dim=1)
        attention_mask = torch.cat(multi_span_attention_mask[i:], dim=1)
        label_embeds = torch.cat(multi_span_context_embeds[i:], dim=1)
        gisted_context_ids = torch.cat(
            multi_span_context_ids[:i],
            dim=1,
        )
        gisted_context_ids = torch.cat(
            [
                gisted_context_ids,
                label_ids[:, :n_seed_token],
            ],
            dim=1,
        )
        multi_gisted_context_ids.append(gisted_context_ids)
        labels = _prepare_ae_labels(
            batch_size,
            label_ids,
            span_gist_features.shape[1],
            num_auto_encoding_flag,
        ).to(device)
        multi_labels.append(labels)

        # gist + flags + context
        inputs_embeds = torch.cat(
            [
                span_gist_features,
                lm_embedding.repeat(batch_size, 1, 1),
                label_embeds,
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                torch.ones(
                    batch_size, span_gist_features.shape[1] + num_auto_encoding_flag
                ).to(device),
                attention_mask,
            ],
            dim=1,
        )
        position_ids = (
            torch.arange(0, inputs_embeds.shape[1]).repeat(batch_size, 1).to(device)
        )
        multi_position_ids.append(position_ids)
        multi_inputs_embeds.append(inputs_embeds)
        multi_attention_masks.append(attention_mask)
        multi_inputs_embeds_for_generate.append(
            inputs_embeds[
                :, : num_gist_tokens + num_auto_encoding_flag + n_seed_token, :
            ]
        )
        multi_gisted_position_ids.append(
            position_ids[:, : num_gist_tokens + num_auto_encoding_flag + n_seed_token]
        )
    return (
        multi_inputs_embeds,
        multi_labels,
        multi_position_ids,
        multi_attention_masks,
        multi_inputs_embeds_for_generate,
        multi_gisted_context_ids,
        multi_gisted_position_ids,
    )


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
