import torch
import torch.nn.functional as F
import traceback
import os

DEBUG = os.getenv("DEBUG", False)


class AutoencodeObjective:
    """
    Reconstructs the original text after condensing.
    """

    def __init__(self, target_model, target_tokenizer):
        """
        Args:
            target_model: A frozen or separate model used for reconstruction.
            target_tokenizer: Tokenizer for the target model.
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer

    def __call__(self, compressor, batch, segment_len=1024):
        """
        Computes the reconstruction loss.

        Args:
            compressor: The Compressor module.
            batch: A dictionary with input_ids, attention_mask, etc.
            segment_len: Segment length for context chunking.

        Returns:
            A scalar reconstruction loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.size(0)

        # Compress
        (
            condensed_states,
            segments_input_ids,
            condense_segments,
            segments_position_ids,
        ) = compressor(input_ids, attention_mask)

        device = condensed_states.device
        if DEBUG:
            print("ae", device)

        segments_input_ids = torch.cat(segments_input_ids, dim=1).to(device)

        condense_position_ids = torch.cat(segments_position_ids, dim=1).to(device)
        target_position_ids = torch.arange(
            0, segments_input_ids.size(1) + 1, device=device
        ).repeat(batch_size, 1)
        position_ids = torch.cat([condense_position_ids, target_position_ids], dim=1)
        # Append AE token
        ae_appended = torch.cat(
            [condensed_states, compressor.ae_embedding.repeat(batch_size, 1, 1)], dim=1
        )
        # Prepare ground-truth embeddings
        if DEBUG:
            print("ae", self.target_model.device)
        target_embeds = self.target_model.get_input_embeddings()(segments_input_ids)
        inputs_embeds = torch.cat([ae_appended, target_embeds], dim=1)

        # Build labels
        pad_token_id = self.target_tokenizer.pad_token_id
        fill_mask = torch.full(
            (batch_size, ae_appended.size(1)),
            pad_token_id,
            device=device,
        )
        labels = torch.cat([fill_mask, segments_input_ids], dim=1)
        if DEBUG:
            print("ae", position_ids)
            print("ae", ae_appended.size())
            print("ae", labels)
        outputs = self.target_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids
        )
        loss = self._cross_entropy_loss(outputs.logits, labels, pad_token_id)
        return loss

    def _cross_entropy_loss(self, logits, labels, pad_token_id):
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        print(labels)
        return F.cross_entropy(logits, labels.long(), ignore_index=pad_token_id)


class ContinuationObjective:
    """
    Predicts subsequent segments after partial context
    has been condensed.
    """

    def __init__(self, target_model, target_tokenizer, continuation_weight=1.0):
        """
        Args:
            target_model: A frozen or separate model for continuation tasks.
            target_tokenizer: Tokenizer for the target model.
            continuation_weight: The multiplier for continuation loss.
        """
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.continuation_weight = continuation_weight

    def __call__(self, compressor, batch, segment_len=1024):
        """
        Computes multi-level continuation loss by
        partially condensing and predicting future segments.

        Args:
            compressor: The Compressor module.
            batch: A dictionary with input_ids, attention_mask, etc.
            segment_len: Segment length for context chunking.

        Returns:
            Weighted continuation loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.size(0)

        _, segments_input_ids, condense_segments, _ = compressor(
            input_ids, attention_mask
        )
        seg_count = len(condense_segments)
        if seg_count < 2:
            return torch.tensor(0.0, device=input_ids.device)

        all_losses = []
        for i in range(1, seg_count):
            prompt = torch.cat(condense_segments[:i], dim=1)
            prompt = torch.cat(
                [prompt, compressor.lm_embedding.repeat(batch_size, 1, 1)], dim=1
            )
            # Next segments to predict
            to_predict_ids = torch.cat(segments_input_ids[i:], dim=1)
            # Build inputs_embeds (prompt + embeddings for to_predict_ids)
            to_predict_embeds = self.target_model.get_input_embeddings()(to_predict_ids)
            inputs_embeds = torch.cat([prompt, to_predict_embeds], dim=1)

            # Build labels
            pad_token_id = self.target_tokenizer.pad_token_id
            fill_mask = torch.full(
                (batch_size, prompt.size(1)), pad_token_id, device=prompt.device
            )
            labels = torch.cat([fill_mask, to_predict_ids], dim=1)
            if DEBUG:
                print("continuation", labels)
            out = self.target_model(inputs_embeds=inputs_embeds)
            loss = self._cross_entropy_loss(out.logits, labels, pad_token_id)
            all_losses.append(loss)

        if len(all_losses) > 0:
            return self.continuation_weight * torch.mean(torch.stack(all_losses))
        else:
            return torch.tensor(0.0, device=input_ids.device)

    def _cross_entropy_loss(self, logits, labels, pad_token_id):
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        return F.cross_entropy(logits, labels.long(), ignore_index=pad_token_id)

    @classmethod
    def generate(
        cls,
        target_model,
        target_tokenizer,
        inputs_embeds,
        position_ids,
        max_length,
    ):
        try:
            # Initial forward pass with condensed embeddings
            out = target_model(
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
                target_model.get_input_embeddings()(next_token_id)
                .unsqueeze(1)
                .to(inputs_embeds.device)
            )

            # Auto-regressive generation
            for _ in range(max_length):
                out = target_model(
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
                if next_token_id.item() == target_tokenizer.eos_token_id:
                    break

                # Update for next iteration
                next_inputs_embeds = (
                    target_model.get_input_embeddings()(next_token_id)
                    .unsqueeze(1)
                    .to(inputs_embeds.device)
                )
                next_position_ids = next_position_ids + 1

            return target_tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            traceback.print_exc()
            return "Error generating continuation"
