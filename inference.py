import torch
from condense_trainer_core.condense_core import LitCondenseLLM
from datasets import load_dataset


def setup_models(base_model_id, target_model_id, pretrained_condenser_id):
    """Initialize the models and tokenizers"""
    model = LitCondenseLLM(
        model_id=base_model_id,
        target_model_id=target_model_id,
        pretrained_id=pretrained_condenser_id,
        num_condense_tokens=128,  # Adjust based on your pretrained model
        max_seq_length=2048,
    )
    model = model.to("cuda")

    return model


def get_condensed_representation(model, context, max_length=2048):
    """Compress the context into condensed tokens"""
    # Tokenize context
    inputs = model.tokenizer(
        context,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Get condensed representation
    with torch.no_grad():
        condensed_tokens, _, _, _, _, _ = model.forward(
            inputs["input_ids"], inputs["attention_mask"]
        )

    # Add LM embedding
    batch_size = condensed_tokens.size(0)
    condensed_tokens_with_lm = torch.cat(
        [condensed_tokens, model.lm_embedding.repeat(batch_size, 1, 1)], dim=1
    )

    return condensed_tokens_with_lm


def get_question_embeddings(model, questions):
    """Get embeddings for the questions"""
    # Tokenize questions
    inputs = model.target_tokenizer(
        questions, padding=False, truncation=False, return_tensors="pt"
    ).to("cuda")

    # Get embeddings from target model
    q_embeddings = model.target_model.get_input_embeddings()(inputs["input_ids"])

    return q_embeddings


def generate_answer(model, inputs_embeds, max_length=128):
    """Generate answer using the combined embeddings"""
    # Create position IDs
    batch_size = inputs_embeds.size(0)
    position_ids = torch.arange(inputs_embeds.size(1), device="cuda")
    position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

    # Generate answer
    generated_text = model.generate_continuation(
        inputs_embeds=inputs_embeds, position_ids=position_ids, max_length=max_length
    )

    return generated_text


def main():
    f = open("result.csv", "w")

    # Model configuration
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your model
    target_model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your model
    pretrained_condenser_id = (
        "Condense-AI/Condenser-Llama-3.2-3B-Instruct-20241229-192753"
    )
    ds = load_dataset("simonjegou/ruler", "4096", split="test")
    ds = ds.filter(lambda x: x["task"] in ["qa_1", "qa_2"])

    # Initialize models
    model = setup_models(base_model_id, target_model_id, pretrained_condenser_id)
    model.eval()  # Set to evaluation mode

    prefix_context = "<|start_header_id|>user<|end_header_id|>\n"
    prefix_question = "\nCarefully read the above context and answer the following question. The answer should be related to the context. If not, say 'I don't know'.\n"
    postfix_question = "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    results = []
    for item in ds:
        context = item["context"]
        question = item["question"]
        gt = item["answer"]

        context = prefix_context + context
        question = prefix_question + question + postfix_question

        # Get condensed representation of context
        condensed_tokens = get_condensed_representation(model, context)
        # Get question embeddings
        q_embeddings = get_question_embeddings(model, question)

        # Combine condensed context and question embeddings
        inputs_embeds = torch.cat([condensed_tokens, q_embeddings], dim=1)
        n_condense_tokens = condensed_tokens.size(1)

        # Generate answer
        answer = generate_answer(model, inputs_embeds)
        print(f"Question: {question}")
        print(f"Answer: {answer.strip()}")
        print(f"Correct answer: {gt}")
        results.append(
            f"{question},{answer.strip()},{'-'.join(gt)},{n_condense_tokens}"
        )
        f.write("\n".join(results))


if __name__ == "__main__":
    main()
