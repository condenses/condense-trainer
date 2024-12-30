import torch
from transformers import AutoTokenizer
from condense_trainer_core.objectives import AutoencodeObjective, ContinuationObjective
from condense_trainer_core.condense_trainer import CondenseTrainer
from datasets import load_dataset

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16


def setup_models(base_model_id, target_model_id, pretrained_condenser_id):
    """Initialize the models and tokenizers"""
    model = CondenseTrainer(
        model_id=base_model_id,
        target_model_id=target_model_id,
        pretrained_id=pretrained_condenser_id,
        num_condense_tokens=128,
        max_seq_length=4096,
        lora_r=512,
        lora_alpha=512,
        lora_dropout=0.0,
        objectives=[AutoencodeObjective(None, None), ContinuationObjective(None, None)],
    ).to(DEVICE, dtype=TORCH_DTYPE)

    return model


def get_condensed_representation(model: CondenseTrainer, context, max_length=4096):
    """Compress the context into condensed tokens"""
    # Tokenize context
    inputs = model.target_tokenizer(
        context,
        max_length=max_length,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    inputs["input_ids"] = inputs["input_ids"].to(DEVICE)
    inputs["attention_mask"] = inputs["attention_mask"].to(DEVICE)

    # Get condensed representation
    with torch.no_grad():
        condensed_tokens, _, _, _ = model.compressor.forward(
            inputs["input_ids"], inputs["attention_mask"]
        )

    # Add LM embedding
    batch_size = condensed_tokens.size(0)
    condensed_tokens_with_lm = torch.cat(
        [
            condensed_tokens[:, :640, :],
            model.compressor.lm_embedding.repeat(batch_size, 1, 1),
        ],
        dim=1,
    )
    print(condensed_tokens_with_lm.shape)

    return condensed_tokens_with_lm


def get_question_embeddings(model, questions):
    """Get embeddings for the questions"""
    # Tokenize questions
    inputs = model.target_tokenizer(
        questions, return_tensors="pt", add_special_tokens=False
    )

    # Get embeddings from target model
    q_embeddings = model.target_model.get_input_embeddings()(
        inputs["input_ids"].to(DEVICE)
    )

    return q_embeddings


@torch.no_grad()
def generate_answer(model, inputs_embeds, max_length=128):
    """Generate answer using the combined embeddings"""
    # Create position IDs
    batch_size = inputs_embeds.size(0)
    position_ids = torch.arange(0, inputs_embeds.size(1), device=inputs_embeds.device)
    position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
    # Generate answer
    generated_text = ContinuationObjective.generate(
        model.target_model,
        model.target_tokenizer,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        max_length=max_length,
    )

    return generated_text


def main():
    # Model configuration
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your model
    target_model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your model
    pretrained_condenser_id = (
        "Condense-AI/Condenser-Llama-3.2-3B-Instruct-20241229-192753"
    )
    ds = load_dataset("Condense-AI/subnet-synthetic-dataset-v0.2", "QA", split="train")
    total_context = ""
    questions = []
    answers = []
    for i in range(10):
        total_context += "\n" + ds[i]["context_seed"]
        questions.extend(ds[i]["questions"])
        answers.extend(ds[i]["answers"])
    q_a_pairs = list(zip(questions, answers))[:4]

    questions, answers = zip(*q_a_pairs)

    # Initialize models
    model = setup_models(base_model_id, target_model_id, pretrained_condenser_id)
    model.eval()  # Set to evaluation mode
    prefix_context = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 29 Dec 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>
"""
    prefix_question = """\nCarefully read the above context and answer the following question. The answer must be in the context. If not, say "I don't know"."""
    postfix_question = "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    context = prefix_context + total_context
    print(context)
    print(questions)
    questions = [
        prefix_question + question + postfix_question for question in questions
    ]

    # Get condensed representation of context
    condensed_tokens = get_condensed_representation(model, context)

    # Process each question
    for question, ground_truth in zip(questions, answers):
        # Get question embeddings
        q_embeddings = get_question_embeddings(model, question)

        # Combine condensed context and question embeddings
        inputs_embeds = torch.cat([condensed_tokens, q_embeddings], dim=1)
        inputs_embeds = inputs_embeds.to("cuda")

        # Generate answer
        answer = generate_answer(model, inputs_embeds)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        print(f"Ground Truth: {ground_truth}")


if __name__ == "__main__":
    main()
