from __future__ import annotations

import torch

from infermatrix_model_lab.config import ModelConfig
from infermatrix_model_lab.data import ByteTokenizer, make_next_token_examples
from infermatrix_model_lab.model.decoder import DecoderOnlyTransformer
from infermatrix_model_lab.training.loss import NextTokenCrossEntropy
from infermatrix_model_lab.training.overfit import overfit_tiny_batch


def main() -> None:
    torch.manual_seed(7)

    text = "hello model lab\n" * 8
    sequence_length = 8
    steps = 200

    tokenizer = ByteTokenizer()
    token_ids = tokenizer.encode(text)

    input_ids, targets = make_next_token_examples(
        token_ids,
        sequence_length,
    )

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=sequence_length,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )

    model = DecoderOnlyTransformer(config)
    loss_fn = NextTokenCrossEntropy()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
    )

    losses = overfit_tiny_batch(
        model,
        loss_fn,
        optimizer,
        input_ids,
        targets,
        steps,
    )

    initial_loss = losses[0]
    final_loss = losses[-1]

    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"initial loss: {initial_loss:.4f}")
    print(f"final loss:   {final_loss:.4f}")
    print(f"reduction:    {reduction:.2f}%")

    model.eval()

    with torch.no_grad():
        logits = model(input_ids)
        predicted_ids = logits.argmax(dim=-1)

    correct = predicted_ids == targets
    accuracy = correct.float().mean().item()

    print(f"token accuracy: {accuracy:.2%}")

    sample_index = 0

    sample_input_ids = input_ids[sample_index].tolist()
    sample_target_ids = targets[sample_index].tolist()
    sample_predicted_ids = predicted_ids[sample_index].tolist()

    print(f"input ids:      {sample_input_ids}")
    print(f"target ids:     {sample_target_ids}")
    print(f"predicted ids:  {sample_predicted_ids}")

    print(f"input text:     {tokenizer.decode(sample_input_ids)!r}")
    print(f"target text:    {tokenizer.decode(sample_target_ids)!r}")
    print(f"predicted text: {tokenizer.decode(sample_predicted_ids)!r}")


if __name__ == "__main__":
    main()
