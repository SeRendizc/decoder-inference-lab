import pytest
import torch

from infermatrix_model_lab.data import ByteTokenizer, make_next_token_examples


def test_byte_tokenizer_round_trips_utf8_text() -> None:
    tokenizer = ByteTokenizer()
    text = "AI Infra，开始！"

    token_ids = tokenizer.encode(text)

    assert tokenizer.vocab_size == 257
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in token_ids)
    assert tokenizer.decode(token_ids) == text


def test_make_next_token_examples_aligns_inputs_and_targets() -> None:
    inputs, targets = make_next_token_examples([10, 11, 12, 13, 14], sequence_length=3)

    assert torch.equal(inputs, torch.tensor([[10, 11, 12], [11, 12, 13]]))
    assert torch.equal(targets, torch.tensor([[11, 12, 13], [12, 13, 14]]))
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_make_next_token_examples_rejects_short_input() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        make_next_token_examples([1, 2, 3], sequence_length=3)
