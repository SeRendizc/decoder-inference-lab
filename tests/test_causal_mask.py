import torch

from decoder_inference_lab.model.attention import (
    _build_causal_mask,
)


def test_build_causal_mask_without_cache() -> None:
    mask = _build_causal_mask(
        query_length=4,
        key_length=4,
        past_length=0,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    assert torch.equal(mask, expected)


def test_build_causal_mask_with_cache() -> None:
    mask = _build_causal_mask(
        query_length=2,
        key_length=5,
        past_length=3,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )

    assert torch.equal(mask, expected)
