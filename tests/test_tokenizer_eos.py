from infermatrix_model_lab.data import ByteTokenizer


def test_byte_tokenizer_reserves_eos_token() -> None:
    tokenizer = ByteTokenizer()

    assert tokenizer.eos_token_id == 256
    assert tokenizer.vocab_size == 257


def test_byte_tokenizer_decode_stops_at_eos() -> None:
    tokenizer = ByteTokenizer()

    token_ids = [
        104,
        105,
        tokenizer.eos_token_id,
        120,
    ]

    decoded = tokenizer.decode(token_ids)

    assert decoded == "hi"


def test_byte_tokenizer_decode_replaces_invalid_utf8() -> None:
    tokenizer = ByteTokenizer()

    decoded = tokenizer.decode([255])

    assert decoded == "�"
