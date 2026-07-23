# Decoder Inference Lab

A minimal PyTorch lab for learning, verifying, and measuring decoder-only
Transformer inference mechanisms.

The project is intentionally small enough to inspect end to end. Its purpose is
to build systems understanding of the path from tokens to logits and
generation, then use that foundation for KV-cache, prefill/decode, profiling,
and inference-optimization experiments.

## Current status

Implemented on `main`:

- Validated YAML model configuration.
- UTF-8 byte tokenizer with a reserved EOS token.
- Next-token training-example construction.
- Multi-head causal self-attention.
- RMSNorm.
- Feed-forward MLP.
- Pre-norm residual Transformer block.
- Decoder-only Transformer with token and positional embeddings.
- Greedy generation with EOS and context-length handling.
- Next-token cross-entropy.
- Optimizer-backed training step.
- Tiny-batch overfit loop.
- CPU forward, backward, causality, validation, and training tests.
- CUDA tests for attention, RMSNorm, Transformer blocks, and decoder logits.

The current WSL2/CUDA environment passes all 54 tests and Ruff. KV cache,
explicit prefill/decode APIs, attention-kernel comparisons, profiling evidence,
and inference-performance experiments remain roadmap work.

## Architecture

```text
UTF-8 text
    -> ByteTokenizer
    -> token ids
    -> DecoderOnlyTransformer
       -> token + position embeddings
       -> repeated pre-norm TransformerBlock
          -> RMSNorm
          -> CausalSelfAttention
          -> residual
          -> RMSNorm
          -> FeedForward
          -> residual
       -> final RMSNorm
       -> language-model head
    -> logits
    -> greedy generation or next-token loss
```

The implementations favor clarity and explicit invariants over peak
performance. Optimized kernels and caching will be added as measured
comparisons against this readable baseline.

## Environment

The verified development environment is:

- WSL2 Ubuntu 22.04
- Python 3.10
- PyTorch 2.12.1 with CUDA 13.0
- NVIDIA RTX 3060 Laptop GPU

The repository pins the PyTorch version but does not install a platform-specific
CUDA wheel index automatically. Select the wheel appropriate for the machine.

## Installation

```bash
git clone https://github.com/SeRendizc/decoder-inference-lab.git
cd decoder-inference-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Verify the environment

```bash
python scripts/verify_environment.py \
  --check-compile \
  --output artifacts/d1_environment.json
```

The environment report records Python, PyTorch, CUDA, GPU, matrix-multiply,
backward, and optional `torch.compile` evidence.

## Run tests

```bash
python -m pytest tests -q
python -m ruff check .
```

CUDA tests skip when CUDA is unavailable; a CPU-only pass is not presented as
CUDA validation.

## Run the tiny training experiment

```bash
python scripts/train_tiny_corpus.py --config configs/smoke.yaml
```

The script tokenizes a tiny corpus, creates next-token examples, trains the
decoder, reports loss history, and generates a short continuation. It is a
correctness and learning experiment, not a claim of useful language-model
quality.

## Configuration

Two checked-in configurations serve different purposes:

- `configs/smoke.yaml`: small and fast for correctness checks.
- `configs/lab.yaml`: larger local experiment configuration.

`ModelConfig` validates positive dimensions, head divisibility, context length,
dropout, and normalization epsilon before model construction.

## Package structure

```text
src/decoder_inference_lab/
|-- model/
|   |-- attention.py
|   |-- block.py
|   |-- decoder.py
|   |-- mlp.py
|   `-- norm.py
|-- training/
|   |-- loss.py
|   |-- overfit.py
|   `-- step.py
|-- config.py
|-- data.py
`-- generation.py
```

The core-model ownership expectations are documented in
[`src/decoder_inference_lab/model/OWNERSHIP.md`](src/decoder_inference_lab/model/OWNERSHIP.md).

## Roadmap

The next inference-specific sequence is:

1. Add an explicit KV-cache data model with correctness tests.
2. Separate prefill and decode execution paths.
3. Prove cached and uncached generation are semantically equivalent.
4. Profile latency, memory, and kernel behavior with reproducible configs.
5. Compare the readable attention baseline with PyTorch SDPA and later
   specialized kernels.

Each optimization should preserve a correctness oracle and produce evidence
that can be consumed by the broader Reliable Agentic LLM Systems portfolio.
