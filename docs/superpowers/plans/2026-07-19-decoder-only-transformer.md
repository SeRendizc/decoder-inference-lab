# Decoder-Only Transformer Implementation Plan

> Historical note: this document predates the rename from InferMatrix Model Lab to Decoder Inference Lab.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The core `forward()` remains user-owned for its first implementation.

**Goal:** Build a minimal Decoder-only Transformer that maps token ids `[B,T]` to vocabulary logits `[B,T,V]`.

**Architecture:** Add learned token and absolute position embeddings, pass their sum through existing Pre-Norm Transformer blocks, then apply a final RMSNorm and an untied LM Head. Keep loss, generation, RoPE and KV Cache outside this change.

**Tech Stack:** Python 3.10+, PyTorch, pytest, Ruff

---

### Task 1: Write Decoder behavior tests first

**Files:**
- Create: `tests/test_decoder.py`
- Create: `tests/test_decoder_cuda.py`

- [ ] **Step 1: Add a shared small config and output contract test**

The test constructs `DecoderOnlyTransformer`, passes token ids shaped `[2, 6]`, and requires logits shaped `[2, 6, vocab_size]` with FP32 dtype.

- [ ] **Step 2: Add sequence-length validation test**

Pass `max_seq_len + 1` tokens and require `ValueError` mentioning `max_seq_len`.

- [ ] **Step 3: Add model-level causality test**

Create two sequences sharing the same first four tokens but differing later. In eval mode, require the first four logits to match.

- [ ] **Step 4: Add backward reachability test**

Backpropagate from `logits.square().mean()` and require non-None gradients on token embedding, position embedding, first-block Attention output projection and LM Head.

- [ ] **Step 5: Add CUDA device test**

When CUDA is available, move model and token ids to CUDA and require logits to remain on CUDA.

- [ ] **Step 6: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_decoder.py tests/test_decoder_cuda.py -q
```

Expected: collection fails because `infermatrix_model_lab.model.decoder` does not exist.

### Task 2: Create the Decoder component shell

**Files:**
- Create: `src/infermatrix_model_lab/model/decoder.py`

- [ ] **Step 1: Define imports and class**

Import PyTorch, `ModelConfig`, `TransformerBlock`, and `RMSNorm`. Define `DecoderOnlyTransformer(nn.Module)`.

- [ ] **Step 2: Assemble components in `__init__`**

Create exactly these members:

```python
self.config
self.token_embedding
self.position_embedding
self.blocks
self.final_norm
self.lm_head
```

Use `nn.ModuleList` for Blocks and `bias=False` for LM Head. Do not tie weights.

- [ ] **Step 3: Leave the learning-owned `forward()` explicit**

Keep the required signature and shape comment, then raise a descriptive `NotImplementedError`. Do not generate the completed core before the user's first attempt.

- [ ] **Step 4: Re-run Decoder tests**

Run:

```bash
python -m pytest tests/test_decoder.py tests/test_decoder_cuda.py -q
```

Expected: tests reach the class and fail specifically at the intentional `NotImplementedError`.

### Task 3: User implements the Decoder forward

**Files:**
- Modify: `src/infermatrix_model_lab/model/decoder.py`

- [ ] **Step 1: Validate rank and sequence length**

Read `batch_size, seq_len` from `[B,T]`; reject non-2D inputs and `seq_len > max_seq_len`.

- [ ] **Step 2: Create positions on the input device**

Build positions `0..T-1` with `torch.arange` on `input_ids.device`. The position shape is `[T]`.

- [ ] **Step 3: Form the initial hidden state**

Add Token Embedding `[B,T,D]` to Position Embedding `[T,D]`; PyTorch broadcasts the latter across `B`.

- [ ] **Step 4: Traverse every Block in order**

Feed the same `hidden` variable through each entry in `self.blocks`, preserving `[B,T,D]`.

- [ ] **Step 5: Produce logits**

Apply `self.final_norm`, then `self.lm_head`, and return `[B,T,V]` logits without softmax.

- [ ] **Step 6: Run focused tests and iterate from evidence**

Run:

```bash
python -m pytest tests/test_decoder.py tests/test_decoder_cuda.py -q
```

Expected: all Decoder tests pass.

### Task 4: Verify and review

**Files:**
- Review: `src/infermatrix_model_lab/model/decoder.py`
- Review: `tests/test_decoder.py`
- Review: `tests/test_decoder_cuda.py`

- [ ] **Step 1: Run the full suite**

```bash
python -m pytest -q
```

Expected: all tests pass with no warnings.

- [ ] **Step 2: Run Ruff**

```bash
