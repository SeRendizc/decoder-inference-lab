# Pre-Norm Transformer Block Learning Slice Design

> Historical note: this document predates the rename from InferMatrix Model Lab to Decoder Inference Lab.

## Goal

Combine the user-implemented RMSNorm, causal self-attention, and feed-forward modules into one decoder-only pre-norm residual Transformer Block that the user can implement and explain.

## Why residual pre-norm exists

Each sublayer learns a correction `F(x)` instead of reconstructing the complete hidden state. The identity path preserves the current representation and provides a direct gradient path. RMSNorm gives each Attention or MLP sublayer a stable-scale input without normalizing away the residual stream itself.

## Interface

- File: `src/infermatrix_model_lab/model/block.py`
- Class: `TransformerBlock(nn.Module)`
- Constructor input: `ModelConfig`
- Input: `x`, shape `[B,T,D]`
- Output: shape `[B,T,D]`
- The module and input must be explicitly placed on the same device.

## Components

Each block owns four independent submodules:

```text
norm_1    RMSNorm(config.d_model, config.norm_eps)
attention CausalSelfAttention(config)
norm_2    RMSNorm(config.d_model, config.norm_eps)
mlp       FeedForward(config)
```

`norm_1` and `norm_2` do not share their trainable `[D]` weights because they normalize two different residual-stream states.

## Data flow

The user-owned `forward()` implements exactly two pre-norm residual updates:

```text
attention_input  = norm_1(x)                         [B,T,D]
attention_delta  = attention(attention_input)        [B,T,D]
x                = x + attention_delta               [B,T,D]

mlp_input        = norm_2(x)                         [B,T,D]
mlp_delta        = mlp(mlp_input)                    [B,T,D]
x                = x + mlp_delta                     [B,T,D]
return x
```

The Attention and MLP run sequentially, not in parallel: the MLP sees the residual stream after the Attention update.

## Responsibility boundaries

- `CausalSelfAttention` continues to own projection, head reshape, scaling, causal masking, softmax, and output projection.
- `FeedForward` continues to own expansion, GELU, contraction, and its configured Dropout.
- `RMSNorm` continues to own per-token scale normalization and its trainable feature weight.
- `TransformerBlock` owns only composition order and the two residual additions.
- No extra Attention Dropout is introduced in this slice.

## Tests

1. FP32 CPU input `[2,10,64]` returns `[2,10,64]` with the same dtype.
2. Zeroing all Attention and MLP parameters makes the block return the input exactly, proving the identity residual path.
3. Backward propagation reaches the input, a representative Attention parameter, and a representative MLP parameter.
4. Causal semantics remain covered by the existing Attention tests rather than duplicated in Block tests.
5. Explicitly moving the block and input to CUDA produces a CUDA output with the same shape.

All deterministic tests use `dropout=0.0`.

## Ownership

Codex may create `block.py` with constructor wiring and an intentional `NotImplementedError`, then add tests. The first working two-update `forward()` must be written by the user.

## Error handling

The block adds no manual shape or device validation. PyTorch and the owned submodules provide actionable errors for a wrong final feature dimension or mismatched devices. This avoids duplicating checks at every composition layer.

## Non-goals

- Embedding or positional encoding
- Stacking multiple blocks
- Final RMSNorm or language-model head
- Loss, optimizer, or training loop
- KV Cache or generation
- Mixed-precision policy
- Fused kernels or additional dropout variants
