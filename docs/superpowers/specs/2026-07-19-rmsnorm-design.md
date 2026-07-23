# RMSNorm Learning Slice Design

> Historical note: this document predates the rename from InferMatrix Model Lab to Decoder Inference Lab.

## Goal

Implement and understand a minimal RMSNorm module for the decoder-only Transformer. The user owns the normalization formula and `forward()` implementation; Codex owns peripheral configuration, tests, and review.

## Why it exists

Residual updates can make hidden-state magnitudes drift across Transformer layers. RMSNorm gives Attention and MLP inputs a stable per-token scale while the residual path preserves the original hidden state.

The later pre-norm block will use it as:

```python
x = x + attention(norm_1(x))
x = x + mlp(norm_2(x))
```

## Interface

- File: `src/infermatrix_model_lab/model/norm.py`
- Class: `RMSNorm(nn.Module)`
- Constructor inputs: `d_model: int`, `eps: float`
- Trainable parameter: `weight`, shape `[d_model]`, initialized to ones
- Input: `x`, shape `[B, T, D]`
- Output: same shape and FP32 dtype as `x`; the module and input must be on the same device
- Validation: `norm_eps` must be strictly positive
- Configuration: add `norm_eps: float = 1e-5` to `ModelConfig`

## Computation

For each token independently, normalize only along the final feature dimension:

```text
mean_square = mean(x ** 2, dim=-1, keepdim=True)  # [B,T,1]
inverse_rms = rsqrt(mean_square + eps)            # [B,T,1]
normalized = x * inverse_rms                      # [B,T,D]
output = normalized * weight                      # [B,T,D]
```

RMSNorm does not subtract the feature mean. Broadcasting applies one inverse-RMS value to all features of the same token and applies the `[D]` weight across batch and sequence dimensions.

## Numerical boundary

- `eps` prevents division by zero for all-zero tokens.
- This first slice targets correct FP32 behavior.
- FP16/BF16 statistics in FP32 and dtype restoration are a separate follow-up, so mixed-precision correctness is not claimed here.

## Ownership

Codex may create the `norm.py` interface shell, add `norm_eps` validation/configuration, and write tests. The first working formula inside `RMSNorm.forward()` must be written by the user.

## Tests

1. Input `[2,10,64]` produces output `[2,10,64]`.
2. With unit `weight`, each nonzero output token has mean square approximately one.
3. All-zero input produces finite zeros rather than NaN or infinity.
4. Backward propagation reaches both `x` and `weight`.
5. CPU execution works without a hard-coded CUDA device; CUDA coverage follows when the module is explicitly moved to CUDA.

## Non-goals

- LayerNorm comparison implementation
- Fused RMSNorm or custom CUDA/Triton kernels
- Mixed-precision optimization
- Residual block assembly
- Full decoder integration
