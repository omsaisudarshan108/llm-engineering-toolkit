# Attention Mechanisms

Scaled dot-product and multi-head attention from scratch.

## Files

| File | Purpose | Run Command |
|------|---------|-------------|
| `attention_from_scratch.py` | Complete attention implementation | `python attention_from_scratch.py` |

## Quick Start

```bash
python attention_from_scratch.py
```

## The Attention Equation

```
                    Q × K^T
Attention(Q,K,V) = softmax(─────────) × V
                      √d_k
```

## Math-to-Code Mapping

```python
# Step 1: Compute attention scores
scores = torch.matmul(Q, K.transpose(-2, -1))  # Q × K^T

# Step 2: Scale by √d_k (prevents vanishing gradients)
scaled_scores = scores / math.sqrt(d_k)

# Step 3: Apply mask (for causal/autoregressive models)
scaled_scores = scaled_scores + mask  # -inf for masked positions

# Step 4: Softmax (convert to probabilities)
attention_weights = F.softmax(scaled_scores, dim=-1)

# Step 5: Weighted sum of values
output = torch.matmul(attention_weights, V)
```

## Demos Included

1. **Basic Attention**: Q, K, V computation and visualization
2. **Temperature Scaling**: Control attention sharpness
3. **Causal Masking**: Prevent attending to future tokens
4. **Multi-Head Attention**: Parallel attention with different heads
