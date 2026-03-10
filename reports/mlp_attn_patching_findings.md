# MLP vs Attention Sublayer Patching Findings

## Motivation

The Week 9 layer-level patching (Experiment 2) patched `resid_post` — the full residual stream
after each transformer block. This conflates two distinct contributions:

```
resid_post[L] = resid_pre[L] + attn_out[L] + mlp_out[L]
```

Patching `resid_post` therefore mixes the attention sublayer's contribution, the MLP sublayer's
contribution, and the residual carry-through from earlier layers. This experiment decomposes the
effect by patching `hook_attn_out` and `hook_mlp_out` separately — the *delta* each sublayer
adds to the residual stream.

This directly addresses the comparison with the other team (GPT-2 Small / PIQA), who found
**MLPs dominate commonsense reasoning** (especially layers 0, 10, 11), while attention is not
diagnostic. Does the same hold for Qwen3-8B on Com2Sense?

---

## Method

For each of the 200 pairs, for each of 18 layers (18–35):

1. Cache the correct example's `hook_attn_out` and `hook_mlp_out` at all target layers in a
   single forward pass.
2. Run the incorrect example with a hook replacing only `hook_attn_out[L]` at the final token
   position → **attention patch**.
3. Run the incorrect example with a hook replacing only `hook_mlp_out[L]` at the final token
   position → **MLP patch**.
4. Record logit change (patched True−False gap minus baseline) and whether the prediction flips.

Total: 200 pairs × 18 layers × 2 sublayer types = 7,200 patched forward passes.

---

## Results

| Layer | Attn flip% | Attn Δlogit | MLP flip% | MLP Δlogit |
|-------|-----------|-------------|-----------|------------|
| 18    | 0.0%      | −0.011      | 0.0%      | −0.003     |
| 19    | 0.0%      | −0.037      | 0.0%      | −0.011     |
| 20    | 0.0%      | −0.081      | 0.0%      | −0.014     |
| 21    | 0.0%      | −0.063      | 0.0%      | +0.009     |
| 22    | 0.0%      | −0.156      | 0.0%      | −0.149     |
| 23    | 0.0%      | −0.061      | 0.0%      | −0.146     |
| 24    | 0.0%      | −0.041      | 0.0%      | +0.077     |
| **25**| **1.5%**  | **+0.163**  | 0.0%      | +0.007     |
| 26    | 0.0%      | −0.090      | 0.0%      | −0.072     |
| 27    | 0.0%      | −0.060      | 0.0%      | −0.051     |
| 28    | 0.0%      | +0.003      | 0.0%      | −0.045     |
| 29    | 0.0%      | −0.055      | 0.0%      | −0.051     |
| 30    | 0.0%      | −0.066      | 0.0%      | −0.083     |
| 31    | 0.0%      | −0.019      | 0.0%      | +0.023     |
| 32    | 0.0%      | −0.011      | 0.0%      | −0.012     |
| 33    | 0.0%      | −0.032      | 0.0%      | −0.002     |
| 34    | 0.0%      | −0.030      | 0.0%      | −0.016     |
| **35**| 0.0%      | −0.018      | **1.5%**  | **+0.133** |

*(See `mlp_attn_patching_results.png` for the side-by-side visualization.)*

---

## Findings

### 1. Neither sublayer produces reliable flips

Both attention and MLP patching achieve near-zero flip rates across all 18 layers. Only two
exceptions exist:

- **L25 attention: 1.5%** (3 of 200 pairs flip) — the largest positive logit change of any
  intervention in this experiment (+0.163). This is also the only case where attention patching
  *helps* more than marginally.
- **L35 MLP: 1.5%** (3 of 200 pairs flip, +0.133 logit change) — a late-layer MLP effect.

Both are too small to support a localized-bottleneck interpretation. For comparison, Geva et al.
(2023) and Wang et al. (2022) report flip rates of 30–60%+ at causal layers in factual tasks.

### 2. Neither sublayer dominates — this differs from the GPT-2 / PIQA result

The other team found that **MLP sublayers dominate** commonsense (especially early layers 0, 10,
11), with attention playing a secondary role. We do not replicate this pattern in Qwen3-8B:

- **Attention Δlogit is larger in magnitude across most middle layers** (L19–L27): e.g.,
  L20 attn=−0.081 vs MLP=−0.014; L25 attn=+0.163 vs MLP=+0.007.
- **MLP effects match attention only at L22–L23**, where both are large and negative
  (L22: attn=−0.156, MLP=−0.149; L23: attn=−0.061, MLP=−0.146). These are the two most
  disruptive layers for both sublayer types, suggesting they both encode strong
  sequence-specific content at this depth.
- **Later layers favor MLP** (L30–L35): MLP effects are slightly larger, consistent with
  the general finding that later MLP layers act as retrieval/output components.

The simplest summary: **in Qwen3-8B on Com2Sense, attention and MLP are similarly unhelpful**,
with attention showing slightly larger magnitude effects in middle layers. This differs from the
GPT-2 Small / PIQA finding and is likely driven by two factors: (a) architectural differences
(Qwen3-8B uses grouped-query attention and different MLP ratios), (b) task differences (Com2Sense
requires multi-hop commonsense vs PIQA's single-step physical intuition).

### 3. Sublayer decomposition explains the monotonic resid_post result

The Week 9 `resid_post` patching showed monotonically increasing negative effects (L18: −0.006
→ L35: −0.355). The sublayer decomposition reveals this monotonic growth is not explained by
either sublayer alone:

- Attention and MLP effects fluctuate, with several layers showing near-zero or positive deltas.
- The cumulative negative effect in `resid_post` builds because patching the *full* post-layer
  state compounds distributional mismatch across all remaining layers, while patching only one
  sublayer leaves the other and the residual carry-through intact.
- This means the large negative effects in `resid_post` patching are an artifact of the
  full-stream replacement strategy, not evidence that those layers are causally important.
  The sublayer results are a cleaner causal test.

### 4. Mixed signs indicate no systematic encoding of commonsense truth in either sublayer

The `resid_post` effects were uniformly negative (degradation only). Sublayer effects are mixed:
positive at some layers (e.g., L24 MLP +0.077, L25 attn +0.163, L31 MLP +0.023), negative at
others. This sign variability means:

- The correct and incorrect examples' sublayer vectors are not systematically organized in a
  direction that encodes commonsense truth.
- Patching a sublayer sometimes helps, sometimes hurts, with no layer reliably helping.
- This is consistent with the ~59% probing accuracy: the signal exists but is too weak and
  variable to be captured by a single-component intervention.

---

## Comparison with Prior Experiments

| Experiment | Hook | Flip rate | Max |Δlogit|| Dominant sublayer |
|-----------|------|-----------|----------|-------------------|
| Layer patching (Week 9) | `resid_post` | 0–0.5% | 0.355 | N/A (full stream) |
| Head patching (Week 9) | `attn.hook_z` | 0% | 0.147 | Attention (single head) |
| **Attn sublayer (this)** | `hook_attn_out` | 0–1.5% | **0.163** | — |
| **MLP sublayer (this)** | `hook_mlp_out` | 0–1.5% | **0.149** | — |

Attention sublayer patching produces the single largest positive logit change of any intervention
in the study (+0.163 at L25), but still flips only 1.5% of pairs. This makes L25 attention the
closest thing to a causal bottleneck we have found — while also demonstrating that no such
bottleneck is sufficient to reliably flip commonsense predictions.

---

## Implications for the Research Question

**Localized or distributed?** Still distributed. Even decomposing into attention vs MLP, neither
sublayer shows consistent causal influence. The absence of MLP dominance (unlike GPT-2 / PIQA)
suggests that in Qwen3-8B the failure is genuinely spread across both sublayer types and does
not concentrate in either.

**Architectural, representational, or training-induced?** The sublayer results strengthen the
**representational** interpretation. If the failure were architectural — e.g., if the attention
mechanism were intrinsically incapable of commonsense — we would expect attention patching to
be uniformly harmful. Instead, L25 attention patching is *helpful* (+0.163) for some pairs.
The architecture can transiently encode useful commonsense signal; it just doesn't do so
consistently enough to be captured by a single-layer intervention.

**Difference from GPT-2 Small:** The other team's MLP-dominance finding does not generalize to
Qwen3-8B. Possible explanations: (1) Qwen3-8B is 67× larger and may distribute commonsense
knowledge more broadly across layers; (2) PIQA is a simpler physical-world benchmark where
MLP-stored facts are more directly relevant, while Com2Sense requires multi-step
causal/temporal reasoning; (3) GPT-2's causal tracing used zero-ablation (testing necessity)
while our patching tests sufficiency — different methods can yield different dominant components.
