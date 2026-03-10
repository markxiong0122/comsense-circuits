# Week 9 Findings: Activation Patching, Head-Level Analysis, and L2 Regularization

## Research Question

From the project proposal: *Are commonsense reasoning failures localized to specific layers
or attention heads, or do they reflect distributed representational deficits? Are such failures
architectural, representational, or training-induced?*

The proposal outlines three phases to answer this:
1. **Behavioral analysis** — characterize failures by domain/scenario (Phase 1, Weeks 6–7)
2. **Representation probing** — identify at which layer the model first encodes the correct
   answer, and whether that layer differs between correct and incorrect predictions;
   determine if divergence is early (knowledge-retrieval failure) or late (inference failure)
   (Phase 2, Weeks 7–8)
3. **Mechanistic intervention** — patch activations from correct into incorrect forward passes
   to identify the minimal set of components sufficient to flip predictions; ablation studies
   to confirm causal roles (Phase 3, Week 9)

This document covers Week 9 (Phase 3) plus the L2 regularization sweep addressing TA
feedback on Phase 2.

---

## Context from Phase 2

The probing results (documented in `probing_findings.md`) established:
- Signal is at chance through layers 0–19, rises from layer 20, peaks at 30–32
- **Divergence is late** → this is an inference failure, not a knowledge-retrieval failure
  (the model doesn't misrepresent the input; it fails at the reasoning step)
- Peak accuracy was 61% (correctness + PCA-50) — but the TA flagged overfitting risk
  with 4096-dim features and ~400 examples
- The correctness probe slightly outperformed the ground truth probe (61% vs 60%),
  suggesting the model is already committed to its (wrong) prediction at the
  representation level

Three open questions remained:
1. Was the 61% genuine, or inflated by overfitting?
2. Can activation patching identify a minimal set of components that flip predictions?
   (testing localized vs distributed)
3. Is the signal concentrated in specific heads or spread across the residual stream?

---

## Experiment 1: L2 Regularization Sweep

**Question:** How much of the 61% probe accuracy is genuine signal vs. overfitting?

**Method:** Swept C values [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0] across the same three
probe configurations from Phase 2. Lower C = stronger L2 penalty = less overfitting risk.

| Probe | Best C | Best Layer | Accuracy |
|-------|--------|-----------|----------|
| Correctness (no PCA) | 0.001 | 34 | 57.50% |
| Correctness + PCA-50 | 0.01 | 23 | 58.75% |
| Ground truth + PCA-50 | 0.01 | 32 | 59.25% |

Comparison with original Phase 2 results (C=1.0):

| Probe | Phase 2 (C=1.0) | Week 9 (optimal C) | Difference |
|-------|---------------|-------------------|------------|
| Correctness (no PCA) | 54.75% (L32) | 57.50% (L34, C=0.001) | +2.75% |
| Correctness + PCA-50 | 61.00% (L30) | 58.75% (L23, C=0.01) | -2.25% |
| Ground truth + PCA-50 | 60.00% (L31) | 59.25% (L32, C=0.01) | -0.75% |

**Findings:**

- The **raw (no PCA) probe improves** with stronger regularization (54.75% → 57.50%),
  confirming the TA's concern: in the full 4096-dim space, C=1.0 was fitting noise.
- The **PCA-50 probes drop ~1–2%** (61% → 58.75%, 60% → 59.25%). PCA already acts as
  implicit regularization, so the original results were less inflated. The ~2% drop
  represents genuine overfitting that PCA didn't fully prevent.
- **PCA-50 accuracy plateaus across C values** (58.75% for C=0.01 through C=10.0),
  confirming dimensionality reduction is the dominant regularizer.
- **The genuine probing signal is ~59%.** This is the corrected baseline.
- Relevance to the research question: a ~59% probe accuracy is weak. The residual stream
  carries only modest linear signal about commonsense correctness, suggesting the
  representation does not cleanly separate correct from incorrect reasoning.

*(See `l2_sweep_results.png` for the accuracy-vs-C curves.)*

---

## Experiment 2: Activation Patching (Layers 18–35)

**Question from proposal:** *Can we identify a minimal set of components whose activations
are necessary and sufficient to flip the incorrect prediction to the correct one?*

This experiment tests the **localized vs distributed** hypothesis at the layer level. If
commonsense failures are localized, patching at the right layer should flip predictions.

**Method:** For each of the 200 pairs, cached the correct example's residual stream via
`run_with_cache()`, then ran the incorrect example with a hook replacing `resid_post` at the
final token position with the correct vector, one layer at a time (layers 18–35).

| Layer | Flip Rate | Mean Logit Change |
|-------|-----------|-------------------|
| 18 | 0.0% | -0.006 |
| 19 | 0.0% | -0.039 |
| 20 | 0.0% | -0.104 |
| 21 | 0.0% | -0.146 |
| 22 | 0.0% | -0.269 |
| 23 | 0.0% | -0.311 |
| 24 | 0.0% | -0.301 |
| 25 | 0.0% | -0.313 |
| 26 | 0.5% | -0.315 |
| 27 | 0.0% | -0.325 |
| 28 | 0.0% | -0.319 |
| 29 | 0.0% | -0.335 |
| 30 | 0.0% | -0.339 |
| 31 | 0.0% | -0.343 |
| 32 | 0.0% | -0.344 |
| 33 | 0.0% | -0.343 |
| 34 | 0.0% | -0.343 |
| 35 | 0.0% | -0.355 |

Per-domain flip rates were 0.0% across all three domains (time, social, temporal) at every
layer, except for a single flip at layer 26.

**Findings:**

- **Near-zero flip rate across all 18 layers.** No single layer's activation, when
  transplanted from the correct forward pass, is sufficient to flip the prediction.
- **All logit changes are negative.** Patching makes predictions *worse*, not better.
  The effect grows monotonically from -0.006 (L18) to -0.355 (L35).
- **Why negative?** The correct and incorrect examples are different sentences with
  different content. Their residual streams encode different token sequences. Substituting
  one into the other creates a distributional mismatch — downstream layers receive
  activations inconsistent with the tokens they processed, causing degradation.
- **Implication for the research question:** This is strong evidence against the
  **localized** hypothesis. If commonsense reasoning were bottlenecked through a specific
  layer, patching that layer would flip predictions (as it does for factual recall in
  Geva et al. 2023). The failure to flip at *any* layer suggests the failure is
  **distributed** across the network.

*(See `patching_results.png` for the flip rate and logit change curves.)*

---

## Experiment 3: Head-Level Patching (Top 5 Layers)

**Question from proposal:** *Identify the minimal set of components whose activations are
necessary and sufficient to flip the incorrect prediction.*

This tests the localized hypothesis at finer granularity: even if no single layer suffices,
perhaps a specific attention head is the bottleneck.

**Method:** Within the top 5 layers by activation patching effect (L18, L19, L20, L21, L26),
patched individual attention head z outputs (the head's output before projection) from
correct into incorrect examples, one head at a time. 5 layers x 32 heads = 160 interventions
per pair.

| Head | |Mean Effect| | Mean Effect | Flip Rate |
|------|---------------|-------------|-----------|
| L20.H8 | 0.147 | -0.068 | 0.0% |
| L21.H19 | 0.096 | -0.039 | 0.0% |
| L26.H26 | 0.079 | -0.031 | 0.0% |
| L26.H25 | 0.078 | -0.038 | 0.0% |
| L19.H31 | 0.061 | -0.029 | 0.0% |
| L18.H16 | 0.056 | -0.012 | 0.0% |
| L21.H15 | 0.055 | -0.016 | 0.0% |
| L21.H16 | 0.055 | -0.021 | 0.0% |
| L18.H6 | 0.054 | -0.004 | 0.0% |
| L20.H9 | 0.054 | -0.006 | 0.0% |

**Findings:**

- **Zero flips from any head.** The most influential head (L20.H8) moves the logit by
  0.147 units — negligible relative to the average logit gap of 2.534 in these pairs.
- **All mean effects are negative**, consistent with layer-level patching.
- **No minimal set of components exists** (at least at single-head granularity) that is
  sufficient to flip commonsense predictions. This further supports the **distributed**
  hypothesis.

*(See `head_patching_results.png` for the top 15 heads by absolute effect.)*

---

## Experiment 4: Head-Level Probing

**Question:** Is the commonsense signal concentrated in specific heads, or is it spread
across the residual stream? (Testing localized vs distributed from the probing side.)

**Method:** For each of 576 heads (18 layers x 32 heads, layers 18–35), trained logistic
regression (C=0.1) on the per-head z vector ([d_head=128]-dimensional — much less
overfitting risk than d_model=4096). Probed at both the final token and statement-end
positions.

| Probe | Best Head | Accuracy |
|-------|-----------|----------|
| Correctness, final token | L32.H20 | 59.50% |
| Correctness, stmt-end | L28.H11 | 58.25% |
| Ground truth, final token | L34.H18 | 59.50% |
| Ground truth, stmt-end | L35.H23 | 59.25% |

Top 3 heads per configuration:

- **Correctness, final:** L32.H20 (59.5%), L24.H11 (59.0%), L28.H25 (59.0%)
- **Ground truth, final:** L34.H18 (59.5%), L32.H12 (59.25%), L35.H23 (59.0%)

**Findings:**

- **Individual heads match the full residual stream** (~59.5% vs ~59% after L2 correction).
  A single 128-dim head vector is as informative as the full 4096-dim stream. This means
  the signal is weakly but redundantly present across multiple heads.
- **No standout head.** Top heads cluster tightly at 59–59.5%. Contrast this with factual
  knowledge probing (e.g., Geva et al. 2023) where specific heads often dominate.
- **Probing and patching identify different heads.** The most informative heads (L32, L34)
  differ from the most causally influential ones (L20, L21). Information presence and
  causal influence are dissociated — a head can encode weak commonsense signal without
  being a causal bottleneck.
- **Implication:** The signal is **distributed**, not concentrated. Many heads carry
  roughly the same weak signal, and none is a privileged locus of commonsense computation.

*(See `head_probe_heatmap.png` for the full layer x head accuracy heatmap.)*

---

## Experiment 5: Intermediate Token Position Probing

**Question from proposal:** *Identify at which layer the model's internal representation
first encodes the correct answer.* The Phase 2 findings noted that the final token might not
be the richest probe site. Does the statement-end position carry stronger signal?

**Method:** Extracted residual stream at the last token of the actual statement (before the
prompt suffix) and probed with PCA-50.

| Position | Best Layer | Accuracy |
|----------|-----------|----------|
| Final token | 31 | 60.00% |
| Statement-end | 28 | 57.25% |

**Findings:**

- **Final token is stronger by ~3%.** The model's commonsense judgment continues to
  develop after the statement ends, through the prompt suffix tokens.
- **Statement-end peaks earlier** (layer 28 vs 31), consistent with the statement content
  being processed first, then the judgment crystallizing during the task instruction.
- This rules out one Phase 2 hypothesis: the weak signal was not caused by probing at the
  wrong position. Final token is the strongest site, and the signal is still only ~60%.

---

## Answering the Research Question

### Localized or distributed?

**Distributed.** Every experiment points the same way:

- Activation patching at all 18 layers: 0% flip rate (no layer is sufficient)
- Head-level patching across 160 layer-head combinations: 0% flip rate (no head is sufficient)
- Head-level probing: individual heads match full-stream accuracy (~59.5%), meaning the
  signal is redundantly spread, not concentrated
- No "minimal set of components" could be identified whose patching flips predictions

This contrasts sharply with factual recall tasks (e.g., Wang et al. 2022 on IOI; Geva et al.
2023 on factual associations), where specific layers and heads can be identified as causal
bottlenecks and patching reliably flips outputs.

### Knowledge-retrieval failure or inference failure?

**Inference failure.** The evidence:

- Layers 0–19 carry zero signal → the model correctly encodes the input (no early divergence)
- Divergence begins at layer 20 → the failure occurs during the *reasoning computation*,
  not during input processing or factual recall
- The model's false-bias (73.6% of failures are on true statements, from Phase 1) suggests
  a systematic inference default: when uncertain about causal/temporal relationships, the
  model defaults to rejection ("false")

### Architectural, representational, or training-induced?

Our evidence most strongly supports **representational**:

- The ~59% probing accuracy means the residual stream carries only weak signal about
  commonsense truth. The model never builds a strong enough internal representation to
  reliably distinguish correct from incorrect commonsense reasoning.
- The correctness probe slightly outperforms the ground truth probe (after L2 correction:
  58.75% vs 59.25% — now nearly identical). The model's representation of *what it will
  predict* is roughly as strong as its representation of *what the correct answer is*.
  This means the model doesn't "know" the answer and fail to use it — it genuinely lacks
  a strong representation of commonsense truth.
- We cannot fully distinguish representational from training-induced with our methods.
  If the model never encountered enough commonsense reasoning examples during training,
  the representational deficit would be a consequence of training data limitations.
  Distinguishing these would require comparing models trained on different data mixtures,
  which is beyond our scope.
- We can partially rule out **architectural**: the signal does exist (59% > 50%), just
  weakly. The architecture *can* encode commonsense distinctions; it just doesn't do so
  strongly. A purely architectural limitation would more likely show 50% (no signal at all)
  or show signal only in specific components.

---

## Gap: Ablation Studies

The proposal called for "selectively zero out or mean-ablate identified components to confirm
their causal role without introducing information from the correct forward pass." We performed
patching (which introduces correct-pass information) but not ablation (which tests necessity
by removing information). Given that patching produced 0% flips — meaning no component is
*sufficient* — ablation would test whether any component is *necessary*. This is a remaining
gap that could be addressed in Week 10 if time permits, using mean-ablation on the top
patching heads (L20.H8, L21.H19, L26.H26) to test if removing them degrades correct
predictions.

---

## Summary of All Results

| Experiment | Key Result | Implication |
|-----------|------------|-------------|
| L2 sweep | Genuine probe accuracy ~59% | Signal is real but weak |
| Layer patching (18 layers) | 0% flip rate, negative logit changes | No layer is sufficient → distributed |
| Head patching (160 combos) | 0% flip rate, max effect 0.147 | No head is sufficient → distributed |
| Head probing (576 heads) | Best head 59.5% ≈ full stream 59% | Signal is redundant, not concentrated |
| Position probing | Final token (60%) > stmt-end (57.25%) | Judgment forms late, at final token |
