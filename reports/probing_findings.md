# Phase 2 Probing Findings

## Setup

200 asymmetric complementary pairs were selected from Phase 1 (611 total), prioritizing
the weakest-performing domains (temporal: 63.5%, time: 66.6%) and highest-confidence
wrong answers (avg confidence on incorrect prediction: 0.896). For each pair, both the
correct and incorrect sentences were run through Qwen3-8B via `run_with_cache()`, capturing
the residual stream at the final token position (`hook_resid_post`) across all 36 layers.
This produced 400 tensors of shape `[36, 4096]`.

A linear probe (logistic regression) was trained at each layer to predict a binary label
from the `[4096]` residual stream vector. Cross-validation used `GroupKFold(n_splits=5)`
with pairs as groups, so both members of a complementary pair always land in the same
fold — preventing leakage from near-duplicate sentences appearing in both train and test.

---

## Three Probes

### Probe 1 — Correctness, no PCA
**Label:** `y=1` if the model answered correctly, `y=0` if incorrectly.
**Features:** raw `[4096]` residual stream vector, standardized.

This probe asks: *can the residual stream linearly predict whether the model will be right
or wrong?*

### Probe 2 — Correctness, PCA-50
Same label as Probe 1, but features are first reduced to 50 principal components (fit on
training folds only).

**Why add PCA:** With ~320 training examples in a 4096-dimensional space the problem is
severely underdetermined. A linear classifier can overfit to noise rather than finding a
meaningful direction. Compressing to 50 components keeps the most variance-explaining
directions and strips out the noise, giving the logistic regression a tractable problem.

### Probe 3 — Ground truth, PCA-50
**Label:** `y=1` if the statement is True, `y=0` if False (the actual ground truth label,
regardless of what the model predicted).
**Features:** PCA-50, same as Probe 2.

**Why swap in ground truth:** The correctness label conflates two things — whether the model
*has* the right answer encoded and whether it *uses* it. Ground truth probing asks a purer
question: *does the residual stream encode the correct answer at all, independent of the
model's output?* If ground truth accuracy is high but correctness accuracy is low, the
representation knows the answer but something downstream fails to act on it. If both are
similarly low, the representation genuinely doesn't encode the correct answer.

---

## Results

| Probe | Best layer | Best accuracy |
|-------|-----------|---------------|
| Correctness, no PCA | 32 | 54.75% |
| Correctness + PCA-50 | 30 | 61.00% |
| Ground truth + PCA-50 | 31 | 60.00% |

All probes are at chance (≈50%) through **layers 0–19**, then rise consistently from
**layer 20 onward**, peaking in the **23–32 range**.

### Per-domain accuracy at best layer (layer 31, ground truth + PCA-50)

| Domain | Accuracy |
|--------|----------|
| time | 60.2% |
| social | 59.6% |
| temporal | 57.0% |

---

## Interpretation

**1. The decision zone is the second half of the network.**
Layers 0–19 carry no linearly separable signal for either probe. Both probes rise sharply
starting at layer 20. This means the features that distinguish correct from incorrect
reasoning — and True from False statements — only crystallize in the back half of the
model. The early layers are doing token-level and syntactic processing; the commonsense
judgment forms late.

**2. PCA reveals signal that raw probing misses.**
The jump from 54.75% (no PCA) to 61.00% (PCA-50) on the same correctness label shows that
the signal is real but sparse in 4096 dimensions. The top 50 principal components carry
enough variance to expose it; the remaining ~4046 dimensions are noise relative to this
task with 400 examples.

**3. Correctness is slightly more separable than ground truth (61% vs 60%).**
The residual stream more cleanly encodes *what the model will predict* than *what the
correct answer is*. This is a subtle but meaningful distinction: at the representation
level, the model is already somewhat committed to its (sometimes wrong) prediction. The
correct answer is less cleanly encoded than the model's own impending output. This rules
out a simple "the model knows the answer but randomly outputs the wrong token" story —
the representational commitment to the wrong answer is already present in the residual
stream before the final projection.

**4. Temporal is the hardest domain to probe.**
Domain probe accuracy at layer 31 mirrors Phase 1 accuracy rankings — time (60.2%) >
social (59.6%) > temporal (57.0%). Temporal reasoning failures are not only more frequent
but also less structured in the residual stream: a linear classifier has the hardest time
separating correct from incorrect representations here, suggesting the failure mode is
more distributed or less consistent across examples.

**5. The signal is weak overall (peak ≈ 61%).**
Even at the best layer with PCA, accuracy is well below what strong probes on clear
factual tasks typically achieve (often 80–95%). Three possible explanations:
- The distinction between correct and incorrect commonsense reasoning is genuinely
  diffuse in the residual stream — no single layer cleanly encodes it.
- The final token position may not be the richest probe site; intermediate token
  positions (over the statement itself) could carry stronger signal.
- 200 pairs may be insufficient to train a reliable probe even with PCA.

---

## Implications for Phase 3 (Activation Patching)

The probe results directly constrain where to focus patching experiments:

- **Target layers 20–32**, where the probes first diverge from chance. Patching outside
  this range is unlikely to causally affect the output.
- **Layer 23 is the earliest consistent rise point** — a natural candidate for the
  earliest intervention layer.
- **The weak overall separability** means activation patching (which swaps the entire
  residual stream vector, not just a linear projection of it) may succeed where the probe
  struggles — patching is a stronger intervention than probing.
- The **temporal domain** being hardest to probe suggests its failure circuits may be more
  distributed across layers and heads, potentially requiring head-level rather than
  layer-level patching.
