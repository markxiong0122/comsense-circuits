# Logit Lens Findings

## What we ran

We ran a logit-lens analysis on the **200 selected asymmetric failure pairs** from Phase 1 / Phase 2. For each example, we took the cached final-token residual stream at every layer and projected it through the model's final normalization and unembedding to recover the layer-by-layer preference for **True** vs **False**.

To make the trajectories comparable across examples with different labels, we used a **signed logit gap**:

- **positive** = the model supports the ground-truth answer
- **negative** = the model supports the wrong answer

This gives a direct view of **what answer the model is moving toward at each layer**, complementing:
- probing: *is information linearly present?*
- patching: *can intervention change the answer?*
- logit lens: *what answer is the model implicitly predicting right now?*

---

## Main quantitative results

### Aggregate summary

- **Model:** `Qwen/Qwen3-8B`
- **Pairs processed:** `200`
- **Layers:** `36`

### Trajectory summary

- **First strong divergence layer:** `0`
- **Mean final signed gap, correct examples:** `+2.80875`
- **Mean final signed gap, incorrect examples:** `-2.531875`

### Failure timing summary

- **Mean incorrect first-wrong layer:** `1.61`
- **Mean incorrect stable-wrong layer:** `19.885`
- **Mean correct stable-correct layer:** `20.235`

---

## Failure timing breakdown

### First wrong layer for incorrect examples

- **Layer 0:** `153 / 200`
- **Layers 1–19:** `47 / 200`
- **Layers 20+:** `0 / 200`

### Stable wrong layer for incorrect examples

- **Layers 1–19:** `152 / 200`
- **Layers 20–29:** `46 / 200`
- **Layers 30+:** `2 / 200`

This means:

1. Most incorrect examples are already on the **wrong side very early**
2. But the model's prediction often becomes **stable** around the late-middle network, near layer 20

---

## Domain-level results

Final signed gap on incorrect examples by domain:

- **social:** `-3.5213`
- **temporal:** `-2.2950`
- **time:** `-2.1954`

Signed gap at layer 20 on incorrect examples:

- **social:** `-1.2380`
- **temporal:** `-0.7713`
- **time:** `-0.6884`

Among the selected failure pairs, **social failures** appear to end in the strongest wrong-state by the final layer.

---

## Layer trajectory snapshot

### Correct examples

- layer 0: `+0.9659`
- layer 1: `+0.7139`
- layer 5: `+0.4152`
- layer 10: `-0.5896`
- layer 15: `-0.5398`
- layer 20: `+0.9480`
- layer 25: `+3.4691`
- layer 30: `+11.2172`
- layer 35: `+2.8087`

### Incorrect examples

- layer 0: `-0.9670`
- layer 1: `-0.7155`
- layer 5: `-0.4159`
- layer 10: `+0.5945`
- layer 15: `+0.5491`
- layer 20: `-0.8383`
- layer 25: `-3.3062`
- layer 30: `-10.3256`
- layer 35: `-2.5319`

A key observation is that these trajectories are **not monotonic**. They do not simply move steadily toward the final answer. Instead, they:

1. start separated early,
2. partially reverse in mid layers,
3. then re-separate strongly around layer 20+.

---

## Interpretation

## 1. The failures are not purely late-onset

Before logit lens, the Phase 2 probing result suggested a relatively clean story:

- layers 0–19 had near-chance probe accuracy,
- signal rose around layer 20,
- therefore commonsense failure looked like a **late inference failure**.

Logit lens complicates that picture.

The new result shows that the model's internal preference is often already tilted toward the wrong answer at the **earliest measured layer**:

- `153 / 200` incorrect examples are wrong at layer 0
- the average correct and incorrect trajectories diverge immediately

So the failure cannot be described as **only** emerging late.

---

## 2. But late layers still matter: they consolidate the final answer

Although many incorrect examples are wrong early, they do **not** instantly become fixed. The average stable-wrong layer is about `19.9`, almost exactly where the probing signal began to rise.

That suggests a two-stage picture:

### Early stage
The model develops or inherits an **initial directional bias**:
- correct examples lean toward the correct answer
- incorrect examples lean toward the wrong answer

### Late stage
The network **stabilizes and amplifies** that tendency:
- incorrect examples settle into a durable wrong answer around layer 20
- correct examples settle into a durable correct answer at roughly the same depth

So the late layers still appear to be the point where the answer becomes committed and behaviorally decisive.

---

## 3. Probing and logit lens are measuring different things

This result also helps explain why the probing and patching findings looked the way they did.

### Probing
Linear probes asked whether correctness / truth was **cleanly linearly separable** at a layer.

### Logit lens
Logit lens asks what answer the model is **already moving toward** at a layer.

These are not the same question.

A layer can contain:
- an early weak directional bias in the output space,
- without containing a strong, clean, linearly separable correctness representation.

That may explain why:

- probing found weak signal early,
- but logit lens detects early directional separation.

So the logit-lens result does **not** invalidate the probe result; it refines it.

---

## 4. The model seems to repeatedly transform the judgment, not compute it once

The trajectory reversal is one of the most interesting findings.

Correct examples start positive, become negative in mid layers, then recover strongly.
Incorrect examples do the opposite pattern.

That suggests the model is not doing a single one-pass decision. Instead, the True/False preference appears to be:

- formed,
- transformed,
- partially overwritten,
- and then re-amplified later.

This is consistent with a **distributed, iterative computation**, rather than a single localized circuit that writes the answer once.

That also matches the patching result:

- no single layer was sufficient to flip the answer
- no single head was sufficient to flip the answer

The logit-lens trajectories support the same broad story: the answer is shaped across many stages, not in one narrow bottleneck.

---

## 5. Updated view of the failure mechanism

A better summary after logit lens is:

> Commonsense failures in Qwen3-8B are not purely late failures. They often begin with an early bias toward the wrong answer, but the final wrong prediction is usually consolidated around layer 20 and amplified in later layers.

This is more precise than saying either:
- “the model is wrong from the start,” or
- “the model only fails late.”

It appears to be **both**:
- an **early bias**
- followed by **late consolidation**

---

## Updated conclusion

## Previous conclusion
The earlier project conclusion was roughly:

> commonsense failure is a distributed late inference failure rather than an early retrieval failure.

## Updated conclusion after logit lens
The revised conclusion should be:

> Commonsense failures in Qwen3-8B reflect a **distributed failure process with both early bias and late consolidation**. Incorrect examples are often already biased toward the wrong answer at the earliest measured layer, but the model typically stabilizes that wrong answer around layer 20 and amplifies it thereafter. This means the failure is not cleanly localizable to a single layer or head, and not well described as a purely late-onset inference error. Instead, commonsense failure appears to emerge from an interaction between early representational bias and later reasoning-stage commitment.

This remains consistent with the broader mechanistic finding of the project:

- **not localized**
- **distributed across the network**
- **not fixable by single-layer or single-head patching**

But it updates the temporal story from:

- “late-only failure”

to:

- “early bias + late commitment.”

---

## What this changes for next steps

These results suggest several useful follow-ups:

1. **Logit-lens by failure direction**
   - split `failed_on_true` vs `failed_on_false`
   - test whether the early bias is specifically a false-bias

2. **Domain-specific trajectory analysis**
   - especially compare `social`, `temporal`, and `time`
   - determine whether different domains fail with different timing signatures

3. **Ablation rather than patching**
   - if the model is carrying an early wrong bias, ablation may be more informative than transplant-style patching

4. **Position-sensitive logit lens**
   - compare statement-end vs final-token trajectories
   - test whether the prompt suffix sharpens or reverses the bias

---

## Bottom line

The logit-lens analysis adds an important refinement to the project:

- **incorrect examples are usually wrong early**
- **correct and incorrect trajectories diverge immediately**
- **but the final answer becomes stable around layer 20**
- **the computation is dynamic and distributed, not localized**

That makes the strongest current summary:

> Qwen3-8B's commonsense failures are best understood as **distributed representational failures with early wrong-answer bias and late-stage consolidation**, rather than as a single localized circuit failure or a purely late inference-only error.