# Project Summary

**Research question:** Are commonsense reasoning failures in LLMs localized to specific
layers or attention heads, or do they reflect distributed representational deficits?
Are such failures architectural, representational, or training-induced?

**Model:** Qwen3-8B (36 layers, 32 heads, d_model=4096) via TransformerLens
**Dataset:** Com2Sense — 2,390 True/False statements, complementary pairs

---

## Phase 1: Behavioral Analysis (Weeks 6–7)

- Overall accuracy: **70.25%**
- 1,195 complementary pairs identified; **611 asymmetric** (one correct, one wrong)
- **False-bias:** 73.6% of failures are on true statements — the model defaults to
  "false" when uncertain, especially in causal scenarios (87% in social-causal)
- Hardest domains: temporal (63.5%), time (66.6%) — not physical-causal as hypothesized
- Selected **200 high-confidence failure pairs** (avg wrong-answer confidence 0.896)

## Phase 2: Representation Probing (Weeks 7–8)

- Layers 0–19: no signal (chance). Layers 20+: signal rises, peaks at 30–34
- **Divergence is late** → inference failure, not knowledge-retrieval failure
- After L2 regularization correction (TA feedback): genuine accuracy **~59%**
  - Original C=1.0 results (61%) were ~2% inflated by overfitting
  - PCA-50 acts as implicit regularization; optimal C=0.01
- Correctness probe ≈ ground truth probe (~59%) → model doesn't "know" the answer
  and fail to use it; it genuinely lacks a strong commonsense representation

## Phase 3: Mechanistic Intervention (Week 9)

**Activation patching (layers 18–35):**
- **0% flip rate** across all 18 layers
- Transplanting correct activations makes predictions *worse* (all logit changes negative)
- No single layer is sufficient to fix commonsense failures

**Head-level patching (5 layers × 32 heads):**
- **0% flip rate** across 160 layer-head combinations
- Most influential head: L20.H8 (effect 0.147 — tiny vs avg logit gap of 2.534)
- No single attention head is sufficient

**Head-level probing (576 heads, 128-dim each):**
- Best individual head: **59.5%** (L32.H20) — matches full residual stream (~59%)
- Signal is redundantly spread across many heads, not concentrated
- Probing top heads (L32, L34) ≠ patching top heads (L20, L21) — information and
  causation are dissociated

**Intermediate position probing:**
- Final token: **60%** (L31) > statement-end: **57.25%** (L28)
- Commonsense judgment keeps developing through the prompt suffix

**Mean-ablation (17 target heads, Week 10):**
- Tests NECESSITY: does removing a head break correct predictions?
- Max flip rate: **1.0%** (L26.H25 — 2 flips out of 200)
- Max |logit change|: **0.028** (L18.H6) — negligible
- No head is necessary for correct predictions, confirming distributed encoding
- Patching top heads and probing top heads both show near-zero ablation effects

---

## Answer to the Research Question

**Localized or distributed?** Distributed. Patching tests sufficiency (0% flips),
ablation tests necessity (max 1% flips). Neither finds a critical component.
Signal is weak (~59%) and redundantly spread across the network.

**Knowledge-retrieval or inference failure?** Inference. Early layers carry no signal;
divergence begins at layer 20 during reasoning computation, not input processing.

**Architectural, representational, or training-induced?** Representational. The architecture
can encode commonsense (59% > 50%), but the representation is too weak and diffuse.
Cannot fully rule out training-induced (would require comparing different training data).

**Key contrast with prior work:** Factual recall tasks (Wang et al. 2022, Geva et al. 2023)
show localized circuits where patching flips outputs. Commonsense reasoning does not —
it is fundamentally more distributed and less amenable to surgical intervention.

---

## Remaining Gaps

- **Logit lens:** Project residual stream through unembedding at each layer (proposal Phase 2)
  — teammate is running this separately

## Key Files

| File | What |
|------|------|
| `eval/` | Phase 1 behavioral results |
| `analysis/probe_results.json` | Phase 2 probing (3 probes × 36 layers) |
| `analysis/l2_sweep_results.json` | L2 regularization sweep (7 C values) |
| `analysis/patching_results.json` | Layer-level patching (18 layers × 200 pairs) |
| `analysis/head_patching_results.json` | Head-level patching (160 combos × 200 pairs) |
| `analysis/head_probe_results.json` | Head-level probing (576 heads) |
| `analysis/ablation_results.json` | Mean-ablation (17 heads × 200 pairs) |
| `analysis/week9_findings.md` | Detailed Week 9 analysis with full tables |
| `analysis/probing_findings.md` | Detailed Phase 2 analysis |
