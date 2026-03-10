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

- Layers 0–19: no strong linearly separable probe signal. Layers 20+: signal rises, peaks at 30–34
- Initial interpretation: **divergence looked late** → inference failure, not knowledge-retrieval failure
- After L2 regularization correction (TA feedback): genuine accuracy **~59%**
  - Original C=1.0 results (61%) were ~2% inflated by overfitting
  - PCA-50 acts as implicit regularization; optimal C=0.01
- Correctness probe ≈ ground truth probe (~59%) → model doesn't "know" the answer
  and fail to use it; it genuinely lacks a strong commonsense representation
- **Important update from logit lens:** incorrect examples often show an early wrong-answer
  bias, but that bias typically becomes stable only around layer 20

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

**Knowledge-retrieval or inference failure?** Best current answer: **early bias plus late
consolidation**. Probing suggested the cleanly separable signal emerged around layer 20,
but logit lens showed that many incorrect examples are already tilted toward the wrong
answer at the earliest measured layer. The final wrong answer usually becomes stable
around layer 20, so the failure is not purely late-onset.

**Architectural, representational, or training-induced?** Representational. The architecture
can encode commonsense (59% > 50%), but the representation is too weak, diffuse, and
unstable. Cannot fully rule out training-induced (would require comparing different
training data).

**Key contrast with prior work:** Factual recall tasks (Wang et al. 2022, Geva et al. 2023)
show localized circuits where patching flips outputs. Commonsense reasoning does not —
it is fundamentally more distributed and less amenable to surgical intervention.

---

## Remaining Gaps

- **Failure-direction logit lens:** Split failed-on-true vs failed-on-false to test whether
  the early bias is specifically a false-bias
- **Position-sensitive logit lens:** Compare statement-end vs final-token trajectories
- These are relatively cheap follow-ups for a final polishing pass

## Key Files

| File | What |
|------|------|
| `eval/` | Phase 1 behavioral results |
| `analysis/` | Python analysis code only |
| `artifacts/analysis/probe_results.json` | Phase 2 probing (3 probes × 36 layers) |
| `artifacts/analysis/l2_sweep_results.json` | L2 regularization sweep (7 C values) |
| `artifacts/analysis/patching_results.json` | Layer-level patching (18 layers × 200 pairs) |
| `artifacts/analysis/head_patching_results.json` | Head-level patching (160 combos × 200 pairs) |
| `artifacts/analysis/head_probe_results.json` | Head-level probing (576 heads) |
| `artifacts/analysis/ablation_results.json` | Mean-ablation (17 heads × 200 pairs) |
| `reports/summary.md` | This cross-phase project summary |
| `reports/week9_findings.md` | Detailed Week 9 analysis with full tables |
| `reports/probing_findings.md` | Detailed Phase 2 analysis |
| `reports/logit_lens_findings.md` | Logit-lens findings and updated conclusion |
| `reports/figures/` | Tracked presentation-quality figures |
| `results_download/logit_lens/` | Downloaded local copies of raw logit-lens outputs |
