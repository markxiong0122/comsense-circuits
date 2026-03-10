# PR: Refactor File Structure + Complete Phase 1–4 Analysis

## What Changed

- Moved all generated outputs (`.json`, `.png`, findings `.md`) out of `analysis/` into their proper homes under `reports/` and `artifacts/analysis/`
- `analysis/` now contains **Python code only**
- `analysis/probe_results.json` moved to `artifacts/analysis/probe_results.json` (last straggler)
- Updated `README.md` to document the new directory contract
- `reports/` and `reports/figures/` are now the canonical home for tracked writeups and presentation-quality figures

---

## Repository Map for Report Generation

### Source Data & Behavioral Results → `eval/`

| File | Contents |
|------|----------|
| `eval/predictions.jsonl` | Per-example predictions: sentence, label, predicted_true, correct, true_logit, false_logit, confidence, domain, scenario, pair_id |
| `eval/accuracy_table.json` | Overall accuracy (70.25%), breakdowns by domain and scenario |
| `eval/asymmetric_pairs.json` | 611 pairs where the model gets exactly one of the two complementary statements wrong — the Phase 2–4 target set |
| `eval/pair_summary.json` | All 1,195 pair classifications (both correct: 534, asymmetric: 611, both wrong: 50) |
| `eval/summary.txt` | Human-readable Phase 1 overview |

### Raw Analysis Artifacts → `artifacts/analysis/`

| File | Contents |
|------|----------|
| `artifacts/analysis/probe_results.json` | Probing accuracy per layer, for all three probe configurations (correctness no-PCA, correctness PCA-50, ground-truth PCA-50) across 36 layers |
| `artifacts/analysis/l2_sweep_results.json` | Probing accuracy vs. regularization strength C ∈ {0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0} for all three probe configs |
| `artifacts/analysis/patching_results.json` | Per-layer activation patching results: flip rate and mean logit change for layers 18–35, over 200 pairs |
| `artifacts/analysis/head_probe_results.json` | Per-head probing accuracy for all 576 heads (18 layers × 32 heads), at both final-token and statement-end positions |
| `artifacts/analysis/head_patching_results.json` | Per-head patching effect (absolute and signed logit change, flip rate) for 160 layer-head combinations (5 layers × 32 heads) |

### Findings Writeups → `reports/`

| File | Phase | Key Claim |
|------|-------|-----------|
| `reports/summary.md` | Cross-phase | Master summary: research question, per-phase conclusions, final answer on localized vs. distributed, knowledge vs. inference failure, architectural vs. representational |
| `reports/probing_findings.md` | Phase 2 | Detailed probing methodology, three-probe design rationale, per-domain accuracy, interpretation of weak ~61% signal, implications for patching |
| `reports/week9_findings.md` | Phase 3 | Full tables for all five experiments: L2 sweep, layer patching, head patching, head probing, position probing; complete answer to the research question |
| `reports/failure_direction_summary.md` | Phase 1 | False-bias statistics: 73.6% of failures are on TRUE statements; breakdown by domain and domain×scenario; social-causal is worst (87% fail on TRUE) |
| `reports/logit_lens_findings.md` | Phase 4 | Layer-by-layer True/False trajectory analysis; updates the earlier "late inference failure" conclusion to "early bias + late consolidation" |

### Figures → `reports/figures/`

| File | What it shows | Supports finding in |
|------|--------------|---------------------|
| `probe_accuracy_curve.png` | Probing accuracy vs. layer for all three probe configs; near-chance through layer 19, rising from layer 20, peak at 30–32 | `reports/probing_findings.md` §Results, `reports/week9_findings.md` §L2 sweep |
| `l2_sweep_results.png` | Probing accuracy vs. regularization C for each probe config; shows PCA-50 is stable (implicit regularization), raw probe improves with stronger L2 | `reports/week9_findings.md` §Experiment 1 |
| `patching_results.png` | Flip rate and mean logit change per layer (18–35); flat 0% flip rate, monotonically worsening logit change | `reports/week9_findings.md` §Experiment 2 |
| `head_patching_results.png` | Top 15 heads by absolute patching effect; bar chart showing L20.H8 is best (0.147) but negligible vs. logit gap of 2.534 | `reports/week9_findings.md` §Experiment 3 |
| `head_probe_heatmap.png` | Full layer × head accuracy heatmap for head-level probing; shows diffuse signal with no dominant head | `reports/week9_findings.md` §Experiment 4 |
| `logit_lens_trajectories.png` | Mean signed logit gap per layer for correct vs. incorrect examples; shows early divergence, mid-layer reversal, re-amplification after layer 20 | `reports/logit_lens_findings.md` §Layer trajectory snapshot |
| `logit_lens_domain_trajectories.png` | Same trajectories split by domain (social, temporal, time); social failures show strongest final wrong-answer signal (−3.52) | `reports/logit_lens_findings.md` §Domain-level results |
| `logit_lens_incorrect_heatmap.png` | Per-example heatmap of signed logit gap across all 200 incorrect examples and all 36 layers; shows which examples flip early vs. late | `reports/logit_lens_findings.md` §Failure timing breakdown |
| `logit_lens_failure_layers.png` | Distribution of first-wrong layer and stable-wrong layer for incorrect examples; 153/200 are wrong at layer 0, stable by ~layer 20 | `reports/logit_lens_findings.md` §Failure timing breakdown |

---

## Narrative Arc for Report Generation

A research report should follow this arc:

### 1. Research Question (`reports/summary.md` §Research Question)
Are commonsense reasoning failures in Qwen3-8B localized to specific layers/heads, or distributed? Are they architectural, representational, or training-induced?

### 2. Phase 1 — Behavioral Characterization
- **Key numbers:** 70.25% overall accuracy, 611/1,195 pairs asymmetric
- **Key finding:** Strong false-bias — 73.6% of failures are on TRUE statements, peaking at 87% in social-causal scenarios
- **Source:** `reports/failure_direction_summary.md`, `eval/accuracy_table.json`
- **No figure needed** (tables suffice, or generate a bar chart from `failure_direction_summary.md`)

### 3. Phase 2 — Representation Probing
- **Key finding:** No linear signal in layers 0–19; signal rises from layer 20, peaks at 30–32 at ~61% (corrected to ~59% after L2 sweep). Correctness ≈ ground-truth probe accuracy → model doesn't "know but fail to use" the answer; it genuinely lacks a strong commonsense representation.
- **Source:** `reports/probing_findings.md`
- **Primary figure:** `probe_accuracy_curve.png`
- **Supporting figure:** `l2_sweep_results.png` (showing the 61% → 59% correction)

### 4. Phase 3 — Mechanistic Intervention
- **Key finding:** Zero flips from layer patching (18 layers) and head patching (160 combinations). Best single head (L20.H8) moves logit by 0.147 vs. average gap of 2.534. Head probing matches full-stream accuracy (59.5%) — signal is redundantly distributed. Probing and patching identify *different* heads, dissociating information presence from causal influence.
- **Source:** `reports/week9_findings.md`
- **Figures:** `patching_results.png`, `head_patching_results.png`, `head_probe_heatmap.png`

### 5. Phase 4 — Logit Lens
- **Key finding:** Updates Phase 2's "late inference failure" conclusion. 153/200 incorrect examples are already biased toward the wrong answer at layer 0. But the wrong answer only *stabilizes* around layer 20 (mean stable-wrong layer: 19.9). Trajectories are non-monotonic: initial bias → mid-layer reversal → re-amplification. This two-stage picture ("early bias + late consolidation") is consistent with the patching failure — no single stage is sufficient.
- **Source:** `reports/logit_lens_findings.md`
- **Figures:** `logit_lens_trajectories.png`, `logit_lens_domain_trajectories.png`, `logit_lens_failure_layers.png`, `logit_lens_incorrect_heatmap.png`

### 6. Synthesis (`reports/summary.md` §Answer to the Research Question)
- **Localized or distributed?** Distributed. Zero patching flips at any layer or head; signal is redundant across heads.
- **Knowledge-retrieval or inference failure?** Both: early bias (present at layer 0) + late consolidation (stabilizes at layer 20). Not a clean "retrieve then fail to use" story.
- **Architectural, representational, or training-induced?** Representational. Architecture can encode commonsense (59% > 50%), but too weakly and diffusely. Cannot rule out training-induced without cross-model comparison.
- **Contrast with prior work:** Factual recall (Wang et al. 2022 IOI; Geva et al. 2023) shows localized circuits where patching flips outputs. Commonsense reasoning does not.

---

## Analysis Code Reference

For any agent that needs to re-run or extend the analysis:

| Script | Purpose |
|--------|---------|
| `analysis/pair_analysis.py` | Computes failure direction statistics; reads from `eval/asymmetric_pairs.json` |
| `analysis/probe.py` | Runs layer-wise probing (3 configurations); reads activations from `activations_dir/`; writes `artifacts/analysis/probe_results.json` |
| `analysis/probe_l2_sweep.py` | Sweeps regularization C; writes `artifacts/analysis/l2_sweep_results.json` |
| `analysis/probe_heads.py` | Head-level probing (576 heads, 2 positions); reads head activations; writes `artifacts/analysis/head_probe_results.json` |
| `analysis/plot_logit_lens.py` | Generates all four logit-lens figures; reads from `results_download/logit_lens/`; writes to `reports/figures/` |