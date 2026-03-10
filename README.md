# ComSense Circuits

A mechanistic interpretability study of commonsense reasoning failures in Large Language Models (LLMs). This project evaluates **Qwen3-8B** on the **Com2Sense** dataset and follows the failures through multiple phases: behavioral evaluation, residual-stream probing, activation patching, head-level analysis, and logit-lens analysis.

## Overview

### Goal
Build infrastructure to:
1. Load Qwen3-8B via TransformerLens on Modal (A100-80GB GPU)
2. Evaluate performance on Com2Sense True/False commonsense statements
3. Identify complementary pairs where the model gets one right and one wrong
4. Probe residual-stream representations of correct vs. incorrect reasoning
5. Test causal interventions with activation and head-level patching
6. Analyze layer-by-layer answer trajectories with logit lens

### Key Design Decisions
- **Model:** Qwen3-8B (instruct version, NOT base)
- **Inference:** TransformerLens (`HookedTransformer.from_pretrained`)
- **GPU:** Modal A100-80GB
- **Dataset:** Com2Sense (tasksource/com2sense) — 2,390 True/False statements in complementary pairs
- **Evaluation:** Compare logits for " True" vs " False" tokens at final position (no generation)
- **Qwen3 Mode:** Non-thinking mode (`/no_think`) for clean residual streams
- **dtype:** bfloat16

## Project Structure

```
comsense-circuits/
├── config.py                  # Central configuration (model, dataset, prompts)
├── data_utils.py              # Dataset loading and prompt formatting
├── evaluate.py                # Core behavioral evaluation
├── extract_activations.py     # Residual activation extraction
├── activation_patching.py     # Residual-stream patching
├── extract_head_activations.py# Head activation extraction
├── head_patching.py           # Head-level patching
├── logit_lens.py              # Logit-lens extraction from saved activations
├── modal_app.py               # Modal infrastructure and remote entrypoints
├── analysis/                  # Python analysis code only
│   ├── pair_analysis.py
│   ├── probe.py
│   ├── probe_heads.py
│   ├── probe_l2_sweep.py
│   └── plot_logit_lens.py
├── reports/                   # Tracked writeups and presentation-quality figures
│   ├── summary.md
│   ├── probing_findings.md
│   ├── week9_findings.md
│   ├── failure_direction_summary.md
│   ├── logit_lens_findings.md
│   └── figures/
├── artifacts/                 # Tracked raw analysis outputs
│   └── analysis/
├── eval/                      # Phase 1 behavioral outputs + selected pair set
└── results_download/          # Downloaded Modal outputs / local exported results
```

## Setup

### Prerequisites
1. Modal account with GPU access
2. HuggingFace account with API token
3. Modal secret named `huggingface-secret` containing your HF token

### Installation

1. **Install Modal CLI:**
```bash
pip install modal
```

2. **Authenticate with Modal:**
```bash
modal token new
```

3. **Create HuggingFace secret in Modal:**
```bash
modal secret create huggingface-secret HF_TOKEN=your_huggingface_token_here
```

### Verify Setup

Run a quick smoke test to verify model loading works:
```bash
modal run modal_app.py::smoke_test
```

This will:
- Build the Modal image (first run ~5-10 minutes)
- Download Qwen3-8B weights (~15GB, first run only)
- Verify model loading and basic functionality
- Test token verification

## Usage

### Run Full Evaluation

```bash
modal run modal_app.py::launch_eval
```

This will:
1. Build the Modal image (cached after first run)
2. Load Qwen3-8B via TransformerLens
3. Evaluate all Com2Sense examples
4. Save results to Modal volume
5. Print summary to stdout

**Runtime:** ~30-60 minutes (depends on dataset size and caching)

### Run Logit Lens Analysis

```bash
uv run modal run modal_app.py::logit_lens
```

This will:
1. Ensure residual activations exist
2. Project each layer's residual stream through the final unembedding
3. Save layer-by-layer True/False trajectories
4. Generate summary plots and findings inputs

### Output Locations

#### `eval/`
Behavioral evaluation outputs tracked in the repo:

#### 1. `predictions.jsonl`
Per-example predictions in JSON Lines format:
```json
{
  "example_id": "ex_0",
  "sentence": "A rock would make a good pillow",
  "label": false,
  "predicted_true": false,
  "correct": true,
  "true_logit": -2.34,
  "false_logit": 1.56,
  "confidence": 0.94,
  "domain": "physical",
  "scenario": "causal",
  "pair_id": "pair_123"
}
```

#### 2. `accuracy_table.json`
Aggregated accuracy statistics:
```json
{
  "overall": {"accuracy": 0.72, "total": 2390, "correct": 1721},
  "by_domain": {
    "physical": {"accuracy": 0.75, "total": 800, "correct": 600},
    "social": {"accuracy": 0.68, "total": 795, "correct": 541},
    "temporal": {"accuracy": 0.73, "total": 795, "correct": 580}
  },
  "by_scenario": {...},
  "cross_tab": {...}
}
```

#### 3. `asymmetric_pairs.json`
Complementary pairs where the model gets one right and one wrong (TARGET for Phase 2):
```json
[
  {
    "pair_id": "pair_123",
    "example_a_id": "ex_45",
    "example_b_id": "ex_46",
    "sentence_a": "Rocks are soft enough to sleep on",
    "sentence_b": "Rocks are too hard to sleep on comfortably",
    "label_a": false,
    "label_b": true,
    "a_correct": false,
    "b_correct": true,
    "category": "asymmetric",
    "correct_example_id": "ex_46",
    "incorrect_example_id": "ex_45"
  }
]
```

#### 4. `pair_summary.json`
All complementary pair classifications with category counts.

#### 5. `summary.txt`
Human-readable summary with key statistics.

### Output Organization

#### `analysis/`
Python analysis code only. This directory should not contain generated `.json`, `.png`, or findings `.md` files.

#### `reports/`
Tracked findings documents and presentation-quality figures.

Examples:
- `reports/summary.md`
- `reports/probing_findings.md`
- `reports/week9_findings.md`
- `reports/logit_lens_findings.md`
- `reports/figures/`

#### `artifacts/analysis/`
Tracked raw analysis outputs such as:
- `probe_results.json`
- `l2_sweep_results.json`
- `patching_results.json`
- `head_probe_results.json`
- `head_patching_results.json`

#### `results_download/`
Local downloaded outputs from Modal, including downloaded logit-lens raw files.

### Access Results

Results are stored in Modal volumes. To download locally:
```bash
modal volume get comsense-results /root/comsense-circuits/results ./local_results
```

## Success Criteria

Before moving to Phase 2, verify:

- ✅ Model loads and runs inference on A100-80GB
- ✅ Accuracy is between 55-85% (not random, not perfect)
- ✅ At least 100 asymmetric complementary pairs identified
- ✅ Accuracy varies meaningfully across domains/scenarios
- ✅ All 5 output files are written and parseable

## Technical Details

### Model Loading
- Uses TransformerLens `HookedTransformer.from_pretrained()`
- Falls back to smaller models if Qwen3-8B isn't available
- Fallback order: Qwen3-8B → Qwen3-8B-Base → Qwen3-4B → Qwen3-0.6B-Base

### Token Verification
- Tries multiple variants: " True", "True", " true", "true"
- Requires single-token representation for clean logit comparison
- Fails fast if tokens can't be verified

### Evaluation Method
- Forward pass only (NO generation)
- Extracts logits at final token position
- Compares logits for True vs False token IDs
- Uses softmax over just the two tokens for confidence

### Qwen3 Non-Thinking Mode
- Prepends `/no_think` to disable chain-of-thought
- Critical for clean residual stream representations
- Ensures model produces direct answers, not CoT reasoning

### Complementary Pairs
- Com2Sense organizes statements in pairs with opposite truth values
- Same scenario, different truth values (e.g., "Rocks are soft" vs "Rocks are hard")
- Asymmetric pairs (one right, one wrong) are targets for activation patching

## Troubleshooting

### Model Loading Fails
- Check TransformerLens supports Qwen3 models
- Error will indicate if model name is wrong
- Fallback models will be tried automatically

### Dataset Loading Fails
+- Verify HuggingFace dataset exists: `tasksource/com2sense`
+- Fallback names are tried: `dali-does/com2sense`, `com2sense/com2sense`, `com2sense`
- Check HF token is valid in Modal secret

### OOM Errors
- Shouldn't happen on A100-80GB for 8B model
- If occurs, process examples one at a time (batch_size=1)
- Or use smaller model (Qwen3-4B)

### Low Accuracy (~50%)
- Check token verification (wrong tokens = garbage results)
- Verify prompt template format
- Check model is using non-thinking mode

## Project Status

The project now includes:

1. **Phase 1 — Behavioral Evaluation**
   - evaluate Qwen3-8B on Com2Sense
   - identify asymmetric complementary pairs

2. **Phase 2 — Representation Probing**
   - probe residual-stream activations by layer
   - compare correctness vs. ground-truth signal

3. **Phase 3 — Mechanistic Intervention**
   - activation patching
   - head-level extraction and patching
   - L2 regularization sweep
   - head-level probing

4. **Follow-up — Logit Lens**
   - inspect layer-by-layer answer trajectories
   - test whether failures emerge early, late, or both

See `reports/summary.md` and `reports/logit_lens_findings.md` for the current high-level conclusions.

## References

- **Com2Sense Dataset:** [HuggingFace](https://huggingface.co/datasets/tasksource/com2sense)
- **TransformerLens:** [GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- **Qwen3 Model:** [HuggingFace](https://huggingface.co/Qwen)
- **Modal:** [Documentation](https://modal.com/docs)

## License

This project is for research purposes. Please refer to the original dataset and model licenses for usage restrictions.

## Contact

For questions or issues, please open an issue in the repository.