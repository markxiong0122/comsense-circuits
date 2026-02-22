# ComSense Circuits — Phase 1: Behavioral Evaluation

A mechanistic interpretability study of commonsense reasoning failures in Large Language Models (LLMs). This project evaluates **Qwen3-8B** on the **Com2Sense** dataset to identify failure patterns and asymmetric complementary pairs for later probing and activation patching work.

## Overview

### Goal
Build infrastructure to:
1. Load Qwen3-8B via TransformerLens on Modal (A100-80GB GPU)
2. Evaluate performance on Com2Sense True/False commonsense statements
3. Identify complementary pairs where the model gets one right and one wrong
4. Save structured results for Phase 2 (probing and patching)

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
├── config.py           # Central configuration (model, dataset, prompts)
├── data_utils.py       # Dataset loading and prompt formatting
├── modal_app.py        # Modal infrastructure (GPU, volumes, entry points)
├── evaluate.py         # Core evaluation script
├── README.md           # This file
└── results/            # Output directory (created at runtime)
    └── eval/
        ├── predictions.jsonl        # Per-example results
        ├── accuracy_table.json      # Aggregated accuracy stats
        ├── asymmetric_pairs.json    # Pairs with one right, one wrong
        ├── pair_summary.json        # All pair classifications
        └── summary.txt              # Human-readable overview
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

### Output Files

All results are saved to `/root/comsense-circuits/results/eval/`:

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

## Phase 2 Preview

The next phase will use the asymmetric pairs identified here to:
1. **Probing:** Train classifiers on model activations to predict correct/incorrect
2. **Activation Patching:** Swap activations between correct and incorrect examples
3. **Circuit Analysis:** Identify specific attention heads and MLP layers responsible for reasoning failures

## References

+- **Com2Sense Dataset:** [HuggingFace](https://huggingface.co/datasets/tasksource/com2sense)
- **TransformerLens:** [GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- **Qwen3 Model:** [HuggingFace](https://huggingface.co/Qwen)
- **Modal:** [Documentation](https://modal.com/docs)

## License

This project is for research purposes. Please refer to the original dataset and model licenses for usage restrictions.

## Contact

For questions or issues, please open an issue in the repository.