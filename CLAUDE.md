# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mechanistic interpretability study of commonsense reasoning failures in LLMs. Phase 1 evaluates Qwen3-8B on Com2Sense True/False statements to identify "asymmetric pairs" (complementary statements where the model gets one right and one wrong) for later probing and activation patching work.

**Status: Phase 1 COMPLETE** - 611 asymmetric pairs identified from 1,195 properly-matched complementary pairs.

## Commands

```bash
# Verify setup (model loading, CUDA, token verification)
uv run modal run modal_app.py::smoke_test

# Run full evaluation (~2 min inference on A100-80GB)
uv run modal run modal_app.py::launch_eval

# Download results from Modal volume
uv run modal volume get comsense-results /eval ./local_results
```

**Prerequisites:**
- `uv sync` to install dependencies
- Modal CLI authenticated (`modal token new`)
- HuggingFace secret in Modal (`modal secret create huggingface-secret HF_TOKEN=...`)

## Architecture

Single-directional data flow through four modules:

```
config.py → data_utils.py → evaluate.py ← modal_app.py
```

| File | Purpose |
|------|---------|
| `config.py` | Central configuration: model name, dataset name, token candidates, prompt template |
| `data_utils.py` | Dataset loading from HuggingFace, prompt formatting, complementary pair assignment via GitHub mapping |
| `evaluate.py` | Core evaluation: model loading, token verification, forward-pass inference, results aggregation |
| `modal_app.py` | Modal infrastructure: A100-80GB GPU, HuggingFace cache volume, results volume, entry points |

## Key Technical Details

**Model Loading (evaluate.py:33-63):**
- Uses TransformerLens `HookedTransformer.from_pretrained()` for interpretability access
- Fallback chain: Qwen3-8B → Qwen3-8B-Base → Qwen3-4B → Qwen3-0.6B-Base
- dtype: bfloat16

**Token Verification (evaluate.py:66-111):**
- Critical step - must find single-token representations for "True"/"False"
- Tries variants: `" True"`, `"True"`, `" true"`, `"true"` (and false equivalents)
- Fails fast if tokens can't be verified (garbage results otherwise)

**Evaluation Method (evaluate.py:114-171):**
- Forward pass only (NO generation)
- Extracts logits at final token position
- Compares `true_logit` vs `false_logit` for prediction
- Confidence via softmax over just the two token logits

**Complementary Pairs (data_utils.py:233-292):**
- **IMPORTANT:** Uses official GitHub pair_id mapping from `PlusLabNLP/Com2Sense`
- Downloads `pair_id_train.json` and `pair_id_dev.json` from GitHub
- Maps `original_id` (hex ID from HuggingFace dataset) to companion ID
- Falls back to adjacency heuristic only if GitHub download fails
- "Asymmetric" pairs (one correct, one wrong) are targets for Phase 2 patching

## Modal Infrastructure

- **GPU:** A100-80GB (required for 8B model at bfloat16)
- **Volumes:**
  - `comsense-hf-cache` → `/root/.cache/huggingface` (model weights, ~15GB)
  - `comsense-results` → `/root/comsense-circuits/results` (evaluation outputs)
- **Timeout:** 2 hours

## Output Structure

Results written to `eval/`:
- `predictions.jsonl` - Per-example results with logits and confidence (2,390 examples)
- `accuracy_table.json` - Aggregated by domain, scenario, cross-tab
- `asymmetric_pairs.json` - 611 pairs with one right, one wrong (Phase 2 targets)
- `pair_summary.json` - All 1,195 pair classifications
- `summary.txt` - Human-readable overview

## Phase 1 Results (Feb 2026)

- **Model:** Qwen/Qwen3-8B
- **Overall Accuracy:** 70.25%
- **Valid Complementary Pairs:** 1,195 (100% paired, 0 orphans)
- **Asymmetric Pairs:** 611 (one correct, one incorrect)
- **Both Correct:** 534 pairs
- **Both Wrong:** 50 pairs

## Phase 2 Next Steps

1. **Activation Extraction** - Modify evaluation to save residual stream activations for asymmetric pairs
2. **Probing** - Train classifiers on activations to predict correct/incorrect
3. **Activation Patching** - Swap activations between correct/incorrect examples to find causal circuits
4. **Circuit Analysis** - Identify specific attention heads and MLP layers responsible for reasoning failures

## Configuration Notes

When changing models or datasets, update `config.py`:
- `MODEL_NAME` / `MODEL_NAME_FALLBACKS` - HuggingFace model IDs
- `DATASET_NAME` / `DATASET_NAME_FALLBACKS` - HuggingFace dataset IDs
- `TRUE_TOKEN_CANDIDATES` / `FALSE_TOKEN_CANDIDATES` - Token variants to try
- `PROMPT_TEMPLATE` - Prompt format for True/False classification

## Lessons Learned

- The HuggingFace `tasksource/com2sense` dataset doesn't include `pair_id` - must download from GitHub
- Adjacency heuristic produces incorrect pairs (unrelated sentences grouped together)
- GitHub mapping provides 2,412 bidirectional pair entries covering train and dev splits
- Run `uv run modal ...` not just `modal ...` to use the project's virtual environment
