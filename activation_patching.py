"""
Phase 3: Activation patching on residual stream.

For each of the 200 selected pairs, patches the residual stream at each layer
(20-35) from the correct example into the incorrect example's forward pass,
measuring whether the output flips from incorrect to correct.

This provides CAUSAL evidence for which layers control commonsense judgments,
beyond the correlational evidence from probing.

Method:
  For each pair:
    1. Run the incorrect sentence normally to get baseline logits
    2. For each layer L in [20..35]:
       - Run the incorrect sentence with a hook that replaces resid_post[L]
         at the final token position with the correct example's cached activation
       - Record whether the prediction flips

Output:
  patching_results.json — per-layer flip rates and per-pair details
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from config import RESULTS_DIR
from data_utils import format_prompt
from evaluate import load_model, verify_tokens

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PATCH_LAYERS = list(range(18, 36))  # layers 18-35


def get_prediction(logits: torch.Tensor, true_id: int, false_id: int) -> dict:
    """Extract prediction from final-position logits."""
    final = logits[0, -1, :]
    true_logit = final[true_id].item()
    false_logit = final[false_id].item()
    predicted_true = true_logit > false_logit
    probs = torch.softmax(torch.tensor([true_logit, false_logit]), dim=0)
    confidence = probs[0].item() if predicted_true else probs[1].item()
    return {
        "predicted_true": bool(predicted_true),
        "true_logit": true_logit,
        "false_logit": false_logit,
        "confidence": confidence,
        "logit_gap": abs(true_logit - false_logit),
    }


def run_patched(
    model,
    incorrect_prompt: str,
    correct_cache: dict,
    patch_layer: int,
) -> torch.Tensor:
    """
    Run incorrect_prompt through the model, but at patch_layer replace
    the residual stream at the final token with the correct example's activation.
    """
    # Get correct example's resid_post at the target layer, final token
    correct_resid = correct_cache["resid_post", patch_layer][0, -1, :]  # [d_model]

    def patch_hook(activation, hook):
        # activation shape: [batch, seq_len, d_model]
        activation[0, -1, :] = correct_resid
        return activation

    hook_name = f"blocks.{patch_layer}.hook_resid_post"

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            incorrect_prompt,
            fwd_hooks=[(hook_name, patch_hook)],
        )

    return patched_logits


def main() -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Phase 3: Activation Patching")
    logger.info("=" * 60)

    code_dir = Path(__file__).parent
    out_dir = Path(RESULTS_DIR) / "patching"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load selected pairs
    selected_path = code_dir / "eval" / "selected_pairs_for_patching.json"
    with open(selected_path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Load model
    model, model_name = load_model()
    true_id, false_id = verify_tokens(model)
    n_layers = model.cfg.n_layers
    logger.info(f"Model: {model_name} | {n_layers} layers")

    # Run patching
    layer_flip_counts = {L: 0 for L in PATCH_LAYERS}
    layer_logit_changes = {L: [] for L in PATCH_LAYERS}
    pair_details = []

    for pair_idx, pair in enumerate(tqdm(pairs, desc="Patching pairs")):
        correct_prompt = format_prompt(pair["sentence_correct"])
        incorrect_prompt = format_prompt(pair["sentence_incorrect"])

        # Cache correct example activations
        with torch.no_grad():
            _, correct_cache = model.run_with_cache(
                correct_prompt,
                names_filter=lambda name: name.endswith("hook_resid_post"),
            )

        # Baseline: run incorrect example normally
        with torch.no_grad():
            baseline_logits = model(incorrect_prompt)
        baseline = get_prediction(baseline_logits, true_id, false_id)

        # Ground truth for incorrect example
        gt_label = bool(pair["label_incorrect"])

        pair_result = {
            "pair_idx": pair_idx,
            "pair_id": pair["pair_id"],
            "domain": pair["domain"],
            "baseline_predicted_true": baseline["predicted_true"],
            "ground_truth_label": gt_label,
            "baseline_correct": baseline["predicted_true"] == gt_label,
            "patched_layers": {},
        }

        for L in PATCH_LAYERS:
            patched_logits = run_patched(model, incorrect_prompt, correct_cache, L)
            patched = get_prediction(patched_logits, true_id, false_id)

            flipped = patched["predicted_true"] != baseline["predicted_true"]
            now_correct = patched["predicted_true"] == gt_label
            logit_change = (patched["true_logit"] - patched["false_logit"]) - \
                           (baseline["true_logit"] - baseline["false_logit"])

            if flipped:
                layer_flip_counts[L] += 1
            layer_logit_changes[L].append(logit_change)

            pair_result["patched_layers"][str(L)] = {
                "flipped": flipped,
                "now_correct": now_correct,
                "logit_change": logit_change,
                "patched_confidence": patched["confidence"],
            }

        pair_details.append(pair_result)

        if (pair_idx + 1) % 20 == 0:
            logger.info(f"  Processed {pair_idx + 1}/{len(pairs)}")

    # Compute summary statistics
    n_pairs = len(pairs)
    layer_summary = {}
    for L in PATCH_LAYERS:
        changes = layer_logit_changes[L]
        layer_summary[str(L)] = {
            "flip_count": layer_flip_counts[L],
            "flip_rate": layer_flip_counts[L] / n_pairs,
            "mean_logit_change": float(sum(changes) / len(changes)),
            "median_logit_change": float(sorted(changes)[len(changes) // 2]),
        }

    # Per-domain flip rates at each layer
    domain_flips = {}
    for pair_result in pair_details:
        domain = pair_result["domain"]
        if domain not in domain_flips:
            domain_flips[domain] = {str(L): {"flips": 0, "total": 0} for L in PATCH_LAYERS}
        for L in PATCH_LAYERS:
            domain_flips[domain][str(L)]["total"] += 1
            if pair_result["patched_layers"][str(L)]["flipped"]:
                domain_flips[domain][str(L)]["flips"] += 1

    domain_summary = {}
    for domain, layer_data in domain_flips.items():
        domain_summary[domain] = {
            L: d["flips"] / d["total"] if d["total"] > 0 else 0
            for L, d in layer_data.items()
        }

    results = {
        "model_name": model_name,
        "n_pairs": n_pairs,
        "patch_layers": PATCH_LAYERS,
        "layer_summary": layer_summary,
        "domain_summary": domain_summary,
        "pair_details": pair_details,
    }

    out_path = out_dir / "patching_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {out_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Activation Patching Summary")
    logger.info("=" * 60)
    for L in PATCH_LAYERS:
        s = layer_summary[str(L)]
        logger.info(
            f"  Layer {L:2d}: flip_rate={s['flip_rate']:.3f} "
            f"mean_logit_change={s['mean_logit_change']:+.3f}"
        )

    return results


if __name__ == "__main__":
    main()
