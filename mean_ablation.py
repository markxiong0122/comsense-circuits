"""
Mean-ablation study on attention heads.

Tests NECESSITY: does removing a head's contribution degrade correct predictions?
This complements activation patching (which tests SUFFICIENCY).

For each target head, replaces its z output with the mean z across all examples
at that position, then measures whether correct predictions flip to incorrect.

Targets the top heads identified by patching (L20.H8, L21.H19, L26.H26) plus
the top probing heads (L32.H20, L34.H18) for comparison.

Output:
  ablation_results.json — per-head degradation metrics
"""

import json
import logging
from collections import defaultdict
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

# Top heads from patching (causal effect) and probing (information)
TARGET_HEADS = [
    # Patching top heads
    (20, 8), (21, 19), (26, 26), (26, 25), (19, 31),
    (18, 16), (21, 15), (21, 16), (18, 6), (20, 9),
    # Probing top heads
    (32, 20), (34, 18), (28, 11), (24, 11), (28, 25),
    (32, 12), (35, 23),
]


def compute_mean_z(model, pairs: list, target_layers: set) -> dict:
    """
    Compute the mean z vector for each (layer, head) across all examples.
    This is the "default" activation we replace with during ablation.
    """
    logger.info("Computing mean z vectors across all examples...")

    z_sums = {}
    z_counts = {}

    for pair in tqdm(pairs, desc="Computing mean z"):
        for role in ("correct", "incorrect"):
            sentence = pair[f"sentence_{role}"]
            prompt = format_prompt(sentence)

            def names_filter(name):
                if name.endswith("hook_z"):
                    parts = name.split(".")
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1]) in target_layers
                return False

            with torch.no_grad():
                _, cache = model.run_with_cache(prompt, names_filter=names_filter)

            for L in target_layers:
                z = cache["z", L][0, -1, :, :]  # [n_heads, d_head] at final token
                if L not in z_sums:
                    z_sums[L] = z.clone()
                    z_counts[L] = 1
                else:
                    z_sums[L] += z
                    z_counts[L] += 1

    mean_z = {L: z_sums[L] / z_counts[L] for L in target_layers}
    return mean_z


def main() -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Mean-Ablation Study")
    logger.info("=" * 60)

    code_dir = Path(__file__).parent
    out_dir = Path(RESULTS_DIR) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load selected pairs
    selected_path = code_dir / "eval" / "selected_pairs_for_patching.json"
    with open(selected_path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Load model
    model, model_name = load_model()
    true_id, false_id = verify_tokens(model)
    logger.info(f"Model: {model_name}")

    # Determine target layers
    target_layers = set(L for L, H in TARGET_HEADS)
    logger.info(f"Target layers: {sorted(target_layers)}")
    logger.info(f"Target heads: {len(TARGET_HEADS)}")

    # Compute mean z for ablation
    mean_z = compute_mean_z(model, pairs, target_layers)

    # Run ablation on CORRECT examples only (test if ablation breaks correct predictions)
    head_results = {f"L{L}.H{H}": {"flips": 0, "logit_changes": [], "total": 0}
                    for L, H in TARGET_HEADS}

    for pair in tqdm(pairs, desc="Ablating"):
        # Only ablate the correct example — test if removing a head breaks it
        sentence = pair["sentence_correct"]
        gt_label = bool(pair["label_correct"])
        prompt = format_prompt(sentence)

        # Baseline: run without ablation
        with torch.no_grad():
            baseline_logits = model(prompt)
        bl_true = baseline_logits[0, -1, true_id].item()
        bl_false = baseline_logits[0, -1, false_id].item()
        baseline_pred_true = bl_true > bl_false
        baseline_correct = baseline_pred_true == gt_label

        if not baseline_correct:
            # Skip pairs where baseline is already wrong (shouldn't happen for "correct"
            # examples but guard against edge cases)
            continue

        # Ablate each target head
        for L, H in TARGET_HEADS:
            mean_vec = mean_z[L][H, :]  # [d_head]

            def make_hook(head_idx, replacement):
                def hook_fn(activation, hook):
                    # activation: [batch, seq_len, n_heads, d_head]
                    activation[0, -1, head_idx, :] = replacement
                    return activation
                return hook_fn

            hook_name = f"blocks.{L}.attn.hook_z"
            hook_fn = make_hook(H, mean_vec)

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    prompt,
                    fwd_hooks=[(hook_name, hook_fn)],
                )

            ab_true = ablated_logits[0, -1, true_id].item()
            ab_false = ablated_logits[0, -1, false_id].item()
            ablated_pred_true = ab_true > ab_false

            flipped = ablated_pred_true != baseline_pred_true
            logit_change = (ab_true - ab_false) - (bl_true - bl_false)

            key = f"L{L}.H{H}"
            head_results[key]["total"] += 1
            head_results[key]["logit_changes"].append(logit_change)
            if flipped:
                head_results[key]["flips"] += 1

    # Compute summaries
    import numpy as np

    summary = {}
    for key, data in head_results.items():
        total = data["total"]
        if total == 0:
            continue
        changes = data["logit_changes"]
        summary[key] = {
            "total": total,
            "flips": data["flips"],
            "flip_rate": data["flips"] / total,
            "mean_logit_change": float(np.mean(changes)),
            "median_logit_change": float(np.median(changes)),
            "std_logit_change": float(np.std(changes)),
            "source": "patching" if any(
                f"L{L}.H{H}" == key for L, H in TARGET_HEADS[:10]
            ) else "probing",
        }

    results = {
        "model_name": model_name,
        "n_pairs": len(pairs),
        "target_heads": [{"layer": L, "head": H} for L, H in TARGET_HEADS],
        "head_summary": summary,
        "top_by_flip_rate": sorted(
            summary.items(), key=lambda x: -x[1]["flip_rate"]
        )[:10],
        "top_by_abs_logit_change": sorted(
            summary.items(), key=lambda x: -abs(x[1]["mean_logit_change"])
        )[:10],
    }

    out_path = out_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {out_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Mean-Ablation Summary (correct examples only)")
    logger.info("=" * 60)
    logger.info(f"  {'Head':>10}  {'Flips':>6}  {'Rate':>6}  {'Mean dLogit':>11}  {'Source':>8}")
    for key in sorted(summary.keys(), key=lambda k: -abs(summary[k]["mean_logit_change"])):
        s = summary[key]
        logger.info(
            f"  {key:>10}  {s['flips']:6d}  {s['flip_rate']:6.3f}  "
            f"{s['mean_logit_change']:+11.4f}  {s['source']:>8}"
        )

    return results


if __name__ == "__main__":
    main()
