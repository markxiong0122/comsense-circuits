"""
Phase 3c: Head-level activation patching.

After activation patching identifies the most critical layers, this script
patches individual attention heads within those layers to isolate specific
heads responsible for commonsense reasoning failures.

For each pair, for each head in the target layers:
  1. Run incorrect example with a hook that replaces ONLY that head's z output
     at the final token with the correct example's cached z vector.
  2. Measure the logit change (toward or away from correct answer).

Output:
  head_patching_results.json — per-head logit effects
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


def main() -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Phase 3c: Head-Level Patching")
    logger.info("=" * 60)

    code_dir = Path(__file__).parent
    out_dir = Path(RESULTS_DIR) / "head_patching"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load patching results to find critical layers
    patching_path = Path(RESULTS_DIR) / "patching" / "patching_results.json"
    if patching_path.exists():
        with open(patching_path) as f:
            patching_data = json.load(f)
        # Pick top 5 layers by flip rate
        layer_summary = patching_data["layer_summary"]
        sorted_layers = sorted(layer_summary.items(), key=lambda x: -x[1]["flip_rate"])
        target_layers = [int(l) for l, _ in sorted_layers[:5]]
        logger.info(f"Top 5 layers by flip rate: {target_layers}")
    else:
        # Fallback: use layers 24-32 based on probing results
        target_layers = [24, 25, 26, 27, 28, 29, 30, 31, 32]
        logger.info(f"No patching results found, using default layers: {target_layers}")

    # Load pairs
    selected_path = code_dir / "eval" / "selected_pairs_for_patching.json"
    with open(selected_path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Load model
    model, model_name = load_model()
    true_id, false_id = verify_tokens(model)
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    logger.info(f"Model: {model_name} | {n_heads} heads | d_head={d_head}")
    logger.info(f"Target layers: {target_layers}")

    # head_effects[layer][head] = list of logit changes
    head_effects = {L: {h: [] for h in range(n_heads)} for L in target_layers}
    head_flips = {L: {h: 0 for h in range(n_heads)} for L in target_layers}

    for pair_idx, pair in enumerate(tqdm(pairs, desc="Head patching")):
        correct_prompt = format_prompt(pair["sentence_correct"])
        incorrect_prompt = format_prompt(pair["sentence_incorrect"])

        # Cache correct example's z at target layers
        layer_set = set(target_layers)

        def z_filter(name: str) -> bool:
            if name.endswith("hook_z"):
                parts = name.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) in layer_set
            return False

        with torch.no_grad():
            _, correct_cache = model.run_with_cache(correct_prompt, names_filter=z_filter)

        # Baseline for incorrect example
        with torch.no_grad():
            baseline_logits = model(incorrect_prompt)
        bl_true = baseline_logits[0, -1, true_id].item()
        bl_false = baseline_logits[0, -1, false_id].item()
        baseline_gap = bl_true - bl_false

        # Patch each head individually
        for L in target_layers:
            correct_z = correct_cache["z", L][0, -1, :, :]  # [n_heads, d_head]

            for h in range(n_heads):
                correct_z_head = correct_z[h, :]  # [d_head]

                def make_hook(head_idx, correct_vec):
                    def hook_fn(activation, hook):
                        # activation: [batch, seq_len, n_heads, d_head]
                        activation[0, -1, head_idx, :] = correct_vec
                        return activation
                    return hook_fn

                hook_name = f"blocks.{L}.attn.hook_z"

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        incorrect_prompt,
                        fwd_hooks=[(hook_name, make_hook(h, correct_z_head))],
                    )

                p_true = patched_logits[0, -1, true_id].item()
                p_false = patched_logits[0, -1, false_id].item()
                patched_gap = p_true - p_false
                logit_change = patched_gap - baseline_gap

                head_effects[L][h].append(logit_change)
                if (patched_gap > 0) != (baseline_gap > 0):
                    head_flips[L][h] += 1

        if (pair_idx + 1) % 20 == 0:
            logger.info(f"  Processed {pair_idx + 1}/{len(pairs)}")

    # Compute summary
    n_pairs = len(pairs)
    summary = {}
    for L in target_layers:
        layer_data = {}
        for h in range(n_heads):
            effects = head_effects[L][h]
            layer_data[str(h)] = {
                "mean_logit_change": float(sum(effects) / len(effects)),
                "abs_mean_logit_change": float(sum(abs(e) for e in effects) / len(effects)),
                "flip_count": head_flips[L][h],
                "flip_rate": head_flips[L][h] / n_pairs,
            }
        summary[str(L)] = layer_data

    # Find top heads
    all_heads = []
    for L in target_layers:
        for h in range(n_heads):
            all_heads.append({
                "layer": L,
                "head": h,
                "abs_mean_effect": summary[str(L)][str(h)]["abs_mean_logit_change"],
                "mean_effect": summary[str(L)][str(h)]["mean_logit_change"],
                "flip_rate": summary[str(L)][str(h)]["flip_rate"],
            })
    all_heads.sort(key=lambda x: -x["abs_mean_effect"])
    top_20 = all_heads[:20]

    results = {
        "model_name": model_name,
        "n_pairs": n_pairs,
        "target_layers": target_layers,
        "n_heads": n_heads,
        "summary": summary,
        "top_20_heads": top_20,
    }

    out_path = out_dir / "head_patching_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    logger.info("\nTop 20 heads by absolute mean logit effect:")
    for entry in top_20:
        logger.info(
            f"  L{entry['layer']:2d}.H{entry['head']:2d}: "
            f"mean={entry['mean_effect']:+.4f} "
            f"|mean|={entry['abs_mean_effect']:.4f} "
            f"flip_rate={entry['flip_rate']:.3f}"
        )

    return results


if __name__ == "__main__":
    main()
