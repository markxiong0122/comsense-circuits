"""
MLP vs Attention sublayer patching across layers 18-35.

The previous resid_post patching told us no single layer is sufficient to flip
predictions, but it conflates two contributions: attention and MLP. This script
decomposes each layer into its sublayer components by patching hook_attn_out and
hook_mlp_out separately, answering: is the distributed failure concentrated in
attention or in MLP layers?

Method:
  For each pair, for each layer L in [18..35]:
    1. Cache correct example's attn_out and mlp_out at all target layers.
    2. Run incorrect example with a hook replacing ONLY hook_attn_out[L] at
       the final token → measures attention sublayer contribution.
    3. Run incorrect example with a hook replacing ONLY hook_mlp_out[L] at
       the final token → measures MLP sublayer contribution.
    4. Record logit change and whether prediction flips.

Hook shapes (TransformerLens):
  hook_attn_out : [batch, seq_len, d_model]  (attention output before residual add)
  hook_mlp_out  : [batch, seq_len, d_model]  (MLP output before residual add)

These are the *deltas* added to the residual stream, so patching them is a clean
decomposition: attn_out + mlp_out = resid_post - resid_pre.

Output:
  mlp_attn_patching_results.json  — per-layer results for both sublayer types
  mlp_attn_patching_results.png   — side-by-side comparison plot
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

PATCH_LAYERS = list(range(18, 36))  # layers 18-35, matching original patching scope


def get_logit_gap(logits: torch.Tensor, true_id: int, false_id: int) -> float:
    """Return true_logit - false_logit at the final token position."""
    final = logits[0, -1, :]
    return final[true_id].item() - final[false_id].item()


def run_sublayer_patched(
    model,
    incorrect_prompt: str,
    cached_vec: torch.Tensor,
    hook_name: str,
) -> torch.Tensor:
    """
    Run incorrect_prompt with a single hook replacing one sublayer's output
    at the final token position with cached_vec from the correct example.

    cached_vec: [d_model] — the correct example's sublayer output at this layer/token.
    hook_name:  e.g. "blocks.20.hook_attn_out" or "blocks.20.hook_mlp_out"
    """
    def patch_hook(activation, hook):
        # activation: [batch, seq_len, d_model]
        activation[0, -1, :] = cached_vec
        return activation

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            incorrect_prompt,
            fwd_hooks=[(hook_name, patch_hook)],
        )
    return patched_logits


def main() -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("MLP vs Attention Sublayer Patching")
    logger.info("=" * 60)

    code_dir = Path(__file__).parent
    out_dir = Path(RESULTS_DIR) / "mlp_attn_patching"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs
    selected_path = code_dir / "eval" / "selected_pairs_for_patching.json"
    with open(selected_path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} pairs")

    # Load model
    model, model_name = load_model()
    true_id, false_id = verify_tokens(model)
    logger.info(f"Model: {model_name} | {model.cfg.n_layers} layers")
    logger.info(f"Patching layers: {PATCH_LAYERS[0]}-{PATCH_LAYERS[-1]}")

    # Accumulators: sublayer_type -> layer -> list of logit changes
    sublayer_types = ["attn", "mlp"]
    logit_changes = {st: {L: [] for L in PATCH_LAYERS} for st in sublayer_types}
    flip_counts   = {st: {L: 0  for L in PATCH_LAYERS} for st in sublayer_types}
    pair_details  = []

    for pair_idx, pair in enumerate(tqdm(pairs, desc="Patching pairs")):
        correct_prompt   = format_prompt(pair["sentence_correct"])
        incorrect_prompt = format_prompt(pair["sentence_incorrect"])

        # Cache both sublayer outputs for all target layers in one forward pass
        def names_filter(name: str) -> bool:
            for L in PATCH_LAYERS:
                if name == f"blocks.{L}.hook_attn_out":
                    return True
                if name == f"blocks.{L}.hook_mlp_out":
                    return True
            return False

        with torch.no_grad():
            _, correct_cache = model.run_with_cache(
                correct_prompt,
                names_filter=names_filter,
            )

        # Baseline: run incorrect example normally
        with torch.no_grad():
            baseline_logits = model(incorrect_prompt)
        baseline_gap = get_logit_gap(baseline_logits, true_id, false_id)
        baseline_pred_true = baseline_gap > 0
        gt_label = bool(pair["label_incorrect"])

        pair_result = {
            "pair_idx":              pair_idx,
            "pair_id":               pair["pair_id"],
            "domain":                pair["domain"],
            "baseline_gap":          baseline_gap,
            "baseline_pred_true":    baseline_pred_true,
            "ground_truth_label":    gt_label,
            "patched_attn":          {},
            "patched_mlp":           {},
        }

        for L in PATCH_LAYERS:
            for sublayer, cache_key, result_key in [
                ("attn", f"blocks.{L}.hook_attn_out", "patched_attn"),
                ("mlp",  f"blocks.{L}.hook_mlp_out",  "patched_mlp"),
            ]:
                hook_name  = cache_key
                cached_vec = correct_cache[cache_key][0, -1, :]  # [d_model]

                patched_logits = run_sublayer_patched(
                    model, incorrect_prompt, cached_vec, hook_name
                )
                patched_gap = get_logit_gap(patched_logits, true_id, false_id)

                delta   = patched_gap - baseline_gap
                flipped = (patched_gap > 0) != baseline_pred_true

                logit_changes[sublayer][L].append(delta)
                if flipped:
                    flip_counts[sublayer][L] += 1

                pair_result[result_key][str(L)] = {
                    "logit_change": delta,
                    "flipped":      flipped,
                    "now_correct":  (patched_gap > 0) == gt_label,
                }

        pair_details.append(pair_result)

        if (pair_idx + 1) % 20 == 0:
            logger.info(f"  Processed {pair_idx + 1}/{len(pairs)}")

    # Summary statistics
    n_pairs = len(pairs)
    layer_summary = {}
    for L in PATCH_LAYERS:
        layer_summary[str(L)] = {}
        for st in sublayer_types:
            changes = logit_changes[st][L]
            layer_summary[str(L)][st] = {
                "flip_count":        flip_counts[st][L],
                "flip_rate":         flip_counts[st][L] / n_pairs,
                "mean_logit_change": float(sum(changes) / len(changes)),
            }

    # Per-domain flip rates
    domain_flips = {}
    for pr in pair_details:
        d = pr["domain"]
        if d not in domain_flips:
            domain_flips[d] = {st: {str(L): {"flips": 0, "total": 0}
                                    for L in PATCH_LAYERS}
                               for st in sublayer_types}
        for L in PATCH_LAYERS:
            for st, rkey in [("attn", "patched_attn"), ("mlp", "patched_mlp")]:
                domain_flips[d][st][str(L)]["total"] += 1
                if pr[rkey][str(L)]["flipped"]:
                    domain_flips[d][st][str(L)]["flips"] += 1

    domain_summary = {}
    for d, st_data in domain_flips.items():
        domain_summary[d] = {
            st: {L: v["flips"] / v["total"] if v["total"] > 0 else 0
                 for L, v in layer_data.items()}
            for st, layer_data in st_data.items()
        }

    results = {
        "model_name":     model_name,
        "n_pairs":        n_pairs,
        "patch_layers":   PATCH_LAYERS,
        "layer_summary":  layer_summary,
        "domain_summary": domain_summary,
        "pair_details":   pair_details,
    }

    out_path = out_dir / "mlp_attn_patching_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {out_path}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("MLP vs Attention Sublayer Patching Summary")
    logger.info(f"{'Layer':<8} {'Attn flip%':<14} {'Attn Δlogit':<16} {'MLP flip%':<14} {'MLP Δlogit'}")
    logger.info("-" * 70)
    for L in PATCH_LAYERS:
        s = layer_summary[str(L)]
        logger.info(
            f"  {L:<6d} "
            f"{s['attn']['flip_rate']*100:>8.1f}%     "
            f"{s['attn']['mean_logit_change']:>+10.3f}       "
            f"{s['mlp']['flip_rate']*100:>7.1f}%     "
            f"{s['mlp']['mean_logit_change']:>+10.3f}"
        )

    # Plot
    _plot(results, out_dir)

    return results


def _plot(results: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    layers = results["patch_layers"]
    layer_summary = results["layer_summary"]

    attn_flip  = [layer_summary[str(L)]["attn"]["flip_rate"] * 100 for L in layers]
    mlp_flip   = [layer_summary[str(L)]["mlp"]["flip_rate"]  * 100 for L in layers]
    attn_delta = [layer_summary[str(L)]["attn"]["mean_logit_change"] for L in layers]
    mlp_delta  = [layer_summary[str(L)]["mlp"]["mean_logit_change"]  for L in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(layers))
    width = 0.38

    # Flip rate panel
    ax1.bar(x - width / 2, attn_flip, width, label="Attention (hook_attn_out)",
            color="#2563eb", alpha=0.85)
    ax1.bar(x + width / 2, mlp_flip,  width, label="MLP (hook_mlp_out)",
            color="#16a34a", alpha=0.85)
    ax1.set_ylabel("Flip rate (%)")
    ax1.set_title(
        "MLP vs Attention Sublayer Patching — Qwen3-8B on Com2Sense\n"
        "Correct example's sublayer output patched into incorrect example"
    )
    ax1.legend(fontsize=9)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", alpha=0.3)

    # Mean logit change panel
    ax2.plot(x, attn_delta, marker="o", markersize=4, linewidth=1.5,
             color="#2563eb", label="Attention Δlogit")
    ax2.plot(x, mlp_delta,  marker="s", markersize=4, linewidth=1.5,
             color="#16a34a", label="MLP Δlogit")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Mean logit change (True − False)")
    ax2.set_xlabel("Layer")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)

    fig.tight_layout()
    plot_path = out_dir / "mlp_attn_patching_results.png"
    fig.savefig(plot_path, dpi=150)
    import matplotlib.pyplot as plt_close
    plt_close.close(fig)
    logger.info(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
