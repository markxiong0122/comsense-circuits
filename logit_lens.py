"""
Logit-lens extraction for selected asymmetric Com2Sense pairs.

This script reuses the saved Phase 2 residual activations (`activations.pt`) and
projects each layer's final-token residual stream through Qwen3-8B's final
normalization + unembedding stack. The result is a layer-by-layer trace of the
model's implicit True/False preference before the final layer.

Output (written to RESULTS_DIR/analysis/):
  logit_lens_results.json — per-pair, per-layer True/False logits and summaries

Why this is useful:
- Shows when the model first starts favoring the wrong answer
- Distinguishes early-vs-late failure trajectories
- Lets us compare correct and incorrect examples layer-by-layer
- Complements probing: probing asks "is signal linearly present?";
  logit lens asks "what answer is the model already moving toward?"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from config import RESULTS_DIR
from evaluate import load_model, verify_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _first_wrong_layer(correct_by_layer: list[bool]) -> int | None:
    """First layer where the intermediate prediction is wrong."""
    for layer_idx, is_correct in enumerate(correct_by_layer):
        if not is_correct:
            return layer_idx
    return None


def _stable_final_prediction_layer(predicted_true: list[bool]) -> int:
    """
    Earliest layer from which the model's intermediate prediction stays equal to
    the final-layer prediction for the rest of the network.
    """
    final_pred = predicted_true[-1]
    for layer_idx in range(len(predicted_true)):
        if all(p == final_pred for p in predicted_true[layer_idx:]):
            return layer_idx
    return len(predicted_true) - 1


def _stable_correctness_layer(correct_by_layer: list[bool]) -> int | None:
    """
    Earliest layer from which correctness stays fixed through the final layer.
    Returns None if no such stable segment exists.
    """
    final_correct = correct_by_layer[-1]
    for layer_idx in range(len(correct_by_layer)):
        if all(c == final_correct for c in correct_by_layer[layer_idx:]):
            return layer_idx
    return None


def ensure_activations_exist() -> Path:
    """
    Ensure residual activations from Phase 2 exist.

    Returns:
        Path to activations.pt
    """
    act_path = Path(RESULTS_DIR) / "activations" / "activations.pt"
    if act_path.exists():
        return act_path

    logger.info("activations.pt not found — extracting residual activations first...")
    from extract_activations import main as extract_main

    extract_main()

    if not act_path.exists():
        raise FileNotFoundError(
            f"Expected activations at {act_path}, but file was not created."
        )

    return act_path


@torch.no_grad()
def apply_logit_lens_to_residuals(
    model,
    resid_by_layer: torch.Tensor,
    true_token_id: int,
    false_token_id: int,
    ground_truth_label: bool,
) -> dict[str, Any]:
    """
    Project cached residual activations through ln_final + unembed at each layer.

    Args:
        model: Loaded HookedTransformer
        resid_by_layer: Tensor [n_layers, d_model] on CPU
        true_token_id: Token id for "True"
        false_token_id: Token id for "False"
        ground_truth_label: Whether the correct answer for this example is True

    Returns:
        Dictionary containing per-layer logits and derived summaries.
    """
    device = model.W_U.device
    dtype = model.W_U.dtype
    n_layers = resid_by_layer.shape[0]

    true_logits: list[float] = []
    false_logits: list[float] = []
    raw_logit_gap: list[float] = []
    signed_logit_gap: list[float] = []
    predicted_true: list[bool] = []
    correct_by_layer: list[bool] = []

    for layer in range(n_layers):
        resid = resid_by_layer[layer].to(device=device, dtype=dtype)
        resid = resid.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]

        normalized = model.ln_final(resid)
        logits = model.unembed(normalized)[0, 0, :]

        t_logit = float(logits[true_token_id].item())
        f_logit = float(logits[false_token_id].item())
        gap = t_logit - f_logit
        pred_true = gap > 0.0
        is_correct = pred_true == bool(ground_truth_label)

        # Signed so that positive = supports the correct answer, negative = supports the wrong answer.
        signed_gap = gap if ground_truth_label else -gap

        true_logits.append(t_logit)
        false_logits.append(f_logit)
        raw_logit_gap.append(float(gap))
        signed_logit_gap.append(float(signed_gap))
        predicted_true.append(bool(pred_true))
        correct_by_layer.append(bool(is_correct))

    first_wrong = _first_wrong_layer(correct_by_layer)
    stable_pred = _stable_final_prediction_layer(predicted_true)
    stable_correctness = _stable_correctness_layer(correct_by_layer)

    return {
        "label": bool(ground_truth_label),
        "true_logits": true_logits,
        "false_logits": false_logits,
        "raw_logit_gap": raw_logit_gap,
        "signed_logit_gap": signed_logit_gap,
        "predicted_true": predicted_true,
        "correct_by_layer": correct_by_layer,
        "first_wrong_layer": first_wrong,
        "stable_final_prediction_layer": stable_pred,
        "stable_correctness_layer": stable_correctness,
        "final_predicted_true": bool(predicted_true[-1]),
        "final_correct": bool(correct_by_layer[-1]),
        "final_signed_gap": float(signed_logit_gap[-1]),
    }


def main() -> dict[str, Any]:
    """
    Run logit-lens extraction on the 200 selected complementary pairs.

    Returns:
        Summary dictionary for Modal / CLI usage.
    """
    logger.info("=" * 60)
    logger.info("Logit Lens Extraction")
    logger.info("=" * 60)

    code_dir = Path(__file__).parent
    results_dir = Path(RESULTS_DIR)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 1) Ensure activations exist
    logger.info("[1/5] Ensuring residual activations exist...")
    act_path = ensure_activations_exist()

    # 2) Load activations + pair metadata
    logger.info("[2/5] Loading saved activations and selected pairs...")
    activations: dict[str, torch.Tensor] = torch.load(act_path, weights_only=False)

    selected_pairs_path = code_dir / "eval" / "selected_pairs_for_patching.json"
    with open(selected_pairs_path) as f:
        pairs = json.load(f)

    first_tensor = next(iter(activations.values()))
    n_layers, d_model = first_tensor.shape
    logger.info(
        f"Loaded {len(activations)} examples | n_layers={n_layers} | d_model={d_model} | "
        f"{len(pairs)} selected pairs"
    )

    # 3) Load model + token ids
    logger.info("[3/5] Loading model for ln_final + unembed...")
    model, model_name = load_model()
    true_token_id, false_token_id = verify_tokens(model)
    model.eval()
    logger.info(
        f"Model: {model_name} | True token id={true_token_id} | False token id={false_token_id}"
    )

    # 4) Apply logit lens pair-by-pair
    logger.info("[4/5] Applying logit lens to cached residual streams...")
    pair_results: list[dict[str, Any]] = []
    skipped_pairs = 0

    for pair in tqdm(pairs, desc="Logit lens pairs"):
        correct_id = pair["correct_example_id"]
        incorrect_id = pair["incorrect_example_id"]

        if correct_id not in activations or incorrect_id not in activations:
            skipped_pairs += 1
            logger.warning(
                f"Skipping pair {pair['pair_id']} — missing activations for "
                f"{correct_id if correct_id not in activations else incorrect_id}"
            )
            continue

        correct_trace = apply_logit_lens_to_residuals(
            model=model,
            resid_by_layer=activations[correct_id],
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            ground_truth_label=bool(pair["label_correct"]),
        )

        incorrect_trace = apply_logit_lens_to_residuals(
            model=model,
            resid_by_layer=activations[incorrect_id],
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            ground_truth_label=bool(pair["label_incorrect"]),
        )

        pair_results.append(
            {
                "pair_id": pair["pair_id"],
                "domain": pair["domain"],
                "scenario": pair["scenario"],
                "incorrect_confidence": float(pair["incorrect_confidence"]),
                "phase1_logit_gap": float(pair["logit_gap"]),
                "failed_on_true": bool(pair["failed_on_true"]),
                "correct_example_id": correct_id,
                "incorrect_example_id": incorrect_id,
                "correct_sentence": pair["sentence_correct"],
                "incorrect_sentence": pair["sentence_incorrect"],
                "correct": correct_trace,
                "incorrect": incorrect_trace,
            }
        )

    # 5) Save results
    logger.info("[5/5] Saving logit-lens results...")
    results = {
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs_requested": len(pairs),
        "n_pairs_processed": len(pair_results),
        "n_pairs_skipped": skipped_pairs,
        "true_token_id": true_token_id,
        "false_token_id": false_token_id,
        "activation_source": str(act_path),
        "analysis_target": "selected asymmetric pairs for patching",
        "pair_results": pair_results,
    }

    out_path = analysis_dir / "logit_lens_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Lightweight summary for logs / Modal
    incorrect_first_wrong = [
        p["incorrect"]["first_wrong_layer"]
        for p in pair_results
        if p["incorrect"]["first_wrong_layer"] is not None
    ]
    incorrect_stable_pred = [
        p["incorrect"]["stable_final_prediction_layer"] for p in pair_results
    ]
    mean_final_signed_gap_correct = (
        sum(p["correct"]["final_signed_gap"] for p in pair_results) / len(pair_results)
        if pair_results
        else 0.0
    )
    mean_final_signed_gap_incorrect = (
        sum(p["incorrect"]["final_signed_gap"] for p in pair_results)
        / len(pair_results)
        if pair_results
        else 0.0
    )

    logger.info("\n" + "=" * 60)
    logger.info("Logit Lens Summary")
    logger.info("=" * 60)
    logger.info(f"Processed pairs: {len(pair_results)} / {len(pairs)}")
    logger.info(f"Saved to: {out_path}")
    logger.info(
        f"Mean final signed gap (correct examples):   {mean_final_signed_gap_correct:+.4f}"
    )
    logger.info(
        f"Mean final signed gap (incorrect examples): {mean_final_signed_gap_incorrect:+.4f}"
    )

    if incorrect_first_wrong:
        logger.info(
            f"Mean first-wrong layer (incorrect examples): "
            f"{sum(incorrect_first_wrong) / len(incorrect_first_wrong):.2f}"
        )
    if incorrect_stable_pred:
        logger.info(
            f"Mean stable-final-prediction layer (incorrect examples): "
            f"{sum(incorrect_stable_pred) / len(incorrect_stable_pred):.2f}"
        )

    return {
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs_processed": len(pair_results),
        "n_pairs_skipped": skipped_pairs,
        "output_path": str(out_path),
        "mean_final_signed_gap_correct": mean_final_signed_gap_correct,
        "mean_final_signed_gap_incorrect": mean_final_signed_gap_incorrect,
    }


if __name__ == "__main__":
    main()
