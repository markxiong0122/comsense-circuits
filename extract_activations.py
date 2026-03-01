"""
Phase 2: Residual stream activation extraction for selected asymmetric pairs.

For each of the 200 selected pairs, runs both the correct and incorrect sentence
through model.run_with_cache() and saves the residual stream at the final token
position across all layers.

Output (written to RESULTS_DIR/activations/):
  activations.pt        — dict {example_id -> tensor[n_layers, d_model]}
  activations_meta.json — pair metadata, model config, run info
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from config import RESULTS_DIR
from data_utils import format_prompt
from evaluate import load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_final_resid(model, sentence: str) -> torch.Tensor:
    """
    Run a single sentence through the model and return the residual stream
    at the final token position for every layer.

    Uses names_filter to cache only resid_post hooks, avoiding the memory
    cost of storing attention patterns and MLP intermediates.

    Args:
        model: Loaded HookedTransformer
        sentence: Raw sentence string (formatted into the eval prompt)

    Returns:
        Tensor of shape [n_layers, d_model] on CPU
    """
    prompt = format_prompt(sentence)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt,
            names_filter=lambda name: name.endswith("hook_resid_post"),
        )

    resid = torch.stack(
        [cache["resid_post", L][0, -1, :].cpu() for L in range(model.cfg.n_layers)]
    )  # [n_layers, d_model]

    return resid


def main() -> dict[str, Any]:
    """
    Load model and selected pairs, run extraction, save results.

    Returns:
        Summary dict for Modal.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Activation Extraction")
    logger.info("=" * 60)

    # selected_pairs_for_patching.json is a local input file bundled with the
    # code (via add_local_dir), not a Modal volume output — so it lives under
    # the code root, not RESULTS_DIR.
    code_dir = Path(__file__).parent
    out_dir = Path(RESULTS_DIR) / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load selected pairs
    selected_path = code_dir / "eval" / "selected_pairs_for_patching.json"
    logger.info(f"\n[1/3] Loading selected pairs from {selected_path}")
    with open(selected_path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} pairs → {len(pairs) * 2} forward passes")

    # Load model
    logger.info("\n[2/3] Loading model...")
    model, model_name = load_model()
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    logger.info(f"Model: {model_name} | n_layers={n_layers} | d_model={d_model}")

    # Extract activations
    logger.info("\n[3/3] Extracting activations...")
    activations: dict[str, torch.Tensor] = {}

    for pair in tqdm(pairs, desc="Pairs"):
        for role in ("correct", "incorrect"):
            ex_id = pair[f"{role}_example_id"]
            sentence = pair[f"sentence_{role}"]
            activations[ex_id] = extract_final_resid(model, sentence)

    # Save activations
    act_path = out_dir / "activations.pt"
    torch.save(activations, act_path)
    size_mb = act_path.stat().st_size / 1e6
    logger.info(f"\nSaved {len(activations)} tensors to {act_path} ({size_mb:.1f} MB)")
    logger.info(f"  Per-example shape: [{n_layers}, {d_model}]")

    # Save metadata
    meta = {
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(pairs),
        "n_examples": len(activations),
        "activation_key": "resid_post",
        "token_position": "final",
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "correct_example_id": p["correct_example_id"],
                "incorrect_example_id": p["incorrect_example_id"],
                "domain": p["domain"],
                "scenario": p["scenario"],
                "failed_on_true": p["failed_on_true"],
                "incorrect_confidence": p["incorrect_confidence"],
                "logit_gap": p["logit_gap"],
            }
            for p in pairs
        ],
    }
    meta_path = out_dir / "activations_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Extraction Complete")
    logger.info(f"  activations.pt        — {len(activations)} tensors [{n_layers}, {d_model}]")
    logger.info(f"  activations_meta.json — pair + run metadata")
    logger.info("=" * 60)

    return {
        "model_name": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(pairs),
        "n_examples": len(activations),
        "output_mb": size_mb,
    }


if __name__ == "__main__":
    main()
