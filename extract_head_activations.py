"""
Phase 3b: Head-level activation extraction.

Extracts per-head attention output (z) at the final token position for layers
18-35 (where probing showed signal). Each head produces a [d_head]-dimensional
vector, much lower dimensional than the full residual stream, addressing the
TA's overfitting concern.

Also extracts activations at intermediate token positions (last token of the
actual statement, before the prompt suffix) to test whether signal is stronger
earlier in the sequence.

Output:
  head_activations.pt        — dict {example_id -> dict of tensors}
  head_activations_meta.json — metadata
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

# Only extract from layers where signal exists (save memory)
EXTRACT_LAYERS = list(range(18, 36))


def find_statement_end_position(model, prompt: str, sentence: str) -> int:
    """
    Find the token position of the last token of the actual statement
    within the formatted prompt.

    Returns the index of the last statement token, or -1 as fallback (final token).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]
    sentence_tokens = model.to_tokens(sentence, prepend_bos=False)[0]

    # Search for the sentence tokens within the full prompt tokens
    n_sent = len(sentence_tokens)
    n_prompt = len(tokens)

    for start in range(n_prompt - n_sent, -1, -1):
        if torch.equal(tokens[start:start + n_sent], sentence_tokens):
            pos = start + n_sent - 1  # last token of sentence
            return min(pos, n_prompt - 1)  # clamp to valid range

    return n_prompt - 1  # fallback to final token


def extract_head_activations(model, sentence: str) -> dict:
    """
    Extract per-head z vectors at final token and statement-end token.

    Returns dict with:
      'z_final': tensor [n_extract_layers, n_heads, d_head] at final token
      'z_stmt_end': tensor [n_extract_layers, n_heads, d_head] at statement end
      'resid_final': tensor [n_extract_layers, d_model] at final token (for comparison)
      'stmt_end_pos': int position of statement end token
    """
    prompt = format_prompt(sentence)

    # Build names filter for z (attention output before projection) and resid_post
    def names_filter(name: str) -> bool:
        if name.endswith("hook_resid_post"):
            # Check if this layer is in our extract range
            parts = name.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) in EXTRACT_LAYERS
        if name.endswith("hook_z"):
            parts = name.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) in EXTRACT_LAYERS
        return False

    with torch.no_grad():
        _, cache = model.run_with_cache(prompt, names_filter=names_filter)

    seq_len = cache["resid_post", EXTRACT_LAYERS[0]].shape[1]
    final_pos = seq_len - 1

    # Find statement end position, clamped to valid range
    stmt_end_pos = find_statement_end_position(model, prompt, sentence)
    stmt_end_pos = min(stmt_end_pos, final_pos)

    # Extract z at final token: hook_z shape is [batch, seq_len, n_heads, d_head]
    z_final = torch.stack([
        cache["z", L][0, final_pos, :, :].cpu()  # [n_heads, d_head]
        for L in EXTRACT_LAYERS
    ])  # [n_extract_layers, n_heads, d_head]

    # Extract z at statement end position
    z_stmt_end = torch.stack([
        cache["z", L][0, stmt_end_pos, :, :].cpu()
        for L in EXTRACT_LAYERS
    ])  # [n_extract_layers, n_heads, d_head]

    # Also extract resid_post at both positions
    resid_final = torch.stack([
        cache["resid_post", L][0, final_pos, :].cpu()
        for L in EXTRACT_LAYERS
    ])  # [n_extract_layers, d_model]

    resid_stmt_end = torch.stack([
        cache["resid_post", L][0, stmt_end_pos, :].cpu()
        for L in EXTRACT_LAYERS
    ])  # [n_extract_layers, d_model]

    return {
        "z_final": z_final,
        "z_stmt_end": z_stmt_end,
        "resid_final": resid_final,
        "resid_stmt_end": resid_stmt_end,
        "stmt_end_pos": stmt_end_pos,
        "final_pos": final_pos,
    }


def main() -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Phase 3b: Head-Level Activation Extraction")
    logger.info("=" * 60)

    code_dir = Path(__file__).parent
    out_dir = Path(RESULTS_DIR) / "head_activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load selected pairs
    selected_path = code_dir / "eval" / "selected_pairs_for_patching.json"
    with open(selected_path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} pairs -> {len(pairs) * 2} forward passes")

    # Load model
    model, model_name = load_model()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_model = model.cfg.d_model
    logger.info(f"Model: {model_name} | {n_layers} layers | {n_heads} heads | d_head={d_head}")

    # Extract
    activations = {}
    positions = {}

    for pair in tqdm(pairs, desc="Extracting head activations"):
        for role in ("correct", "incorrect"):
            ex_id = pair[f"{role}_example_id"]
            sentence = pair[f"sentence_{role}"]
            result = extract_head_activations(model, sentence)

            activations[ex_id] = {
                "z_final": result["z_final"],
                "z_stmt_end": result["z_stmt_end"],
                "resid_final": result["resid_final"],
                "resid_stmt_end": result["resid_stmt_end"],
            }
            positions[ex_id] = {
                "stmt_end_pos": result["stmt_end_pos"],
                "final_pos": result["final_pos"],
            }

    # Save
    act_path = out_dir / "head_activations.pt"
    torch.save(activations, act_path)
    size_mb = act_path.stat().st_size / 1e6
    logger.info(f"\nSaved {len(activations)} examples to {act_path} ({size_mb:.1f} MB)")

    meta = {
        "model_name": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
        "extract_layers": EXTRACT_LAYERS,
        "n_extract_layers": len(EXTRACT_LAYERS),
        "n_pairs": len(pairs),
        "n_examples": len(activations),
        "positions": positions,
        "tensors_per_example": {
            "z_final": f"[{len(EXTRACT_LAYERS)}, {n_heads}, {d_head}]",
            "z_stmt_end": f"[{len(EXTRACT_LAYERS)}, {n_heads}, {d_head}]",
            "resid_final": f"[{len(EXTRACT_LAYERS)}, {d_model}]",
            "resid_stmt_end": f"[{len(EXTRACT_LAYERS)}, {d_model}]",
        },
    }
    meta_path = out_dir / "head_activations_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved metadata to {meta_path}")
    logger.info(f"\nPer-example shapes:")
    for k, v in meta["tensors_per_example"].items():
        logger.info(f"  {k}: {v}")

    return {
        "model_name": model_name,
        "n_examples": len(activations),
        "output_mb": size_mb,
        "extract_layers": EXTRACT_LAYERS,
    }


if __name__ == "__main__":
    main()
