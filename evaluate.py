"""
Core evaluation script for ComSense Circuits behavioral evaluation.
Loads model, runs inference on Com2Sense dataset, and saves structured results.
"""

import json
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from config import (
    DTYPE,
    FALSE_TOKEN_CANDIDATES,
    MODEL_NAME,
    MODEL_NAME_FALLBACKS,
    RESULTS_DIR,
    TRUE_TOKEN_CANDIDATES,
)
from data_utils import format_prompt, get_complementary_pairs, load_com2sense

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model() -> Tuple[HookedTransformer, str]:
    """
    Load Qwen3-8B model via TransformerLens.

    Tries fallback models if primary fails.

    Returns:
        Tuple of (model, successful_model_name)
    """
    model_names_to_try = [MODEL_NAME] + MODEL_NAME_FALLBACKS
    model = None
    successful_name = None

    for model_name in model_names_to_try:
        try:
            logger.info(f"Attempting to load model: {model_name}")
            model = HookedTransformer.from_pretrained(
                model_name,
                dtype=DTYPE,
            )
            successful_name = model_name
            logger.info(f"Successfully loaded model: {model_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue

    if model is None:
        raise ValueError(f"Could not load any model from: {model_names_to_try}")

    return model, successful_name


def verify_tokens(model: HookedTransformer) -> Tuple[int, int]:
    """
    Verify token IDs for True/False tokens.

    Tries multiple variants to find single-token representations.

    Args:
        model: Loaded HookedTransformer model

    Returns:
        Tuple of (true_token_id, false_token_id)
    """
    logger.info("Verifying True/False tokens...")

    true_token_id = None
    false_token_id = None

    # Try True token variants
    for candidate in TRUE_TOKEN_CANDIDATES:
        tokens = model.to_tokens(candidate, prepend_bos=False)
        if tokens.shape[-1] == 1:
            true_token_id = tokens[0, 0].item()
            logger.info(f"  ✓ True token: {candidate!r} -> id {true_token_id}")
            break

    if true_token_id is None:
        raise ValueError(
            f"Could not find single-token representation for True. "
            f"Tried: {TRUE_TOKEN_CANDIDATES}"
        )

    # Try False token variants
    for candidate in FALSE_TOKEN_CANDIDATES:
        tokens = model.to_tokens(candidate, prepend_bos=False)
        if tokens.shape[-1] == 1:
            false_token_id = tokens[0, 0].item()
            logger.info(f"  ✓ False token: {candidate!r} -> id {false_token_id}")
            break

    if false_token_id is None:
        raise ValueError(
            f"Could not find single-token representation for False. "
            f"Tried: {FALSE_TOKEN_CANDIDATES}"
        )

    return true_token_id, false_token_id


def evaluate_example(
    model: HookedTransformer,
    example: Dict[str, Any],
    true_token_id: int,
    false_token_id: int,
) -> Dict[str, Any]:
    """
    Evaluate a single example from the dataset.

    Runs a forward pass (NO generation), extracts logits at final position,
    and compares True vs False token logits.

    Args:
        model: Loaded HookedTransformer model
        example: Dataset example with 'sentence' and 'label' fields
        true_token_id: Token ID for True
        false_token_id: Token ID for False

    Returns:
        Dictionary with prediction results
    """
    # Format prompt
    prompt = format_prompt(example["sentence"])

    # Run forward pass (NO generation)
    with torch.no_grad():
        logits = model(prompt)  # shape: [1, seq_len, vocab_size]

    # Get logits at final position
    final_logits = logits[0, -1, :]  # shape: [vocab_size]

    # Extract True and False logits
    true_logit = final_logits[true_token_id].item()
    false_logit = final_logits[false_token_id].item()

    # Prediction
    predicted_true = true_logit > false_logit

    # Confidence (softmax over just the two tokens)
    probs = torch.softmax(torch.tensor([true_logit, false_logit]), dim=0)
    confidence = probs[0].item() if predicted_true else probs[1].item()

    # Build result dictionary
    result = {
        "example_id": example["example_id"],
        "sentence": example["sentence"],
        "label": bool(example["label"]),  # Force Python native bool
        "predicted_true": bool(predicted_true),  # Force Python native bool
        "correct": bool(predicted_true == example["label"]),  # Force Python native bool
        "true_logit": float(true_logit),  # Force Python native float
        "false_logit": float(false_logit),  # Force Python native float
        "confidence": float(confidence),  # Force Python native float
        "domain": example["domain"],
        "scenario": example["scenario"],
        "pair_id": example["pair_id"],
    }

    return result


def compute_accuracy_tables(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute accuracy statistics by domain, scenario, and cross-tab.

    Args:
        results: List of evaluation results from evaluate_example()

    Returns:
        Dictionary with accuracy tables
    """
    # Overall accuracy
    overall_acc = sum(r["correct"] for r in results) / len(results)

    # By domain
    domain_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        domain_acc[r["domain"]]["total"] += 1
        if r["correct"]:
            domain_acc[r["domain"]]["correct"] += 1

    # By scenario
    scenario_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        scenario_acc[r["scenario"]]["total"] += 1
        if r["correct"]:
            scenario_acc[r["scenario"]]["correct"] += 1

    # Cross-tab: domain × scenario
    cross_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        key = f"{r['domain']}_{r['scenario']}"
        cross_acc[key]["total"] += 1
        if r["correct"]:
            cross_acc[key]["correct"] += 1

    # Build accuracy table
    accuracy_table = {
        "overall": {
            "accuracy": overall_acc,
            "total": len(results),
            "correct": sum(r["correct"] for r in results),
        },
        "by_domain": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
            for k, v in domain_acc.items()
        },
        "by_scenario": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
            for k, v in scenario_acc.items()
        },
        "cross_tab": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
            for k, v in cross_acc.items()
        },
    }

    return accuracy_table


def classify_complementary_pairs(
    examples: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Classify complementary pairs based on model performance.

    Categories:
    - both_correct: Model got both statements in the pair correct
    - both_wrong: Model got both statements wrong
    - asymmetric: Model got one correct and one wrong (TARGET for patching)

    Args:
        examples: Original dataset examples
        results: Evaluation results

    Returns:
        Tuple of (pair_results_list, category_counts)
    """
    # Get complementary pairs
    pairs = get_complementary_pairs(examples)

    # Build lookup: example_id -> result
    result_lookup = {r["example_id"]: r for r in results}

    pair_results = []

    for ex_a, ex_b in pairs:
        res_a = result_lookup[ex_a["example_id"]]
        res_b = result_lookup[ex_b["example_id"]]

        both_correct = res_a["correct"] and res_b["correct"]
        both_wrong = not res_a["correct"] and not res_b["correct"]
        asymmetric = res_a["correct"] != res_b["correct"]

        category = (
            "both_correct"
            if both_correct
            else ("both_wrong" if both_wrong else "asymmetric")
        )

        pair_result = {
            "pair_id": ex_a["pair_id"],
            "example_a_id": ex_a["example_id"],
            "example_b_id": ex_b["example_id"],
            "sentence_a": ex_a["sentence"],
            "sentence_b": ex_b["sentence"],
            "label_a": ex_a["label"],
            "label_b": ex_b["label"],
            "a_correct": res_a["correct"],
            "b_correct": res_b["correct"],
            "category": category,
        }

        # For asymmetric pairs, record which one was correct
        if asymmetric:
            pair_result["correct_example_id"] = (
                res_a["example_id"] if res_a["correct"] else res_b["example_id"]
            )
            pair_result["incorrect_example_id"] = (
                res_b["example_id"] if res_a["correct"] else res_a["example_id"]
            )

        pair_results.append(pair_result)

    # Count categories
    categories = Counter(p["category"] for p in pair_results)

    logger.info(f"Pair categories: {dict(categories)}")
    logger.info(f"Asymmetric pairs available for patching: {categories['asymmetric']}")

    return pair_results, categories


def save_results(
    results: List[Dict[str, Any]],
    accuracy_table: Dict[str, Any],
    pair_results: List[Dict[str, Any]],
    categories: Counter,
    model_name: str,
) -> None:
    """
    Save all evaluation results to structured files.

    Creates 5 output files in results/eval/:
    1. predictions.jsonl - per-example results
    2. accuracy_table.json - aggregated accuracy stats
    3. asymmetric_pairs.json - pairs where model gets one right and one wrong
    4. pair_summary.json - all pair classifications
    5. summary.txt - human-readable overview

    Args:
        results: Per-example evaluation results
        accuracy_table: Accuracy statistics
        pair_results: Complementary pair classifications
        categories: Category counts
        model_name: Name of the model used
    """
    # Create output directory
    eval_dir = os.path.join(RESULTS_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    logger.info(f"Saving results to {eval_dir}")

    # 1. predictions.jsonl - per-example results
    predictions_path = os.path.join(eval_dir, "predictions.jsonl")
    with open(predictions_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"  ✓ Saved {len(results)} predictions to predictions.jsonl")

    # 2. accuracy_table.json - aggregated accuracy stats
    accuracy_path = os.path.join(eval_dir, "accuracy_table.json")
    with open(accuracy_path, "w") as f:
        json.dump(accuracy_table, f, indent=2)
    logger.info(f"  ✓ Saved accuracy table to accuracy_table.json")

    # 3. asymmetric_pairs.json - pairs where model gets one right and one wrong
    asymmetric = [p for p in pair_results if p["category"] == "asymmetric"]
    asymmetric_path = os.path.join(eval_dir, "asymmetric_pairs.json")
    with open(asymmetric_path, "w") as f:
        json.dump(asymmetric, f, indent=2)
    logger.info(
        f"  ✓ Saved {len(asymmetric)} asymmetric pairs to asymmetric_pairs.json"
    )

    # 4. pair_summary.json - all pair classifications
    pair_summary = {
        "categories": dict(categories),
        "total_pairs": len(pair_results),
        "pairs": pair_results,
    }
    pair_summary_path = os.path.join(eval_dir, "pair_summary.json")
    with open(pair_summary_path, "w") as f:
        json.dump(pair_summary, f, indent=2)
    logger.info(f"  ✓ Saved pair summary to pair_summary.json")

    # 5. summary.txt - human-readable overview
    summary_path = os.path.join(eval_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"ComSense Circuits — Behavioral Evaluation Summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total examples: {len(results)}\n")
        f.write(f"Overall accuracy: {accuracy_table['overall']['accuracy']:.4f} ")
        f.write(
            f"({accuracy_table['overall']['correct']}/{accuracy_table['overall']['total']})\n\n"
        )

        f.write(f"By domain:\n")
        for k, v in sorted(accuracy_table["by_domain"].items()):
            f.write(f"  {k}: {v['accuracy']:.4f} ({v['correct']}/{v['total']})\n")

        f.write(f"\nBy scenario:\n")
        for k, v in sorted(accuracy_table["by_scenario"].items()):
            f.write(f"  {k}: {v['accuracy']:.4f} ({v['correct']}/{v['total']})\n")

        f.write(f"\nComplementary pair categories:\n")
        for cat, count in sorted(categories.items()):
            f.write(f"  {cat}: {count}\n")

        f.write(
            f"\nAsymmetric pairs (target for patching): {categories.get('asymmetric', 0)}\n"
        )
        f.write(f"{'=' * 60}\n")

    logger.info(f"  ✓ Saved human-readable summary to summary.txt")
    logger.info(f"\nResults saved to {eval_dir}")


def main() -> Dict[str, Any]:
    """
    Main evaluation function.

    Loads model, runs evaluation on Com2Sense dataset, and saves results.

    Returns:
        Dictionary with evaluation summary statistics
    """
    logger.info("=" * 60)
    logger.info("ComSense Circuits - Behavioral Evaluation")
    logger.info("=" * 60)

    # Step 1: Load dataset (fail fast on dataset issues)
    logger.info("\n[1/5] Loading dataset...")
    examples = load_com2sense()
    logger.info(f"Loaded {len(examples)} examples")

    # Step 2: Load model
    logger.info("\n[2/5] Loading model...")
    model, model_name = load_model()
    logger.info(f"Model loaded: {model_name}")

    # Step 3: Verify tokens
    logger.info("\n[3/5] Verifying tokens...")
    true_token_id, false_token_id = verify_tokens(model)

    # Step 4: Run evaluation
    logger.info("\n[4/5] Running evaluation...")
    results = []
    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        result = evaluate_example(model, example, true_token_id, false_token_id)
        results.append(result)

        # Log progress every 100 examples
        if (i + 1) % 100 == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            logger.info(
                f"  Processed {i + 1}/{len(examples)}, accuracy so far: {acc_so_far:.4f}"
            )

    # Step 5: Compute statistics and save
    logger.info("\n[5/5] Computing statistics and saving results...")

    # Compute accuracy tables
    accuracy_table = compute_accuracy_tables(results)

    # Classify complementary pairs
    pair_results, categories = classify_complementary_pairs(examples, results)

    # Save all results
    save_results(results, accuracy_table, pair_results, categories, model_name)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Summary:")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Total examples: {len(results)}")
    logger.info(f"Overall accuracy: {accuracy_table['overall']['accuracy']:.4f}")
    logger.info(f"Asymmetric pairs: {categories.get('asymmetric', 0)}")
    logger.info("=" * 60)

    # Return summary for Modal
    return {
        "model_name": model_name,
        "total_examples": len(results),
        "overall_accuracy": accuracy_table["overall"]["accuracy"],
        "asymmetric_pairs": categories.get("asymmetric", 0),
        "categories": dict(categories),
    }


if __name__ == "__main__":
    main()
