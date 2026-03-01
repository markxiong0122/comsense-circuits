"""
Complementary Pair Deep Analysis

Analyzes asymmetric pairs to:
1. Quantify failure direction (failed on TRUE vs FALSE)
2. Correlate with domain
3. Select ~200 high-confidence failures for Phase 2 patching
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Any


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def main():
    # Paths
    eval_dir = Path(__file__).parent.parent / "eval"
    analysis_dir = Path(__file__).parent

    # Load data
    print("Loading data...")
    predictions_list = load_jsonl(eval_dir / "predictions.jsonl")
    asymmetric_pairs = load_json(eval_dir / "asymmetric_pairs.json")
    accuracy_table = load_json(eval_dir / "accuracy_table.json")

    # Create prediction lookup
    predictions = {p["example_id"]: p for p in predictions_list}

    print(f"Loaded {len(predictions)} predictions")
    print(f"Loaded {len(asymmetric_pairs)} asymmetric pairs")

    # Step 2: Quantify failure direction
    print("\n=== Step 2: Quantifying Failure Direction ===")

    failed_on_true = 0
    failed_on_false = 0

    for pair in asymmetric_pairs:
        correct_id = pair["correct_example_id"]
        incorrect_id = pair["incorrect_example_id"]

        # Get the label of the incorrect example (what the model got wrong)
        incorrect_label = pair["label_a"] if incorrect_id == pair["example_a_id"] else pair["label_b"]

        if incorrect_label:  # Model got the TRUE statement wrong
            failed_on_true += 1
        else:  # Model got the FALSE statement wrong
            failed_on_false += 1

    total_asymmetric = len(asymmetric_pairs)
    print(f"\nFailure Direction Summary:")
    print(f"  Failed on TRUE:  {failed_on_true:4d} ({100*failed_on_true/total_asymmetric:.1f}%)")
    print(f"  Failed on FALSE: {failed_on_false:4d} ({100*failed_on_false/total_asymmetric:.1f}%)")

    # Step 3: Correlate with domain
    print("\n=== Step 3: Failure Direction by Domain ===")

    # Track failures by domain
    domain_stats = defaultdict(lambda: {"failed_on_true": 0, "failed_on_false": 0, "total": 0})

    for pair in asymmetric_pairs:
        incorrect_id = pair["incorrect_example_id"]

        # Get domain from predictions
        if incorrect_id not in predictions:
            print(f"Warning: {incorrect_id} not found in predictions")
            continue

        pred = predictions[incorrect_id]
        domain = pred["domain"]

        # Get the label of the incorrect example
        incorrect_label = pair["label_a"] if incorrect_id == pair["example_a_id"] else pair["label_b"]

        domain_stats[domain]["total"] += 1
        if incorrect_label:  # Failed on TRUE
            domain_stats[domain]["failed_on_true"] += 1
        else:  # Failed on FALSE
            domain_stats[domain]["failed_on_false"] += 1

    print("\nBy Domain:")
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        total = stats["total"]
        if total > 0:
            pct_true = 100 * stats["failed_on_true"] / total
            pct_false = 100 * stats["failed_on_false"] / total
            print(f"  {domain}:")
            print(f"    failed_on_true:  {stats['failed_on_true']:3d} ({pct_true:.1f}%)")
            print(f"    failed_on_false: {stats['failed_on_false']:3d} ({pct_false:.1f}%)")

    # Also track by domain_scenario combination
    print("\n=== Failure Direction by Domain+Scenario ===")

    domain_scenario_stats = defaultdict(lambda: {"failed_on_true": 0, "failed_on_false": 0, "total": 0})

    for pair in asymmetric_pairs:
        incorrect_id = pair["incorrect_example_id"]

        if incorrect_id not in predictions:
            continue

        pred = predictions[incorrect_id]
        domain = pred["domain"]
        scenario = pred["scenario"]
        key = f"{domain}_{scenario}"

        incorrect_label = pair["label_a"] if incorrect_id == pair["example_a_id"] else pair["label_b"]

        domain_scenario_stats[key]["total"] += 1
        if incorrect_label:
            domain_scenario_stats[key]["failed_on_true"] += 1
        else:
            domain_scenario_stats[key]["failed_on_false"] += 1

    # Sort by total (descending) and show
    print("\nBy Domain_Scenario (sorted by count):")
    for key in sorted(domain_scenario_stats.keys(), key=lambda k: domain_scenario_stats[k]["total"], reverse=True):
        stats = domain_scenario_stats[key]
        total = stats["total"]
        if total > 0:
            pct_true = 100 * stats["failed_on_true"] / total
            pct_false = 100 * stats["failed_on_false"] / total
            print(f"  {key}:")
            print(f"    failed_on_true:  {stats['failed_on_true']:3d} ({pct_true:.1f}%)")
            print(f"    failed_on_false: {stats['failed_on_false']:3d} ({pct_false:.1f}%)")

    # Step 4: Select ~200 high-confidence failures
    print("\n=== Step 4: Selecting High-Confidence Failures ===")

    # Enrich pairs with confidence and domain info
    enriched_pairs = []
    for pair in asymmetric_pairs:
        incorrect_id = pair["incorrect_example_id"]
        correct_id = pair["correct_example_id"]

        if incorrect_id not in predictions or correct_id not in predictions:
            continue

        incorrect_pred = predictions[incorrect_id]
        correct_pred = predictions[correct_id]

        # Get the label of the incorrect example
        incorrect_label = pair["label_a"] if incorrect_id == pair["example_a_id"] else pair["label_b"]

        # Calculate logit gap (confidence signal strength)
        logit_gap = abs(incorrect_pred["true_logit"] - incorrect_pred["false_logit"])

        enriched_pair = {
            **pair,
            "domain": incorrect_pred["domain"],
            "scenario": incorrect_pred["scenario"],
            "domain_scenario": f"{incorrect_pred['domain']}_{incorrect_pred['scenario']}",
            "incorrect_confidence": incorrect_pred["confidence"],
            "correct_confidence": correct_pred["confidence"],
            "logit_gap": logit_gap,
            "failed_on_true": incorrect_label,  # True if model failed on TRUE statement
            "failed_on_false": not incorrect_label,
        }
        enriched_pairs.append(enriched_pair)

    # Filter criteria:
    # 1. From lowest-performing domains (temporal: 63.5%, time: 66.6%)
    # 2. High model confidence on WRONG answer (confidence > 0.7)
    # 3. Large logit gap (strong but wrong signal)

    LOW_ACCURACY_DOMAINS = ["temporal", "time"]
    MIN_CONFIDENCE = 0.7
    MIN_LOGIT_GAP = 1.0

    # First pass: high confidence failures from low-accuracy domains
    high_conf_from_low_acc = [
        p for p in enriched_pairs
        if p["domain"] in LOW_ACCURACY_DOMAINS
        and p["incorrect_confidence"] >= MIN_CONFIDENCE
        and p["logit_gap"] >= MIN_LOGIT_GAP
    ]

    print(f"High-confidence failures from temporal/time domains: {len(high_conf_from_low_acc)}")

    # Second pass: if we need more, add from social domain with high confidence
    social_high_conf = [
        p for p in enriched_pairs
        if p["domain"] == "social"
        and p["incorrect_confidence"] >= MIN_CONFIDENCE
        and p["logit_gap"] >= MIN_LOGIT_GAP
    ]

    print(f"High-confidence failures from social domain: {len(social_high_conf)}")

    # Third pass: physical domain high confidence
    physical_high_conf = [
        p for p in enriched_pairs
        if p["domain"] == "physical"
        and p["incorrect_confidence"] >= MIN_CONFIDENCE
        and p["logit_gap"] >= MIN_LOGIT_GAP
    ]

    print(f"High-confidence failures from physical domain: {len(physical_high_conf)}")

    # Combine and prioritize by confidence
    selected = high_conf_from_low_acc.copy()

    # Add social if needed to reach ~200
    if len(selected) < 200:
        social_sorted = sorted(social_high_conf, key=lambda p: -p["incorrect_confidence"])
        selected.extend(social_sorted[:200 - len(selected)])

    # Add physical if still needed
    if len(selected) < 200:
        physical_sorted = sorted(physical_high_conf, key=lambda p: -p["incorrect_confidence"])
        selected.extend(physical_sorted[:200 - len(selected)])

    # Sort final selection by confidence (descending)
    selected = sorted(selected, key=lambda p: -p["incorrect_confidence"])

    print(f"\nSelected {len(selected)} pairs for Phase 2 patching")

    # Stats on selected pairs
    selected_domains = defaultdict(int)
    selected_failed_on_true = 0
    selected_failed_on_false = 0

    for p in selected:
        selected_domains[p["domain"]] += 1
        if p["failed_on_true"]:
            selected_failed_on_true += 1
        else:
            selected_failed_on_false += 1

    print("\nSelected pairs breakdown:")
    print(f"  By domain:")
    for domain, count in sorted(selected_domains.items(), key=lambda x: -x[1]):
        print(f"    {domain}: {count}")
    print(f"  By failure direction:")
    print(f"    failed_on_true:  {selected_failed_on_true}")
    print(f"    failed_on_false: {selected_failed_on_false}")
    print(f"  Avg confidence on wrong answer: {sum(p['incorrect_confidence'] for p in selected)/len(selected):.3f}")
    print(f"  Avg logit gap: {sum(p['logit_gap'] for p in selected)/len(selected):.3f}")

    # Write outputs
    print("\n=== Writing Output Files ===")

    # 1. Failure direction summary (markdown)
    md_lines = [
        "# Failure Direction Analysis",
        "",
        "## Overall Failure Direction",
        "",
        f"| Direction | Count | Percentage |",
        f"|-----------|-------|------------|",
        f"| Failed on TRUE  | {failed_on_true} | {100*failed_on_true/total_asymmetric:.1f}% |",
        f"| Failed on FALSE | {failed_on_false} | {100*failed_on_false/total_asymmetric:.1f}% |",
        f"| **Total** | {total_asymmetric} | 100% |",
        "",
        "## Failure Direction by Domain",
        "",
        f"| Domain | Failed on TRUE | Failed on FALSE | Total | % TRUE |",
        f"|--------|----------------|-----------------|-------|--------|",
    ]

    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        total = stats["total"]
        if total > 0:
            pct_true = 100 * stats["failed_on_true"] / total
            md_lines.append(
                f"| {domain} | {stats['failed_on_true']} | {stats['failed_on_false']} | {total} | {pct_true:.1f}% |"
            )

    md_lines.extend([
        "",
        "## Failure Direction by Domain+Scenario",
        "",
        f"| Domain_Scenario | Failed on TRUE | Failed on FALSE | Total | % TRUE |",
        f"|-----------------|----------------|-----------------|-------|--------|",
    ])

    for key in sorted(domain_scenario_stats.keys(), key=lambda k: domain_scenario_stats[k]["total"], reverse=True):
        stats = domain_scenario_stats[key]
        total = stats["total"]
        if total > 0:
            pct_true = 100 * stats["failed_on_true"] / total
            md_lines.append(
                f"| {key} | {stats['failed_on_true']} | {stats['failed_on_false']} | {total} | {pct_true:.1f}% |"
            )

    md_lines.extend([
        "",
        "## Selected Pairs for Phase 2",
        "",
        f"- **Total selected:** {len(selected)}",
        f"- **Avg confidence on wrong answer:** {sum(p['incorrect_confidence'] for p in selected)/len(selected):.3f}",
        f"- **Avg logit gap:** {sum(p['logit_gap'] for p in selected)/len(selected):.3f}",
        "",
        "### By Domain:",
        "",
    ])

    for domain, count in sorted(selected_domains.items(), key=lambda x: -x[1]):
        md_lines.append(f"- {domain}: {count}")

    md_lines.extend([
        "",
        "### By Failure Direction:",
        "",
        f"- Failed on TRUE: {selected_failed_on_true}",
        f"- Failed on FALSE: {selected_failed_on_false}",
    ])

    summary_path = analysis_dir / "failure_direction_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {summary_path}")

    # 2. Detailed breakdown (JSON)
    breakdown = {
        "overall": {
            "failed_on_true": failed_on_true,
            "failed_on_false": failed_on_false,
            "total": total_asymmetric,
        },
        "by_domain": dict(domain_stats),
        "by_domain_scenario": dict(domain_scenario_stats),
    }

    breakdown_path = analysis_dir / "failure_direction_by_domain.json"
    with open(breakdown_path, "w") as f:
        json.dump(breakdown, f, indent=2)
    print(f"Wrote {breakdown_path}")

    # 3. Selected pairs for patching (minimal format for Phase 2)
    selected_for_output = [
        {
            "pair_id": p["pair_id"],
            "correct_example_id": p["correct_example_id"],
            "incorrect_example_id": p["incorrect_example_id"],
            "sentence_correct": p["sentence_b"] if p["example_b_id"] == p["correct_example_id"] else p["sentence_a"],
            "sentence_incorrect": p["sentence_a"] if p["example_a_id"] == p["incorrect_example_id"] else p["sentence_b"],
            "label_correct": p["label_b"] if p["example_b_id"] == p["correct_example_id"] else p["label_a"],
            "label_incorrect": p["label_a"] if p["example_a_id"] == p["incorrect_example_id"] else p["label_b"],
            "domain": p["domain"],
            "scenario": p["scenario"],
            "incorrect_confidence": p["incorrect_confidence"],
            "logit_gap": p["logit_gap"],
            "failed_on_true": p["failed_on_true"],
        }
        for p in selected
    ]

    selected_path = eval_dir / "selected_pairs_for_patching.json"
    with open(selected_path, "w") as f:
        json.dump(selected_for_output, f, indent=2)
    print(f"Wrote {selected_path}")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
