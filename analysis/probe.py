"""
Phase 2: Linear probing of residual stream activations.

Runs three probes across all 36 layers:
  1. Correctness probe       — predict correct vs. incorrect (model behavior)
  2. Correctness probe + PCA — same, after reducing to 50 PCA components
  3. Ground truth probe + PCA — predict True vs. False ground truth label

Cross-validation is grouped by pair so that both members of a complementary
pair always land in the same fold — preventing data leakage from near-duplicate
sentences appearing in both train and test.

Outputs (written to analysis/):
  probe_results.json       — all per-layer accuracies + per-domain breakdown
  probe_accuracy_curve.png — all three probes plotted together
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def load_data(repo_dir: Path) -> tuple[dict, list]:
    # Try local activations_dir first, then Modal volume path
    act_candidates = [
        repo_dir / "activations",
        repo_dir / "activations_dir" / "activations",
        repo_dir / "results" / "activations",
    ]

    # Also look for .pt files inside activation directories
    acts = None
    for p in act_candidates:
        if p.is_dir():
            pt_file = p / "activations.pt"
            if pt_file.exists():
                acts = torch.load(pt_file, weights_only=False)
                break
            # Try torch directory-format load
            try:
                acts = torch.load(p, weights_only=False)
                break
            except (IsADirectoryError, Exception):
                continue
        elif p.exists():
            acts = torch.load(p, weights_only=False)
            break
    if acts is None:
        raise FileNotFoundError(f"No activations found in: {act_candidates}")

    with open(repo_dir / "eval" / "selected_pairs_for_patching.json") as f:
        pairs = json.load(f)
    return acts, pairs


def build_arrays(
    acts: dict,
    pairs: list,
    layer: int,
    label_type: str = "correctness",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build X, y, groups arrays for a single layer.

    label_type:
      "correctness"  — y=1 if model was correct, y=0 if incorrect
      "ground_truth" — y=1 if statement is True, y=0 if False
    """
    X_rows, y_rows, group_rows, meta_rows = [], [], [], []

    for pair_idx, pair in enumerate(pairs):
        for role, correctness_label in (("correct", 1), ("incorrect", 0)):
            ex_id = pair[f"{role}_example_id"]
            if ex_id not in acts:
                continue

            if label_type == "correctness":
                y_label = correctness_label
            else:
                y_label = int(pair[f"label_{role}"])  # True=1, False=0

            X_rows.append(acts[ex_id][layer].float().numpy())
            y_rows.append(y_label)
            group_rows.append(pair_idx)
            meta_rows.append({"domain": pair["domain"], "scenario": pair["scenario"]})

    return (
        np.array(X_rows),
        np.array(y_rows),
        np.array(group_rows),
        meta_rows,
    )


def probe_layer(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_pca_components: int | None = None,
) -> float:
    """
    GroupKFold cross-validation with logistic regression.

    StandardScaler and optional PCA are fit inside each fold to avoid leakage.
    """
    cv = GroupKFold(n_splits=n_splits)
    fold_scores = []

    for train_idx, test_idx in cv.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if n_pca_components is not None:
            n_components = min(n_pca_components, X_tr.shape[0], X_tr.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        fold_scores.append(clf.score(X_te, y_te))

    return float(np.mean(fold_scores))


def probe_domain_breakdown(
    acts: dict,
    pairs: list,
    layer: int,
    label_type: str,
    n_pca_components: int | None,
    n_splits: int = 5,
) -> dict[str, float]:
    """Per-domain probe accuracy at a single layer."""
    domain_pairs = defaultdict(list)
    for pair in pairs:
        domain_pairs[pair["domain"]].append(pair)

    results = {}
    for domain, dpairs in domain_pairs.items():
        if len(dpairs) < n_splits * 2:
            continue
        X, y, groups, _ = build_arrays(acts, dpairs, layer, label_type)
        results[domain] = probe_layer(X, y, groups, n_splits, n_pca_components)
    return results


def run_probe(
    acts: dict,
    pairs: list,
    n_layers: int,
    label_type: str,
    n_pca_components: int | None,
    label: str,
) -> list[float]:
    print(f"\n  [{label}]")
    accuracies = []
    for layer in range(n_layers):
        X, y, groups, _ = build_arrays(acts, pairs, layer, label_type)
        acc = probe_layer(X, y, groups, n_pca_components=n_pca_components)
        accuracies.append(acc)
        print(f"    Layer {layer:2d}: {acc:.4f}")
    best = int(np.argmax(accuracies))
    print(f"    Best layer: {best} ({accuracies[best]:.4f})")
    return accuracies


def main():
    repo_dir = Path(__file__).parent.parent
    analysis_dir = Path(__file__).parent

    print("Loading activations and pair metadata...")
    acts, pairs = load_data(repo_dir)
    first_tensor = next(iter(acts.values()))
    n_layers, d_model = first_tensor.shape
    print(f"  {len(acts)} examples | {n_layers} layers | d_model={d_model} | {len(pairs)} pairs")

    probes = [
        ("correctness",  None, "Correctness (no PCA)"),
        ("correctness",  50,   "Correctness + PCA-50"),
        ("ground_truth", 50,   "Ground truth + PCA-50"),
    ]

    all_results = {}
    print("\nRunning probes across all layers...")
    for label_type, n_pca, label in probes:
        accs = run_probe(acts, pairs, n_layers, label_type, n_pca, label)
        all_results[label] = accs

    # Per-domain breakdown at best layer for the ground truth + PCA probe
    gt_accs = all_results["Ground truth + PCA-50"]
    best_layer = int(np.argmax(gt_accs))
    print(f"\nPer-domain breakdown at layer {best_layer} (ground truth + PCA-50)...")
    domain_accs = probe_domain_breakdown(acts, pairs, best_layer, "ground_truth", 50)
    for domain, acc in sorted(domain_accs.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {acc:.4f}")

    # Save results
    results = {
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(pairs),
        "n_examples": len(acts),
        "best_layer_ground_truth_pca": best_layer,
        "best_accuracy_ground_truth_pca": gt_accs[best_layer],
        "per_layer": {
            label: accs for label, accs in
            zip([p[2] for p in probes], all_results.values())
        },
        "domain_accuracies_at_best_layer": domain_accs,
    }

    results_path = analysis_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plot
    layers = list(range(n_layers))
    colors = ["#94a3b8", "#2563eb", "#16a34a"]
    styles = ["--", "-", "-"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for (_, _, label), color, style in zip(probes, colors, styles):
        accs = all_results[label]
        ax.plot(layers, accs, marker="o", markersize=3.5, linewidth=1.5,
                color=color, linestyle=style, label=label)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (50%)")
    ax.axvline(best_layer, color="#dc2626", linestyle=":", linewidth=1.2,
               label=f"Best layer {best_layer} ({gt_accs[best_layer]:.2%})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title("Linear Probe Accuracy on Residual Stream — Qwen3-8B on Com2Sense\n"
                 "Correct vs. Incorrect | Ground Truth — Final Token Position")
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    plot_path = analysis_dir / "probe_accuracy_curve.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close(fig)

    print("\n=== Probing Complete ===")


if __name__ == "__main__":
    main()
