"""
L2 regularization sweep on existing linear probes.

Addresses TA feedback: 4096-dim activations on 200 examples may overfit.
Sweeps C values (inverse regularization strength) to find optimal regularization.

Runs locally on saved activations — no GPU needed.
"""

import json
from pathlib import Path

import numpy as np
import torch
from probe import build_arrays, load_data
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def probe_layer_with_C(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float,
    n_splits: int = 5,
    n_pca_components: int | None = None,
) -> float:
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

        clf = LogisticRegression(C=C, max_iter=2000, random_state=42, penalty="l2")
        clf.fit(X_tr, y_tr)
        fold_scores.append(clf.score(X_te, y_te))

    return float(np.mean(fold_scores))


def main():
    repo_dir = Path(__file__).parent.parent
    output_dir = repo_dir / "artifacts" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading activations...")
    acts, pairs = load_data(repo_dir)
    first_tensor = next(iter(acts.values()))
    n_layers = first_tensor.shape[0]

    C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    # Focus on layers 20-35 where signal exists, plus a few early layers as baseline
    probe_layers = list(range(0, 5)) + list(range(18, n_layers))

    configs = [
        ("correctness", None, "Correctness (no PCA)"),
        ("correctness", 50, "Correctness + PCA-50"),
        ("ground_truth", 50, "Ground truth + PCA-50"),
    ]

    results = {}

    for label_type, n_pca, label in configs:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

        best_overall = {"C": None, "layer": None, "acc": 0.0}
        c_results = {}

        for C in C_values:
            layer_accs = {}
            for layer in probe_layers:
                X, y, groups, _ = build_arrays(acts, pairs, layer, label_type)
                acc = probe_layer_with_C(X, y, groups, C, n_pca_components=n_pca)
                layer_accs[layer] = acc

            best_layer = max(layer_accs, key=layer_accs.get)
            best_acc = layer_accs[best_layer]
            c_results[str(C)] = {"best_layer": best_layer, "best_acc": best_acc}

            if best_acc > best_overall["acc"]:
                best_overall = {"C": C, "layer": best_layer, "acc": best_acc}

            print(f"  C={C:<8} best layer={best_layer:2d}  acc={best_acc:.4f}")

        results[label] = {
            "by_C": c_results,
            "best": best_overall,
        }
        print(
            f"  >>> Best: C={best_overall['C']} layer={best_overall['layer']} acc={best_overall['acc']:.4f}"
        )

    # Save
    out_path = output_dir / "l2_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
