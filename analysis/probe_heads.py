"""
Head-level probing on per-head z activations.

For each head in layers 18-35, trains a logistic regression probe on the
[d_head]-dimensional attention output vector. This is much lower dimensional
than the full residual stream (128 vs 4096), drastically reducing overfitting
risk with only 200 pairs.

Also probes at the statement-end token position (intermediate position probing)
to test whether signal is stronger earlier in the sequence.

Outputs:
  reports/figures/head_probe_heatmap.png     — tracked heatmap figure
  artifacts/analysis/head_probe_results.json — raw per-head accuracy results
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def load_head_data(repo_dir: Path) -> tuple:
    """Load head activations and pair metadata."""
    act_candidates = [
        repo_dir / "head_activations",
        repo_dir / "results" / "head_activations",
    ]
    acts = None
    for p in act_candidates:
        act_file = p / "head_activations.pt" if p.is_dir() else p
        if act_file.exists():
            acts = torch.load(act_file, weights_only=False)
            # Look for meta in same dir
            meta_dir = p if p.is_dir() else p.parent
            break
    if acts is None:
        raise FileNotFoundError(f"No head activations found in: {act_candidates}")

    with open(repo_dir / "eval" / "selected_pairs_for_patching.json") as f:
        pairs = json.load(f)

    meta_path = meta_dir / "head_activations_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    return acts, pairs, meta


def build_head_arrays(
    acts: dict,
    pairs: list,
    layer_idx: int,
    head_idx: int,
    position: str = "final",
    label_type: str = "correctness",
) -> tuple:
    """Build X, y, groups for a single head at a single layer."""
    key = f"z_{position}"
    X_rows, y_rows, group_rows = [], [], []

    for pair_idx, pair in enumerate(pairs):
        for role, correctness_label in (("correct", 1), ("incorrect", 0)):
            ex_id = pair[f"{role}_example_id"]
            if ex_id not in acts:
                continue

            if label_type == "correctness":
                y_label = correctness_label
            else:
                y_label = int(pair[f"label_{role}"])

            # z shape: [n_extract_layers, n_heads, d_head]
            vec = acts[ex_id][key][layer_idx, head_idx, :].float().numpy()
            X_rows.append(vec)
            y_rows.append(y_label)
            group_rows.append(pair_idx)

    return np.array(X_rows), np.array(y_rows), np.array(group_rows)


def build_resid_arrays(
    acts: dict,
    pairs: list,
    layer_idx: int,
    position: str = "final",
    label_type: str = "correctness",
) -> tuple:
    """Build X, y, groups for residual stream at a position."""
    key = f"resid_{position}"
    X_rows, y_rows, group_rows = [], [], []

    for pair_idx, pair in enumerate(pairs):
        for role, correctness_label in (("correct", 1), ("incorrect", 0)):
            ex_id = pair[f"{role}_example_id"]
            if ex_id not in acts:
                continue

            if label_type == "correctness":
                y_label = correctness_label
            else:
                y_label = int(pair[f"label_{role}"])

            vec = acts[ex_id][key][layer_idx, :].float().numpy()
            X_rows.append(vec)
            y_rows.append(y_label)
            group_rows.append(pair_idx)

    return np.array(X_rows), np.array(y_rows), np.array(group_rows)


def probe_cv(X, y, groups, C=0.1, n_splits=5, n_pca=None):
    """GroupKFold cross-validated accuracy."""
    from sklearn.decomposition import PCA

    cv = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if n_pca is not None:
            n_components = min(n_pca, X_tr.shape[0], X_tr.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = LogisticRegression(C=C, max_iter=2000, random_state=42, penalty="l2")
        clf.fit(X_tr, y_tr)
        scores.append(clf.score(X_te, y_te))
    return float(np.mean(scores))


def main(output_dir: Path | None = None):
    repo_dir = Path(__file__).parent.parent
    base_dir = output_dir or repo_dir
    artifacts_dir = base_dir / "artifacts" / "analysis"
    figures_dir = base_dir / "reports" / "figures"

    print("Loading head activations...")
    acts, pairs, meta = load_head_data(repo_dir)
    extract_layers = meta["extract_layers"]
    n_heads = meta["n_heads"]
    d_head = meta["d_head"]
    print(
        f"  {len(acts)} examples | layers {extract_layers[0]}-{extract_layers[-1]} | {n_heads} heads | d_head={d_head}"
    )

    results = {}

    # 1. Head-level probing at final token position
    for label_type, label_name in [
        ("correctness", "correctness"),
        ("ground_truth", "ground_truth"),
    ]:
        for position in ["final", "stmt_end"]:
            config_name = f"head_{label_name}_{position}"
            print(f"\n{'=' * 60}")
            print(f"  {config_name}")
            print(f"{'=' * 60}")

            head_accs = np.zeros((len(extract_layers), n_heads))

            for li, layer in enumerate(extract_layers):
                for hi in range(n_heads):
                    X, y, groups = build_head_arrays(
                        acts, pairs, li, hi, position, label_type
                    )
                    acc = probe_cv(X, y, groups, C=0.1)
                    head_accs[li, hi] = acc

                best_head = int(np.argmax(head_accs[li]))
                print(
                    f"  Layer {layer:2d}: best head {best_head} ({head_accs[li, best_head]:.4f}), "
                    f"mean {head_accs[li].mean():.4f}"
                )

            results[config_name] = {
                "head_accuracies": head_accs.tolist(),
                "best_per_layer": [
                    {
                        "layer": extract_layers[li],
                        "head": int(np.argmax(head_accs[li])),
                        "acc": float(head_accs[li].max()),
                    }
                    for li in range(len(extract_layers))
                ],
                "overall_best": {
                    "layer": extract_layers[
                        int(np.unravel_index(head_accs.argmax(), head_accs.shape)[0])
                    ],
                    "head": int(
                        np.unravel_index(head_accs.argmax(), head_accs.shape)[1]
                    ),
                    "acc": float(head_accs.max()),
                },
            }

            best = results[config_name]["overall_best"]
            print(
                f"  >>> Best: layer {best['layer']} head {best['head']} acc={best['acc']:.4f}"
            )

    # 2. Intermediate position probing on residual stream (with PCA)
    for position in ["final", "stmt_end"]:
        config_name = f"resid_ground_truth_{position}"
        print(f"\n{'=' * 60}")
        print(f"  {config_name} (PCA-50)")
        print(f"{'=' * 60}")

        layer_accs = []
        for li, layer in enumerate(extract_layers):
            X, y, groups = build_resid_arrays(acts, pairs, li, position, "ground_truth")
            acc = probe_cv(X, y, groups, C=0.1, n_pca=50)
            layer_accs.append(acc)
            print(f"  Layer {layer:2d}: {acc:.4f}")

        best_li = int(np.argmax(layer_accs))
        results[config_name] = {
            "layer_accuracies": {
                str(extract_layers[i]): layer_accs[i]
                for i in range(len(extract_layers))
            },
            "best_layer": extract_layers[best_li],
            "best_acc": layer_accs[best_li],
        }
        print(
            f"  >>> Best: layer {extract_layers[best_li]} acc={layer_accs[best_li]:.4f}"
        )

    # Save results
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_path = artifacts_dir / "head_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {out_path}")

    # Plot heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, config_name in enumerate(
        [
            "head_correctness_final",
            "head_ground_truth_final",
            "head_correctness_stmt_end",
            "head_ground_truth_stmt_end",
        ]
    ):
        ax = axes[idx // 2][idx % 2]
        data = np.array(results[config_name]["head_accuracies"])
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=0.70)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(extract_layers)))
        ax.set_yticklabels(extract_layers)
        ax.set_title(config_name.replace("_", " ").title())
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Head-Level Probe Accuracy — Qwen3-8B on Com2Sense", fontsize=14)
    fig.tight_layout()
    plot_path = figures_dir / "head_probe_heatmap.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {plot_path}")


if __name__ == "__main__":
    main()
