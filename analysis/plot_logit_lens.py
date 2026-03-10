"""
CPU-side plotting and summary for logit-lens results.

Reads:
  results/analysis/logit_lens_results.json

Writes:
  results/analysis/logit_lens_summary.json
  results/analysis/logit_lens_trajectories.png
  results/analysis/logit_lens_failure_layers.png
  results/analysis/logit_lens_domain_trajectories.png
  results/analysis/logit_lens_incorrect_heatmap.png
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_json: Path) -> dict[str, Any]:
    with open(results_json) as f:
        return json.load(f)


def _mean_and_sem(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(arrays, axis=0)
    mean = stack.mean(axis=0)
    sem = stack.std(axis=0, ddof=0) / np.sqrt(max(len(arrays), 1))
    return mean, sem


def _first_strong_divergence_layer(
    correct_mean: np.ndarray,
    incorrect_mean: np.ndarray,
    threshold: float = 0.5,
) -> int | None:
    delta = np.abs(correct_mean - incorrect_mean)
    idx = np.where(delta >= threshold)[0]
    return int(idx[0]) if len(idx) > 0 else None


def _bucket_layer(layer: int | None, n_layers: int) -> str:
    if layer is None:
        return "never"
    if layer == 0:
        return "layer_0"
    if layer < 20:
        return "layers_1_19"
    if layer < 30:
        return "layers_20_29"
    if layer < n_layers - 1:
        return "layers_30_plus"
    if layer == n_layers - 1:
        return "final_layer_only"
    return "other"


def _stable_wrong_layer(trace: dict[str, Any]) -> int | None:
    correct_by_layer = trace["correct_by_layer"]
    final_correct = bool(correct_by_layer[-1])
    for layer_idx in range(len(correct_by_layer)):
        if all(c == final_correct for c in correct_by_layer[layer_idx:]):
            return layer_idx
    return None


def plot_main_trajectories(
    pair_results: list[dict[str, Any]],
    n_layers: int,
    output_path: Path,
) -> dict[str, Any]:
    correct_signed = [
        np.array(pair["correct"]["signed_logit_gap"], dtype=float)
        for pair in pair_results
    ]
    incorrect_signed = [
        np.array(pair["incorrect"]["signed_logit_gap"], dtype=float)
        for pair in pair_results
    ]

    correct_mean, correct_sem = _mean_and_sem(correct_signed)
    incorrect_mean, incorrect_sem = _mean_and_sem(incorrect_signed)

    divergence_layer = _first_strong_divergence_layer(
        correct_mean, incorrect_mean, threshold=0.5
    )
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        layers, correct_mean, color="#16a34a", linewidth=2.2, label="Correct examples"
    )
    ax.fill_between(
        layers,
        correct_mean - correct_sem,
        correct_mean + correct_sem,
        color="#16a34a",
        alpha=0.18,
    )

    ax.plot(
        layers,
        incorrect_mean,
        color="#dc2626",
        linewidth=2.2,
        label="Incorrect examples",
    )
    ax.fill_between(
        layers,
        incorrect_mean - incorrect_sem,
        incorrect_mean + incorrect_sem,
        color="#dc2626",
        alpha=0.18,
    )

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="Decision boundary",
    )
    ax.axvline(
        20,
        color="#2563eb",
        linestyle=":",
        linewidth=1.2,
        alpha=0.9,
        label="Probe rise begins (L20)",
    )
    if divergence_layer is not None:
        ax.axvline(
            divergence_layer,
            color="#7c3aed",
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
            label=f"First strong divergence (L{divergence_layer})",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Signed logit gap")
    ax.set_title(
        "Logit Lens Trajectories on Selected Com2Sense Failure Pairs\n"
        "Positive = supports ground-truth answer; Negative = supports wrong answer"
    )
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "first_strong_divergence_layer": divergence_layer,
        "mean_final_signed_gap_correct": float(correct_mean[-1]),
        "mean_final_signed_gap_incorrect": float(incorrect_mean[-1]),
    }


def plot_failure_layers(
    pair_results: list[dict[str, Any]],
    n_layers: int,
    output_path: Path,
) -> dict[str, Any]:
    first_wrong_layers = []
    stable_wrong_layers = []

    for pair in pair_results:
        trace = pair["incorrect"]
        first_wrong_layers.append(trace["first_wrong_layer"])
        stable_wrong_layers.append(_stable_wrong_layer(trace))

    first_wrong_counter = Counter(
        _bucket_layer(layer, n_layers) for layer in first_wrong_layers
    )
    stable_wrong_counter = Counter(
        _bucket_layer(layer, n_layers) for layer in stable_wrong_layers
    )

    order = [
        "layer_0",
        "layers_1_19",
        "layers_20_29",
        "layers_30_plus",
        "final_layer_only",
        "never",
    ]
    labels = [
        "Layer 0",
        "Layers 1-19",
        "Layers 20-29",
        "Layers 30+",
        "Final layer only",
        "Never",
    ]

    first_vals = [first_wrong_counter.get(k, 0) for k in order]
    stable_vals = [stable_wrong_counter.get(k, 0) for k in order]

    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(
        x - width / 2,
        first_vals,
        width=width,
        color="#f97316",
        label="First wrong layer",
    )
    ax.bar(
        x + width / 2,
        stable_vals,
        width=width,
        color="#8b5cf6",
        label="Stable wrong layer",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Number of incorrect examples")
    ax.set_title(
        "When do incorrect examples first go wrong, and when do they stay wrong?"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "first_wrong_bucket_counts": dict(first_wrong_counter),
        "stable_wrong_bucket_counts": dict(stable_wrong_counter),
    }


def plot_domain_trajectories(
    pair_results: list[dict[str, Any]],
    n_layers: int,
    output_path: Path,
) -> dict[str, Any]:
    by_domain: dict[str, list[np.ndarray]] = defaultdict(list)
    for pair in pair_results:
        by_domain[pair["domain"]].append(
            np.array(pair["incorrect"]["signed_logit_gap"], dtype=float)
        )

    colors = {
        "physical": "#2563eb",
        "social": "#16a34a",
        "temporal": "#dc2626",
        "time": "#f59e0b",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    layers = np.arange(n_layers)
    domain_summary = {}

    for domain, traces in sorted(by_domain.items()):
        mean, sem = _mean_and_sem(traces)
        color = colors.get(domain, None)
        ax.plot(layers, mean, linewidth=2.0, label=domain, color=color)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.15, color=color)
        domain_summary[domain] = {
            "n_examples": len(traces),
            "mean_final_signed_gap": float(mean[-1]),
            "mean_layer_20_signed_gap": float(mean[20])
            if n_layers > 20
            else float(mean[-1]),
        }

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(20, color="#64748b", linestyle=":", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Signed logit gap")
    ax.set_title("Incorrect-example logit-lens trajectories by domain")
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Domain")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return domain_summary


def plot_incorrect_heatmap(
    pair_results: list[dict[str, Any]],
    n_layers: int,
    output_path: Path,
) -> dict[str, Any]:
    rows = []
    row_meta = []

    for pair in pair_results:
        trace = pair["incorrect"]
        arr = np.array(trace["signed_logit_gap"], dtype=float)
        stable_wrong = _stable_wrong_layer(trace)
        first_wrong = trace["first_wrong_layer"]
        sort_key = (
            stable_wrong
            if stable_wrong is not None
            else (first_wrong if first_wrong is not None else 10_000)
        )
        rows.append(arr)
        row_meta.append(
            {
                "pair_id": pair["pair_id"],
                "domain": pair["domain"],
                "first_wrong_layer": first_wrong,
                "stable_wrong_layer": stable_wrong,
                "sort_key": sort_key,
            }
        )

    order = np.argsort([m["sort_key"] for m in row_meta])
    matrix = np.stack([rows[i] for i in order], axis=0)

    vmax = max(1.0, float(np.percentile(np.abs(matrix), 95)))
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Incorrect examples (sorted by stable wrong layer)")
    ax.set_title(
        "Heatmap of incorrect-example signed logit gaps\nRed = supports wrong answer, Blue = supports correct answer"
    )
    ax.axvline(20, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Signed logit gap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "heatmap_examples": int(matrix.shape[0]),
        "heatmap_layers": int(matrix.shape[1]),
    }


def build_summary(
    data: dict[str, Any],
    trajectory_stats: dict[str, Any],
    failure_stats: dict[str, Any],
    domain_stats: dict[str, Any],
    heatmap_stats: dict[str, Any],
) -> dict[str, Any]:
    pair_results = data["pair_results"]
    n_layers = int(data["n_layers"])

    incorrect_first_wrong = [
        pair["incorrect"]["first_wrong_layer"]
        for pair in pair_results
        if pair["incorrect"]["first_wrong_layer"] is not None
    ]
    incorrect_stable_wrong = []
    for pair in pair_results:
        stable_wrong = _stable_wrong_layer(pair["incorrect"])
        if stable_wrong is not None:
            incorrect_stable_wrong.append(stable_wrong)
    correct_stable_correct = [
        pair["correct"]["stable_correctness_layer"]
        for pair in pair_results
        if pair["correct"]["stable_correctness_layer"] is not None
    ]

    summary = {
        "model_name": data["model_name"],
        "n_layers": n_layers,
        "n_pairs_processed": int(data["n_pairs_processed"]),
        "trajectory_stats": trajectory_stats,
        "failure_stats": failure_stats,
        "domain_stats": domain_stats,
        "heatmap_stats": heatmap_stats,
        "aggregate_metrics": {
            "mean_incorrect_first_wrong_layer": (
                float(np.mean(incorrect_first_wrong)) if incorrect_first_wrong else None
            ),
            "mean_incorrect_stable_wrong_layer": (
                float(np.mean(np.array(incorrect_stable_wrong, dtype=float)))
                if incorrect_stable_wrong
                else None
            ),
            "mean_correct_stable_correct_layer": (
                float(np.mean(correct_stable_correct))
                if correct_stable_correct
                else None
            ),
        },
    }

    return summary


def main(output_dir: Path | None = None) -> dict[str, Any]:
    repo_dir = Path(__file__).parent.parent
    if output_dir is None:
        output_dir = repo_dir / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = output_dir / "logit_lens_results.json"
    if not results_json.exists():
        alt = repo_dir / "analysis" / "logit_lens_results.json"
        if alt.exists():
            results_json = alt
        else:
            raise FileNotFoundError(
                f"Could not find logit-lens results at {results_json} or {alt}"
            )

    data = load_results(results_json)
    pair_results = data["pair_results"]
    n_layers = int(data["n_layers"])

    trajectory_stats = plot_main_trajectories(
        pair_results,
        n_layers,
        output_dir / "logit_lens_trajectories.png",
    )
    failure_stats = plot_failure_layers(
        pair_results,
        n_layers,
        output_dir / "logit_lens_failure_layers.png",
    )
    domain_stats = plot_domain_trajectories(
        pair_results,
        n_layers,
        output_dir / "logit_lens_domain_trajectories.png",
    )
    heatmap_stats = plot_incorrect_heatmap(
        pair_results,
        n_layers,
        output_dir / "logit_lens_incorrect_heatmap.png",
    )

    summary = build_summary(
        data=data,
        trajectory_stats=trajectory_stats,
        failure_stats=failure_stats,
        domain_stats=domain_stats,
        heatmap_stats=heatmap_stats,
    )

    with open(output_dir / "logit_lens_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"  {output_dir / 'logit_lens_summary.json'}")
    print(f"  {output_dir / 'logit_lens_trajectories.png'}")
    print(f"  {output_dir / 'logit_lens_failure_layers.png'}")
    print(f"  {output_dir / 'logit_lens_domain_trajectories.png'}")
    print(f"  {output_dir / 'logit_lens_incorrect_heatmap.png'}")

    if summary["trajectory_stats"]["first_strong_divergence_layer"] is not None:
        print(
            f"First strong divergence layer: "
            f"{summary['trajectory_stats']['first_strong_divergence_layer']}"
        )

    return summary


if __name__ == "__main__":
    main()
