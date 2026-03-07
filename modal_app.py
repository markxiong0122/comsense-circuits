"""
Modal infrastructure for ComSense Circuits project.
Runs evaluation on A100-80GB GPU with proper volume mounting.
"""

import modal

app = modal.App("comsense-circuits")

# Volumes for caching and results
hf_cache_vol = modal.Volume.from_name("comsense-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("comsense-results", create_if_missing=True)

# Image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformer_lens",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
        "tqdm",
        "scikit-learn",     # for later probing phase
        "matplotlib",       # for later analysis phase
        gpu="A100",         # compile torch for A100
    )
    .add_local_dir(
        ".",
        remote_path="/root/comsense-circuits",
        ignore=["results/", "__pycache__/", ".git/", ".venv/", "*.pyc"],
    )
)

# Shared kwargs for all GPU functions
SHARED_KWARGS = dict(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    timeout=2 * 3600,  # 2 hours
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/comsense-circuits/results": results_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)

# CPU-only kwargs for analysis scripts (no GPU needed)
CPU_KWARGS = dict(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=1 * 3600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/comsense-circuits/results": results_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)


@app.function(**SHARED_KWARGS)
def run_evaluate() -> dict:
    """
    Main GPU function that runs the behavioral evaluation.

    Loads the model, runs evaluation on Com2Sense dataset, and saves results.

    Returns:
        Dictionary with evaluation summary statistics
    """
    import sys
    import logging
    import subprocess

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("Starting ComSense Circuits Behavioral Evaluation")
    logger.info("="*60)

    # Run the main evaluation script
    logger.info("\n[MAIN] Running behavioral evaluation...")
    sys.path.insert(0, "/root/comsense-circuits")

    try:
        # Import and call main() from evaluate.py
        from evaluate import main
        results = main()

        # Commit results to volume
        logger.info("\n[MAIN] Committing results to volume...")
        results_vol.commit()

        logger.info("\n[MAIN] ✓ Evaluation completed successfully!")
        return results

    except Exception as e:
        logger.error(f"\n[MAIN] ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.function(**SHARED_KWARGS)
def run_smoke_test() -> dict:
    """
    Quick smoke test to verify model loading and basic functionality.
    Useful for debugging without running the full evaluation.

    Returns:
        Dictionary with smoke test results
    """
    import sys
    import logging
    import torch

    # Add local modules to path
    sys.path.insert(0, "/root/comsense-circuits")

    from transformer_lens import HookedTransformer
    from config import MODEL_NAME, MODEL_NAME_FALLBACKS, DTYPE

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("Running Smoke Test")
    logger.info("="*60)

    results = {
        "cuda_available": False,
        "model_loaded": False,
        "forward_pass_success": False,
        "tokens_verified": False,
        "hook_extraction_success": False,
        "n_layers": None,
        "d_model": None,
    }

    # Check CUDA
    results["cuda_available"] = torch.cuda.is_available()
    logger.info(f"CUDA available: {results['cuda_available']}")

    if results["cuda_available"]:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Try loading model
    model_names_to_try = [MODEL_NAME] + MODEL_NAME_FALLBACKS
    model = None
    successful_model_name = None

    for model_name in model_names_to_try:
        try:
            logger.info(f"\nTrying to load model: {model_name}")
            model = HookedTransformer.from_pretrained(
                model_name,
                dtype=DTYPE,
            )
            successful_model_name = model_name
            results["model_loaded"] = True
            results["model_name"] = model_name
            logger.info(f"✓ Successfully loaded: {model_name}")
            break
        except Exception as e:
            logger.warning(f"✗ Failed to load {model_name}: {e}")
            continue

    if not results["model_loaded"]:
        logger.error("Failed to load any model!")
        return results

    # Test forward pass
    try:
        logger.info("\nTesting forward pass...")
        test_prompt = "Is the following statement true or false? The sky is blue."
        logits = model(test_prompt)
        logger.info(f"✓ Forward pass successful!")
        logger.info(f"  Input shape: {test_prompt}")
        logger.info(f"  Logits shape: {logits.shape}")
        results["forward_pass_success"] = True
        results["logits_shape"] = list(logits.shape)
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        return results

    # Verify tokens
    try:
        logger.info("\nVerifying True/False tokens...")
        from config import TRUE_TOKEN_CANDIDATES, FALSE_TOKEN_CANDIDATES

        true_token_id = None
        false_token_id = None

        for candidate in TRUE_TOKEN_CANDIDATES:
            tokens = model.to_tokens(candidate, prepend_bos=False)
            if tokens.shape[-1] == 1:
                true_token_id = tokens[0, 0].item()
                logger.info(f"  ✓ True token: {candidate!r} -> id {true_token_id}")
                break

        for candidate in FALSE_TOKEN_CANDIDATES:
            tokens = model.to_tokens(candidate, prepend_bos=False)
            if tokens.shape[-1] == 1:
                false_token_id = tokens[0, 0].item()
                logger.info(f"  ✓ False token: {candidate!r} -> id {false_token_id}")
                break

        if true_token_id is not None and false_token_id is not None:
            results["tokens_verified"] = True
            results["true_token_id"] = true_token_id
            results["false_token_id"] = false_token_id
        else:
            logger.warning("  ✗ Could not verify both True and False tokens")

    except Exception as e:
        logger.error(f"✗ Token verification failed: {e}")

    # Verify hook-based activation extraction
    try:
        logger.info("\nVerifying hook-based activation extraction...")
        from evaluate import verify_hook_extraction
        hook_results = verify_hook_extraction(model)
        results.update(hook_results)
        logger.info(f"✓ Hook extraction verified: {hook_results['n_layers']} layers, d_model={hook_results['d_model']}")
    except Exception as e:
        logger.error(f"✗ Hook extraction failed: {e}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    logger.info("\n" + "="*60)
    logger.info("Smoke Test Summary:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)

    return results


@app.function(**SHARED_KWARGS)
def run_extract_activations() -> dict:
    """
    GPU function that extracts residual stream activations for the 200
    selected asymmetric pairs and saves them to the results volume.

    Returns:
        Dictionary with extraction summary statistics
    """
    import sys
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("Starting Phase 2: Activation Extraction")
    logger.info("="*60)

    sys.path.insert(0, "/root/comsense-circuits")

    try:
        from extract_activations import main
        results = main()

        logger.info("\n[MAIN] Committing results to volume...")
        results_vol.commit()

        logger.info("\n[MAIN] ✓ Extraction completed successfully!")
        return results

    except Exception as e:
        logger.error(f"\n[MAIN] ✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.local_entrypoint()
def extract_activations():
    """
    Local entrypoint to run activation extraction on Modal.

    Usage:
        modal run modal_app.py::extract_activations
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Launching activation extraction on Modal...")
    logger.info("This will:")
    logger.info("  1. Load the 200 selected asymmetric pairs")
    logger.info("  2. Run 400 forward passes with run_with_cache()")
    logger.info("  3. Save resid_post [n_layers, d_model] per example")
    logger.info("  4. Write activations.pt and activations_meta.json to results volume")
    logger.info("")

    results = run_extract_activations.remote()

    logger.info("\n" + "="*60)
    logger.info("Extraction Complete!")
    logger.info("="*60)
    logger.info(f"Model:      {results.get('model_name', 'N/A')}")
    logger.info(f"n_layers:   {results.get('n_layers', 'N/A')}")
    logger.info(f"d_model:    {results.get('d_model', 'N/A')}")
    logger.info(f"Pairs:      {results.get('n_pairs', 'N/A')}")
    logger.info(f"Examples:   {results.get('n_examples', 'N/A')}")
    logger.info(f"Output:     {results.get('output_mb', 'N/A'):.1f} MB")
    logger.info(f"\nResults saved to: /root/comsense-circuits/results/activations/")
    logger.info("="*60)


@app.function(**SHARED_KWARGS)
def run_activation_patching() -> dict:
    """Run activation patching on Modal GPU."""
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Phase 3: Activation Patching")
    sys.path.insert(0, "/root/comsense-circuits")

    try:
        from activation_patching import main
        results = main()
        results_vol.commit()
        logger.info("Activation patching completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Activation patching failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.function(**SHARED_KWARGS)
def run_extract_head_activations() -> dict:
    """Extract per-head activations on Modal GPU."""
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Phase 3b: Head-Level Activation Extraction")
    sys.path.insert(0, "/root/comsense-circuits")

    try:
        from extract_head_activations import main
        results = main()
        results_vol.commit()
        logger.info("Head activation extraction completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Head activation extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.function(**SHARED_KWARGS)
def run_head_patching() -> dict:
    """Run head-level patching on Modal GPU."""
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Phase 3c: Head-Level Patching")
    sys.path.insert(0, "/root/comsense-circuits")

    try:
        from head_patching import main
        results = main()
        results_vol.commit()
        logger.info("Head patching completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Head patching failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.function(**SHARED_KWARGS)
def run_l2_sweep() -> dict:
    """Run L2 regularization sweep. Uses GPU kwargs to access saved activations from volume."""
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    sys.path.insert(0, "/root/comsense-circuits")

    try:
        import json
        import torch
        import numpy as np
        from pathlib import Path
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler

        code_dir = Path("/root/comsense-circuits")
        results_dir = Path("/root/comsense-circuits/results")

        # Load activations from the results volume
        act_path = results_dir / "activations" / "activations.pt"
        if not act_path.exists():
            # Re-extract activations (we have GPU access)
            logger.info("activations.pt not found — re-extracting from model...")
            from extract_activations import main as extract_main
            extract_main()
            results_vol.commit()

        acts = torch.load(act_path, weights_only=False)

        with open(code_dir / "eval" / "selected_pairs_for_patching.json") as f:
            pairs = json.load(f)

        first_tensor = next(iter(acts.values()))
        n_layers = first_tensor.shape[0]
        logger.info(f"Loaded {len(acts)} examples, {n_layers} layers")

        from analysis.probe import build_arrays

        C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        probe_layers = list(range(0, 5)) + list(range(18, n_layers))

        configs = [
            ("correctness", None, "Correctness (no PCA)"),
            ("correctness", 50, "Correctness + PCA-50"),
            ("ground_truth", 50, "Ground truth + PCA-50"),
        ]

        all_results = {}

        for label_type, n_pca, label in configs:
            logger.info(f"\n  [{label}]")
            best_overall = {"C": None, "layer": None, "acc": 0.0}
            c_results = {}

            for C in C_values:
                layer_accs = {}
                for layer in probe_layers:
                    X, y, groups, _ = build_arrays(acts, pairs, layer, label_type)

                    cv = GroupKFold(n_splits=5)
                    fold_scores = []
                    for train_idx, test_idx in cv.split(X, y, groups=groups):
                        X_tr, X_te = X[train_idx], X[test_idx]
                        y_tr, y_te = y[train_idx], y[test_idx]

                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_tr)
                        X_te = scaler.transform(X_te)

                        if n_pca is not None:
                            n_comp = min(n_pca, X_tr.shape[0], X_tr.shape[1])
                            pca = PCA(n_components=n_comp, random_state=42)
                            X_tr = pca.fit_transform(X_tr)
                            X_te = pca.transform(X_te)

                        clf = LogisticRegression(C=C, max_iter=2000, random_state=42, penalty="l2")
                        clf.fit(X_tr, y_tr)
                        fold_scores.append(clf.score(X_te, y_te))

                    acc = float(np.mean(fold_scores))
                    layer_accs[layer] = acc

                best_layer = max(layer_accs, key=layer_accs.get)
                best_acc = layer_accs[best_layer]
                c_results[str(C)] = {"best_layer": best_layer, "best_acc": best_acc}

                if best_acc > best_overall["acc"]:
                    best_overall = {"C": C, "layer": best_layer, "acc": best_acc}

                logger.info(f"  C={C:<8} best layer={best_layer:2d}  acc={best_acc:.4f}")

            all_results[label] = {"by_C": c_results, "best": best_overall}
            logger.info(f"  >>> Best: C={best_overall['C']} layer={best_overall['layer']} acc={best_overall['acc']:.4f}")

        out_dir = results_dir / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "l2_sweep_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved to {out_path}")

        results_vol.commit()
        return all_results

    except Exception as e:
        logger.error(f"L2 sweep failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.function(**CPU_KWARGS)
def run_head_probing() -> dict:
    """Run head-level probing on saved head activations (CPU only)."""
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    sys.path.insert(0, "/root/comsense-circuits/analysis")
    sys.path.insert(0, "/root/comsense-circuits")

    try:
        from pathlib import Path
        out_dir = Path("/root/comsense-circuits/results/analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        from analysis.probe_heads import main
        main(output_dir=out_dir)
        results_vol.commit()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Head probing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.local_entrypoint()
def l2_sweep():
    """Run just the L2 regularization sweep. Usage: modal run modal_app.py::l2_sweep"""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Launching L2 regularization sweep on Modal...")
    result = run_l2_sweep.remote()
    logger.info("\nL2 Sweep Results:")
    for label, data in result.items():
        if isinstance(data, dict) and "best" in data:
            b = data["best"]
            logger.info(f"  {label}: best C={b['C']} layer={b['layer']} acc={b['acc']:.4f}")
    logger.info("\nDone! Download: uv run modal volume get comsense-results /analysis ./results_download/analysis")


@app.local_entrypoint()
def fix_run():
    """
    Re-run the failed steps from the initial pipeline.
    Usage: modal run modal_app.py::fix_run
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Running fixed steps...")

    # Step 1: L2 sweep + Head extraction in parallel
    logger.info("\n[1/2] L2 sweep + Head extraction (parallel)...")
    l2_handle = run_l2_sweep.spawn()
    head_handle = run_extract_head_activations.spawn()

    try:
        l2_result = l2_handle.get()
        logger.info("  L2 sweep done!")
        for label, data in l2_result.items():
            if isinstance(data, dict) and "best" in data:
                b = data["best"]
                logger.info(f"    {label}: best C={b['C']} layer={b['layer']} acc={b['acc']:.4f}")
    except Exception as e:
        logger.error(f"  L2 sweep failed: {e}")

    try:
        head_result = head_handle.get()
        logger.info(f"  Head extraction done! {head_result.get('n_examples', '?')} examples")
    except Exception as e:
        logger.error(f"  Head extraction failed: {e}")
        logger.info("  Skipping head probing.")
        return

    # Step 2: Head probing (needs head extraction data)
    logger.info("\n[2/2] Head-level probing...")
    try:
        run_head_probing.remote()
        logger.info("  Head probing done!")
    except Exception as e:
        logger.error(f"  Head probing failed: {e}")

    logger.info("\nFix run complete!")
    logger.info("Download all results: uv run modal volume get comsense-results / ./results")


@app.local_entrypoint()
def run_all():
    """
    Run the complete Week 9 pipeline sequentially.

    Usage: modal run modal_app.py::run_all

    Order:
      1. L2 regularization sweep (CPU, uses existing activations)
      2. Activation patching (GPU)
      3. Head-level activation extraction (GPU)
      4. Head-level probing (CPU, uses head activations from step 3)
      5. Head-level patching (GPU, uses patching results from step 2)
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Week 9 Full Pipeline")
    logger.info("=" * 60)

    # Step 1: L2 sweep (CPU, no dependencies)
    logger.info("\n[1/5] L2 regularization sweep...")
    try:
        run_l2_sweep.remote()
        logger.info("  Done!")
    except Exception as e:
        logger.error(f"  Failed: {e}")

    # Step 2: Activation patching (GPU)
    logger.info("\n[2/5] Activation patching (layers 18-35)...")
    try:
        patch_results = run_activation_patching.remote()
        logger.info("  Done!")
        layer_summary = patch_results.get("layer_summary", {})
        top_layers = sorted(layer_summary.items(), key=lambda x: -x[1]["flip_rate"])[:3]
        for L, s in top_layers:
            logger.info(f"    Layer {L}: flip_rate={s['flip_rate']:.3f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")

    # Step 3: Head activation extraction (GPU)
    logger.info("\n[3/5] Head-level activation extraction...")
    try:
        head_ext_results = run_extract_head_activations.remote()
        logger.info(f"  Done! {head_ext_results.get('n_examples', '?')} examples, {head_ext_results.get('output_mb', 0):.1f} MB")
    except Exception as e:
        logger.error(f"  Failed: {e}")

    # Step 4: Head probing (CPU, needs step 3)
    logger.info("\n[4/5] Head-level probing...")
    try:
        run_head_probing.remote()
        logger.info("  Done!")
    except Exception as e:
        logger.error(f"  Failed: {e}")

    # Step 5: Head patching (GPU, needs step 2)
    logger.info("\n[5/5] Head-level patching...")
    try:
        head_patch_results = run_head_patching.remote()
        logger.info("  Done!")
        top = head_patch_results.get("top_20_heads", [])[:5]
        for h in top:
            logger.info(f"    L{h['layer']}.H{h['head']}: effect={h['mean_effect']:+.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("Download results: uv run modal volume get comsense-results / ./results")
    logger.info("=" * 60)


@app.local_entrypoint()
def patch():
    """Run activation patching. Usage: modal run modal_app.py::patch"""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Launching activation patching on Modal...")
    results = run_activation_patching.remote()

    logger.info("\nActivation Patching Complete!")
    logger.info(f"Pairs processed: {results.get('n_pairs', 'N/A')}")
    layer_summary = results.get("layer_summary", {})
    for L, s in sorted(layer_summary.items(), key=lambda x: int(x[0])):
        logger.info(f"  Layer {L}: flip_rate={s['flip_rate']:.3f}")


@app.local_entrypoint()
def extract_heads():
    """Extract head activations. Usage: modal run modal_app.py::extract_heads"""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Launching head activation extraction on Modal...")
    results = run_extract_head_activations.remote()

    logger.info("\nHead Extraction Complete!")
    logger.info(f"Examples: {results.get('n_examples', 'N/A')}")
    logger.info(f"Output: {results.get('output_mb', 0):.1f} MB")


@app.local_entrypoint()
def patch_heads():
    """Run head-level patching. Usage: modal run modal_app.py::patch_heads"""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Launching head-level patching on Modal...")
    results = run_head_patching.remote()

    logger.info("\nHead Patching Complete!")
    top = results.get("top_20_heads", [])[:5]
    for h in top:
        logger.info(f"  L{h['layer']}.H{h['head']}: effect={h['mean_effect']:+.4f} flip_rate={h['flip_rate']:.3f}")


@app.local_entrypoint()
def launch_eval():
    """
    Local entrypoint to launch the evaluation on Modal.

    Usage:
        modal run modal_app.py::launch_eval
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Launching evaluation on Modal...")
    logger.info("This will:")
    logger.info("  1. Build the Modal image (first run caches dependencies)")
    logger.info("  2. Download Qwen3-8B weights (first run only)")
    logger.info("  3. Load model via TransformerLens")
    logger.info("  4. Run evaluation on all Com2Sense examples")
    logger.info("  5. Save results to the results volume")
    logger.info("  6. Print summary to stdout")
    logger.info("")

    # Call the remote function
    results = run_evaluate.remote()

    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)
    logger.info(f"Overall accuracy: {results.get('overall_accuracy', 'N/A')}")
    logger.info(f"Total examples: {results.get('total_examples', 'N/A')}")
    logger.info(f"Asymmetric pairs: {results.get('asymmetric_pairs', 'N/A')}")
    logger.info(f"\nResults saved to: /root/comsense-circuits/results/eval/")
    logger.info("="*60)


@app.local_entrypoint()
def smoke_test():
    """
    Local entrypoint to run a quick smoke test.

    Usage:
        modal run modal_app.py::smoke_test
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Running smoke test on Modal...")
    results = run_smoke_test.remote()

    all_passed = (
        results.get("model_loaded")
        and results.get("forward_pass_success")
        and results.get("tokens_verified")
        and results.get("hook_extraction_success")
    )
    if all_passed:
        logger.info("\n✓ Smoke test PASSED!")
        logger.info(f"  n_layers={results.get('n_layers')}, d_model={results.get('d_model')}")
    else:
        logger.error("\n✗ Smoke test FAILED!")
        for key, value in results.items():
            logger.error(f"  {key}: {value}")
