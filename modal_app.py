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
        ignore=["results/", "__pycache__/", ".git/"],
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
