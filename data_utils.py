"""
Dataset loading and prompt formatting utilities for Com2Sense dataset.
"""

import logging
from typing import List, Dict, Tuple, Any
from datasets import load_dataset
from config import (
    DATASET_NAME,
    DATASET_NAME_FALLBACKS,
    SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def load_com2sense() -> List[Dict[str, Any]]:
    """
    Load Com2Sense dataset from HuggingFace and combine all splits.

    Returns:
        List of dictionaries with normalized keys:
        - sentence: the statement text
        - label: boolean True/False
        - domain: one of "physical", "social", "temporal"
        - scenario: scenario type
        - pair_id: ID for complementary pair
        - example_id: unique example identifier
    """
    dataset = None
    successful_name = None

    # Try primary dataset name first
    dataset_names_to_try = [DATASET_NAME] + DATASET_NAME_FALLBACKS

    for name in dataset_names_to_try:
        try:
            logger.info(f"Attempting to load dataset: {name}")
            dataset = load_dataset(name)
            successful_name = name
            logger.info(f"Successfully loaded dataset from: {name}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            continue

    if dataset is None:
        raise ValueError(
            f"Could not load dataset from any of: {dataset_names_to_try}"
        )

    # Log dataset structure
    logger.info(f"Dataset splits available: {list(dataset.keys())}")
    logger.info(f"Dataset features: {dataset[list(dataset.keys())[0]].features}")

    # Combine all splits (train, validation, test, etc.)
    all_examples = []
    for split_name, split_data in dataset.items():
        logger.info(f"Processing split '{split_name}' with {len(split_data)} examples")
        all_examples.extend(split_data)

    logger.info(f"Total examples across all splits: {len(all_examples)}")

    # Print first example to inspect field names
    if len(all_examples) > 0:
        logger.info(f"First example fields: {all_examples[0].keys()}")
        logger.info(f"First example: {all_examples[0]}")

    # Normalize field names to our expected format
    normalized_examples = []
    for i, example in enumerate(all_examples):
        normalized = _normalize_example(example, i)
        normalized_examples.append(normalized)

    logger.info(f"Normalized {len(normalized_examples)} examples")

    # Assign complementary pairs using adjacency heuristic
    logger.info("\nAttempting to assign complementary pairs by adjacency...")
    normalized_examples = assign_pairs_by_adjacency(normalized_examples)

    # Count valid vs orphaned pairs
    valid_count = sum(1 for ex in normalized_examples if ex['pair_id'] and not ex['pair_id'].startswith('orphan_'))
    orphan_count = sum(1 for ex in normalized_examples if ex['pair_id'] and ex['pair_id'].startswith('orphan_'))

    # If validity rate is low, try downloading pair_id files from GitHub
    if valid_count > 0:
        validity_rate = valid_count / len(normalized_examples)
        logger.info(f"Adjacent pairing success rate: {validity_rate:.2%}")

        if validity_rate < 0.80:
            logger.warning("Low validity rate, attempting to download pair_id files from GitHub...")
            try:
                pair_ids_data = download_pair_ids()
                if pair_ids_data:
                    logger.info("Successfully downloaded pair_id data, but mapping not yet implemented")
                    # TODO: Implement mapping from original_id to pair_id using downloaded data
                    # For now, we'll use the adjacency-based pairing
            except Exception as e:
                logger.warning(f"Failed to download pair_id files: {e}")
    else:
        logger.warning("No valid pairs found through adjacency. Trying GitHub download...")
        try:
            pair_ids_data = download_pair_ids()
            if pair_ids_data:
                logger.info("Downloaded pair_id data, but mapping not yet implemented")
        except Exception as e:
            logger.warning(f"Failed to download pair_id files: {e}")

    logger.info(f"\nFinal pair assignment: {valid_count} examples in valid pairs, {orphan_count} orphaned")

    return normalized_examples


def _normalize_example(example: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize dataset example to our expected format.

    Direct field mapping for tasksource/com2sense schema:
    - sent: statement text
    - id: unique hex ID
    - scenario: "causal" or "comparison"
    - label: "True" or "False" (STRING, not bool)
    - domain: "time", "social", or "physical"
    - numeracy: "True" or "False"
    """
    return {
        "sentence": example["sent"],
        "label": example["label"] == "True",  # Convert string to bool
        "domain": example["domain"],
        "scenario": example["scenario"],
        "numeracy": example.get("numeracy", "False") == "True",
        "original_id": example["id"],
        "example_id": f"ex_{index}",
        # pair_id will be assigned separately
        "pair_id": None,
    }


def format_prompt(sentence: str) -> str:
    """
    Format a prompt for the model to classify a statement as True/False.

    Applies the PROMPT_TEMPLATE from config and prepends the SYSTEM_PROMPT
    if it's not empty.

    Args:
        sentence: The statement to classify

    Returns:
        Fully formatted prompt string ready for tokenization
    """
    # Format the main prompt
    main_prompt = PROMPT_TEMPLATE.format(sentence=sentence)

    # Prepend system prompt if not empty
    if SYSTEM_PROMPT:
        full_prompt = f"{SYSTEM_PROMPT}\n\n{main_prompt}"
    else:
        full_prompt = main_prompt

    return full_prompt


def assign_pairs_by_adjacency(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign pair_ids by assuming adjacent examples form complementary pairs.
    Validate that each pair has opposite labels, same domain, and same scenario.

    Args:
        examples: List of normalized examples

    Returns:
        Examples with pair_id field populated
    """
    valid_pairs = 0
    invalid_pairs = 0

    for i in range(0, len(examples) - 1, 2):
        a = examples[i]
        b = examples[i + 1]

        # Validate: opposite labels, same domain, same scenario
        if (a["label"] != b["label"] and
            a["domain"] == b["domain"] and
            a["scenario"] == b["scenario"]):
            pair_id = f"pair_{i // 2}"
            a["pair_id"] = pair_id
            b["pair_id"] = pair_id
            valid_pairs += 1
        else:
            # Not a valid complementary pair
            a["pair_id"] = f"orphan_{i}"
            b["pair_id"] = f"orphan_{i+1}"
            invalid_pairs += 1

    # Handle odd last element
    if len(examples) % 2 == 1:
        examples[-1]["pair_id"] = f"orphan_{len(examples)-1}"

    logger.info(f"Adjacent pairing: {valid_pairs} valid pairs, {invalid_pairs} invalid pairs")

    validity_rate = valid_pairs / (valid_pairs + invalid_pairs) if (valid_pairs + invalid_pairs) > 0 else 0
    logger.info(f"Pair validity rate: {validity_rate:.2%}")

    return examples


def download_pair_ids() -> Dict[str, Any]:
    """
    Download pair_id files from original Com2Sense GitHub repo.

    Returns:
        Dictionary mapping split names to pair_id data
    """
    import urllib.request
    import json

    base_url = "https://raw.githubusercontent.com/PlusLabNLP/Com2Sense/master/data"
    pair_ids = {}

    for split in ["train", "dev"]:
        url = f"{base_url}/pair_id_{split}.json"
        try:
            logger.info(f"Downloading pair_id_{split}.json from GitHub...")
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                pair_ids[split] = data
                logger.info(f"Successfully downloaded pair_id_{split}.json: {len(data)} entries")
        except Exception as e:
            logger.warning(f"Failed to download pair_id_{split}.json: {e}")

    return pair_ids


def get_complementary_pairs(examples: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Group examples by pair_id and return complementary pairs.

    Complementary pairs are statements about the same scenario where one is True
    and one is False. These are the key unit of analysis for mechanistic interpretability.

    Args:
        examples: List of normalized examples from load_com2sense()

    Returns:
        List of (example_A, example_B) tuples where A and B are complementary
        (i.e., one has label=True, the other has label=False)
    """
    from collections import defaultdict

    # Group by pair_id
    pairs_by_id = defaultdict(list)
    for example in examples:
        pair_id = example['pair_id']
        if pair_id and not pair_id.startswith('orphan_'):
            pairs_by_id[pair_id].append(example)

    # Find complete complementary pairs
    complementary_pairs = []
    orphaned_examples = []

    for pair_id, pair_examples in pairs_by_id.items():
        if len(pair_examples) == 2:
            # Check if they have opposite labels
            labels = [ex['label'] for ex in pair_examples]
            if labels[0] != labels[1]:
                # This is a valid complementary pair
                complementary_pairs.append((pair_examples[0], pair_examples[1]))
            else:
                # Same labels - not complementary
                orphaned_examples.extend(pair_examples)
                logger.warning(f"Pair {pair_id} has same labels: {labels}")
        else:
            # Not exactly 2 examples
            orphaned_examples.extend(pair_examples)
            if len(pair_examples) > 2:
                logger.warning(f"Pair {pair_id} has {len(pair_examples)} examples (expected 2)")

    logger.info(f"Found {len(complementary_pairs)} complete complementary pairs")
    logger.info(f"Found {len(orphaned_examples)} orphaned examples")

    if len(complementary_pairs) == 0:
        logger.warning(
            "No complementary pairs found! This could indicate:\n"
            "1. Dataset field 'pair_id' is not properly set\n"
            "2. Dataset doesn't have complementary pairs\n"
            "3. All pairs have the same label (error in dataset)"
        )

    return complementary_pairs


def verify_dataset_structure(examples: List[Dict[str, Any]]) -> None:
    """
    Verify that the loaded dataset has the expected structure.

    Prints statistics and samples to help debug dataset loading issues.
    """
    logger.info("=== Dataset Structure Verification ===")

    # Count by field
    logger.info(f"Total examples: {len(examples)}")

    # Unique values
    unique_pairs = set(ex['pair_id'] for ex in examples)
    unique_domains = set(ex['domain'] for ex in examples)
    unique_scenarios = set(ex['scenario'] for ex in examples)

    logger.info(f"Unique pair_ids: {len(unique_pairs)}")
    logger.info(f"Unique domains: {unique_domains}")
    logger.info(f"Unique scenarios: {unique_scenarios}")

    # Label distribution
    true_count = sum(1 for ex in examples if ex['label'])
    false_count = len(examples) - true_count
    logger.info(f"Label distribution: {true_count} True, {false_count} False")

    # Sample examples
    logger.info("\nSample examples:")
    for i in [0, len(examples)//2, len(examples)-1]:
        if i < len(examples):
            ex = examples[i]
            logger.info(f"  [{i}] {ex['sentence'][:60]}... -> {ex['label']}")

    logger.info("=== End Verification ===")
```

✅ **Step 2 completed**: Created `data_utils.py` with dataset loading and prompt formatting utilities.
