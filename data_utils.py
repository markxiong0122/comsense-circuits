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
    return normalized_examples


def _normalize_example(example: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize dataset example to our expected format.

    Handles variations in field names across different dataset versions.
    """
    normalized = {}

    # Sentence text - try common field names
    for field in ['sent', 'sentence', 'text', 'statement']:
        if field in example:
            normalized['sentence'] = example[field]
            break

    if 'sentence' not in normalized:
        logger.warning(f"Example {index} missing sentence field. Keys: {example.keys()}")
        normalized['sentence'] = str(example)

    # Label (ground truth)
    for field in ['label', 'answer', 'is_true', 'truth']:
        if field in example:
            value = example[field]
            # Convert to boolean if needed
            if isinstance(value, bool):
                normalized['label'] = value
            elif isinstance(value, str):
                normalized['label'] = value.lower() in ['true', '1', 'yes']
            elif isinstance(value, int):
                normalized['label'] = bool(value)
            else:
                normalized['label'] = bool(value)
            break

    if 'label' not in normalized:
        logger.warning(f"Example {index} missing label field")
        normalized['label'] = False

    # Domain
    for field in ['domain', 'category', 'type']:
        if field in example:
            normalized['domain'] = str(example[field])
            break

    if 'domain' not in normalized:
        normalized['domain'] = 'unknown'

    # Scenario
    for field in ['scenario', 'scenarios', 'context']:
        if field in example:
            normalized['scenario'] = str(example[field])
            break

    if 'scenario' not in normalized:
        normalized['scenario'] = 'unknown'

    # Pair ID - for grouping complementary pairs
    for field in ['pair_id', 'pair', 'group_id', 'qid', 'question_id']:
        if field in example:
            normalized['pair_id'] = str(example[field])
            break

    if 'pair_id' not in normalized:
        # If no pair_id, use a hash of the sentence as fallback
        normalized['pair_id'] = f"auto_{hash(normalized['sentence']) % 1000000}"

    # Example ID - unique identifier
    for field in ['example_id', 'id', 'idx', 'index']:
        if field in example:
            normalized['example_id'] = str(example[field])
            break

    if 'example_id' not in normalized:
        normalized['example_id'] = f"ex_{index}"

    return normalized


def format_prompt(sentence: str) -> str:
    """
    Format a prompt for the model to classify a statement as True/False.

    Applies the PROMPT_TEMPLATE from config and prepends the SYSTEM_PROMPT
    for Qwen3's non-thinking mode.

    Args:
        sentence: The statement to classify

    Returns:
        Fully formatted prompt string ready for tokenization
    """
    # Format the main prompt
    main_prompt = PROMPT_TEMPLATE.format(sentence=sentence)

    # Prepend system prompt for Qwen3 non-thinking mode
    full_prompt = f"{SYSTEM_PROMPT}\n\n{main_prompt}"

    return full_prompt


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
