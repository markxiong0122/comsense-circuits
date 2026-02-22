"""
Central configuration for ComSense Circuits project.
All other modules import from this file.
"""

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-8B"  # Instruct version, NOT base
DATASET_NAME = "tasksource/com2sense"  # Com2Sense dataset on HuggingFace

# Paths and Storage
RESULTS_DIR = "/root/comsense-circuits/results"

# Hardware and Performance
DEVICE = "cuda"
DTYPE = "bfloat16"

# Token Labels for True/False Classification
# IMPORTANT: Include leading space. Verify by running model.to_single_token(" True")
# after model loads. If that fails, try without space, or "true"/"false"
TRUE_TOKEN = " True"
FALSE_TOKEN = " False"

# Qwen3 Non-Thinking Mode System Prompt
# Qwen3 supports /no_think to disable chain-of-thought thinking mode
# This is critical - we need direct representations, not CoT-contaminated residual streams
SYSTEM_PROMPT = "/no_think"

# Prompt Template for True/False Classification
# Keep it simple and direct. The model should answer with just "True" or "False".
PROMPT_TEMPLATE = """Is the following statement true or false? Respond with only "True" or "False".

Statement: "{sentence}"

Answer:"""

# Alternative dataset names to try if the primary one fails
DATASET_NAME_FALLBACKS = [
    "dali-does/com2sense",  # Original expected name
    "com2sense/com2sense",
    "com2sense",
]

# Fallback model names if TransformerLens doesn't support Qwen3-8B
MODEL_NAME_FALLBACKS = [
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-0.6B-Base",
]

# Token candidates to try (in order of preference)
# We need single-token representations for clean logit comparison
TRUE_TOKEN_CANDIDATES = [" True", "True", " true", "true"]
FALSE_TOKEN_CANDIDATES = [" False", "False", " false", "false"]
