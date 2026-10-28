"""Configuration parameters and paths for reasoning probes."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
PROBES_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PROBES_DIR.parent
RESULTS_DIR = PROBES_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load .env files
load_dotenv(PROBES_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("OPENCODE_BASE_URI") or os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL") or "https://opencode.ai/inference/openai/v1"
API_KEY = os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "EMPTY"
DEFAULT_API_MODEL = os.getenv("OPENCODE_MODEL_ID") or os.getenv("MODEL_ID") or "deepseek-v4-pro"

# Random seed for reproducibility
DEFAULT_SEED = 42

# Confidence extraction regex patterns
CONFIDENCE_PATTERN = r"(?i)(?:confidence|probability|certainty)[\s:]*([0-9]{1,3}(?:\.[0-9]+)?)\s*%"
PROBABILITY_PATTERN = r"(?i)(?:probability|p\s*\([^)]*\))[\s:]*([0-1](?:\.[0-9]+)?)"
BOXED_ANSWER_PATTERN = r"\\boxed\{([^}]+)\}"
ANSWER_PATTERN = r"(?i)(?:final answer|answer is|conclusion is)[\s:]*([^\n\.]+)"
