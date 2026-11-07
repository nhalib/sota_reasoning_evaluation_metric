"""Configuration parameters and paths for reasoning probes."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
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

# Random seed for reproducibility
DEFAULT_SEED = 42

# HTTP timeout and completion token bounds
DEFAULT_TIMEOUT = float(os.getenv("PROBE_TIMEOUT", "180.0"))
DEFAULT_TOKEN_CEILING = int(os.getenv("COMPLETION_TOKEN_CEILING", "16384"))
DEFAULT_TRUNCATION_RETRIES = int(os.getenv("TRUNCATION_RETRIES", "4"))

def get_timestamp_str(dt: Optional[datetime] = None) -> str:
    """Returns a UTC timestamp formatted as YYYYMMDD_HHMMSS."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")

# Confidence extraction regex patterns
CONFIDENCE_PATTERN = r"(?i)(?:confidence|probability|certainty)[\s:]*([0-9]{1,3}(?:\.[0-9]+)?)\s*%"
PROBABILITY_PATTERN = r"(?i)(?:probability|p\s*\([^)]*\))[\s:]*([0-1](?:\.[0-9]+)?)"
BOXED_ANSWER_PATTERN = r"\\boxed\{([^}]+)\}"
ANSWER_PATTERN = r"(?i)(?:final answer|answer is|conclusion is)[\s:]*([^\n\.]+)"
