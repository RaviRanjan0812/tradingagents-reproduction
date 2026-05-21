"""
Central configuration for the trading agents project.

Design principle: every "knob" (model choice, rate limit, file path) lives here.
When you want to swap a model or change a setting, you edit ONE file, not ten.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / ".cache"
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
RESULTS_DIR = PROJECT_ROOT / "results"

for directory in [CACHE_DIR, DATA_CACHE_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# MODEL ROUTING
# ============================================================================
MODEL_ROUTING = {
    "fundamentals_analyst": ("gemini", "gemini-2.5-flash-lite"),
    "sentiment_analyst":    ("gemini", "gemini-2.5-flash-lite"),
    "news_analyst":         ("gemini", "gemini-2.5-flash-lite"),
    "technical_analyst":    ("gemini", "gemini-2.5-flash-lite"),
    "bull_researcher":      ("gemini", "gemini-2.5-flash"),
    "bear_researcher":      ("gemini", "gemini-2.5-flash"),
    "trader":               ("gemini", "gemini-2.5-flash"),
    "risk_aggressive":      ("gemini", "gemini-2.5-flash"),
    "risk_neutral":         ("gemini", "gemini-2.5-flash"),
    "risk_conservative":    ("gemini", "gemini-2.5-flash"),
    "fund_manager":         ("gemini", "gemini-2.5-flash"),
}
# ============================================================================
# RATE LIMITS (requests per minute)
# ============================================================================

RATE_LIMITS_RPM = {
    "gemini": 60,
}


# ============================================================================
# RETRY BEHAVIOR
# ============================================================================
MAX_RETRIES = 5
RETRY_INITIAL_WAIT = 2
RETRY_MAX_WAIT = 60

# ============================================================================
# CACHING
# ============================================================================
CACHE_ENABLED = True
CACHE_SIZE_LIMIT_GB = 2

# ============================================================================
# LLM CALL DEFAULTS
# ============================================================================
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4000