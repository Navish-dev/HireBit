# config.py

# === Calibration Thresholds ===
COSINE_THRESHOLD = {
    "fresher": 0.45,
    "senior": 0.30
}

FK_DELTA_THRESHOLD = {
    "fresher": 2.5,
    "senior": 1.5
}

TIME_RATIO_THRESHOLD = {
    "fresher": 5.0,    # More lenient — freshers may copy-paste for formatting
    "senior": 3.0
}

CALIBRATION_MULTIPLIER = {
    "fresher": 0.80,
    "senior": 1.00
}

# === Score Bands ===
RISK_BANDS = {
    "LOW": (0, 29),
    "MODERATE": (30, 59),
    "HIGH": (60, 100)
}

PROBE_TRIGGER_THRESHOLD = 30  # Minimum fraud_score to fire Recovery Probe

# === LLM Config ===
MAX_TOKENS_QUESTION = 200
MAX_TOKENS_PROBE = 150
TEMPERATURE = 0.4  # Low temp = deterministic, professional tone

# === Embedding Config ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# === Session Config ===
QUESTIONS_PER_SESSION = 5
MIN_RESPONSES_FOR_BASELINE = 2  # Cosine drift not computed before this
