from config import (
    COSINE_THRESHOLD,
    FK_DELTA_THRESHOLD,
    TIME_RATIO_THRESHOLD,
    CALIBRATION_MULTIPLIER,
    RISK_BANDS
)

def classify_risk(score):
    if score <= RISK_BANDS["LOW"][1]:
        return "LOW"
    elif score <= RISK_BANDS["MODERATE"][1]:
        return "MODERATE"
    else:
        return "HIGH"

def compute_fraud_score(
    drift_score: float,
    fk_delta: float,
    time_ratio: float,
    delta_t: float,
    word_count: int,
    candidate_level: str,
    ai_probability: int = 0
):
    """
    Computes composite fraud score and handles triggers.
    Returns fraud_score (0-100) and weighted signals.
    """
    # Thresholds mapped by level
    c_thresh = COSINE_THRESHOLD[candidate_level]
    fk_thresh = FK_DELTA_THRESHOLD[candidate_level]
    t_thresh = TIME_RATIO_THRESHOLD[candidate_level]
    
    # Cosine Signal (Max 50)
    cosine_conf = 0.0
    signal_cosine = False
    if drift_score > c_thresh:
        signal_cosine = True
        cosine_conf = min((drift_score / c_thresh) * 50, 50)
        
    # Vocab Signal (Max 25)
    vocab_conf = 0.0
    signal_vocab = False
    if fk_delta > fk_thresh:
        signal_vocab = True
        vocab_conf = min((fk_delta / fk_thresh) * 25, 25)
        
    # Time Signal (Max 25)
    time_conf = 0.0
    signal_time = False
    if (delta_t < 15 and word_count > 80) or (time_ratio > t_thresh):
        signal_time = True
        time_conf = min((time_ratio / t_thresh) * 25, 25)
        if time_conf == 0:
            time_conf = 10.0 # Assign minimal confidence if triggered by short time + long words alone
            
    # AI Generation Signal (Max 40)
    ai_conf = 0.0
    signal_ai = False
    if ai_probability > 60:
        signal_ai = True
        ai_conf = min(((ai_probability - 60) / 40.0) * 40, 40)
            
    raw_score = cosine_conf + vocab_conf + time_conf + ai_conf
    multiplier = CALIBRATION_MULTIPLIER[candidate_level]
    adjusted_score = raw_score * multiplier
    
    fraud_score = min(max(round(adjusted_score, 1), 0), 100)
    
    return {
        "fraud_score": fraud_score,
        "risk_level": classify_risk(fraud_score),
        "calibration_multiplier": multiplier,
        "signals": {
            "ai_generation": {
                "probability": ai_probability,
                "threshold": 60,
                "triggered": signal_ai,
                "confidence_points": round(ai_conf, 1)
            },
            "cosine_drift": {
                "value": round(drift_score, 2),
                "threshold": c_thresh,
                "triggered": signal_cosine,
                "confidence_points": round(cosine_conf, 1)
            },
            "vocabulary_delta": {
                "delta": round(fk_delta, 2),
                "threshold": fk_thresh,
                "triggered": signal_vocab,
                "confidence_points": round(vocab_conf, 1)
            },
            "time_complexity": {
                "time_ratio": round(time_ratio, 2),
                "threshold": t_thresh,
                "triggered": signal_time,
                "confidence_points": round(time_conf, 1)
            }
        }
    }
