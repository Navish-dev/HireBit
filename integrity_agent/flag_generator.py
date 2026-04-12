from datetime import datetime, timezone
import json
from config import EMBEDDING_MODEL

def generate_flag_json(
    session_id: str,
    candidate_level: str,
    response_index: int,
    question_text: str,
    response_text: str,
    word_count: int,
    fk_grade: float,
    delta_t: float,
    metrics_result: dict,
    score_result: dict,
    recovery_probe_triggered: bool,
    jargon_terms: list,
    mean_fk: float,
    expected_min_time: float,
    target_model_name: str
):
    """
    Format the complete analysis into the standard JSON explainability artifact.
    """
    
    # Determine reasoning note based on triggered signals
    reasoning_parts = []
    signals = score_result["signals"]
    
    if signals["vocabulary_delta"]["triggered"]:
        reasoning_parts.append(f"FK-delta of +{signals['vocabulary_delta']['delta']}")
        
    if signals["time_complexity"]["triggered"]:
        reasoning_parts.append(f"delivered in {delta_t:.1f}s (ratio {signals['time_complexity']['time_ratio']}x below minimum)")
        
    if signals["cosine_drift"]["triggered"]:
        reasoning_parts.append(f"cosine drift of {signals['cosine_drift']['value']} vs. session mean")
        
    reasoning_note = " and ".join(reasoning_parts)
    if not reasoning_note:
        reasoning_note = "Normal baseline."
    else:
        reasoning_note = f"{word_count}-word Grade-{fk_grade:.1f} response showing " + reasoning_note + " co-fired."

    flag_payload = {
        "session_id": session_id,
        "candidate_level": candidate_level,
        "response_index": response_index,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "response_metadata": {
            "question_text": question_text,
            "response_text": response_text[:200] + "..." if len(response_text)>200 else response_text,
            "word_count": word_count,
            "flesch_kincaid_grade": round(fk_grade, 1),
            "delta_t_seconds": round(delta_t, 1),
            "words_per_second": round(word_count / delta_t, 1) if delta_t > 0 else float('inf')
        },
        "signals": {
            "cosine_drift": signals["cosine_drift"],
            "vocabulary_delta": {
                "current_fk": round(fk_grade, 1),
                "session_mean_fk": round(mean_fk, 1),
                **signals["vocabulary_delta"]
            },
            "time_complexity": {
                "delta_t_seconds": round(delta_t, 1),
                "expected_min_seconds": round(expected_min_time, 1),
                **signals["time_complexity"]
            }
        },
        "fraud_score": score_result["fraud_score"],
        "risk_level": score_result["risk_level"],
        "calibration_multiplier": score_result["calibration_multiplier"],
        "flag": {
            "triggered": recovery_probe_triggered,
            "reasoning_note": reasoning_note,
            "recommended_action": "HUMAN_REVIEW_REQUIRED" if recovery_probe_triggered else "NONE",
            "auto_reject": False
        },
        "recovery_probe": {
            "triggered": recovery_probe_triggered,
            "jargon_terms_extracted": jargon_terms,
            "probe_question": None, # Populated later once LLM generates it
            "probe_response_index": response_index + 1 if recovery_probe_triggered else None,
            "post_probe_fraud_score": None,
            "probe_delta": None,
            "probe_verdict": "PENDING" if recovery_probe_triggered else "NONE"
        },
        "audit": {
            "model_used": target_model_name,
            "embedding_model": EMBEDDING_MODEL,
            "schema_version": "1.0.0"
        }
    }
    
    return flag_payload
