import textstat
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer model globally for Streamlit."""
    return SentenceTransformer(EMBEDDING_MODEL)

def get_text_metrics(text: str):
    """Calculates readability and length metrics."""
    fk_score = textstat.flesch_kincaid_grade(text)
    word_count = textstat.lexicon_count(text)
    
    # Extract hard jargon (difficult words usually represent technical terms if not standard English vocabulary)
    # textstat's difficult words logic isn't perfect for tech, but it serves as a good proxy for the MVP.
    # Alternatively we could just split long words.
    # We will use it as a proxy for jargon.
    # textstat doesn't directly expose the list of difficult words, just the count via textstat.difficult_words(text)
    # So we'll implement a simple heuristic to find complex words (length >= 8) as 'jargon' fallback for MVP.
    words = text.replace(",", "").replace(".", "").split()
    long_words = [w for w in set(words) if len(w) >= 8]
    
    return {
        "fk_score": fk_score,
        "word_count": word_count,
        "jargon_candidates": long_words
    }

def calculate_cosine_drift(model, session_embeddings, new_text: str):
    """
    Computes cosine similarity of new_text against the session mean embedding.
    Returns the drift score (1.0 - similarity) and the new embedding.
    """
    new_emb = model.encode([new_text]) # shape (1, dim)
    
    if len(session_embeddings) < 2:
        return 0.0, new_emb
    
    # Stack into an (N, dim) array to avoid 3D array creation
    stacked_embeddings = np.vstack(session_embeddings)
    session_mean = np.mean(stacked_embeddings, axis=0, keepdims=True)
    
    similarity = cosine_similarity(new_emb, session_mean)[0][0]
    
    drift_score = float(1.0 - similarity)
    return drift_score, new_emb

def calculate_time_ratio(word_count: int, question_word_count: int, delta_t: float):
    """
    Calculates time-to-complexity ratio to flag copy-pasting.
    Assumes avg human types 40 WPM = 0.67 words/sec.
    Reading time = 15 seconds per 100 words of question.
    """
    expected_min_time = (word_count / 0.67) + (question_word_count / (100 / 15))
    time_ratio = expected_min_time / max(delta_t, 1.0)
    
    return {
        "expected_min_seconds": expected_min_time,
        "time_ratio": time_ratio
    }
