import streamlit as st
import time

def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess_{int(time.time())}"
        st.session_state.candidate_level = None
        st.session_state.job_field = None
        st.session_state.started = False
        st.session_state.responses = []  # list of dicts for each Q&A
        st.session_state.session_embeddings = []  # rolling embeddings
        st.session_state.session_fk_scores = []   # rolling FK
        
        st.session_state.current_question = None
        st.session_state.t_question = None
        st.session_state.recovery_probe_triggered = False
        st.session_state.pending_probe = None
        st.session_state.flags = [] # Audit JSONs
        st.session_state.completed = False
        
        # New Integrity Trackers
        st.session_state.screen_violations = 0
        st.session_state.camera_violations = 0
        
def start_interview(level, job_field="Software Engineer"):
    st.session_state.candidate_level = level
    st.session_state.job_field = job_field
    st.session_state.started = True
    
def set_current_question(question_text):
    # Only update timestamp when question changes
    if st.session_state.current_question != question_text:
        st.session_state.current_question = question_text
        st.session_state.t_question = time.time()

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
