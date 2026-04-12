import streamlit as st
import time
import json
import numpy as np
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
except (ImportError, AttributeError):
    pass

from session_manager import init_session_state, start_interview, set_current_question, reset_session
from database import save_flag_payload, get_session_history
from llm_agent import generate_question, generate_recovery_probe, TARGET_MODEL, evaluate_answer, generate_mentor_review
from analyzer import load_embedding_model, get_text_metrics, calculate_cosine_drift, calculate_time_ratio
from scorer import compute_fraud_score
from flag_generator import generate_flag_json
from config import QUESTIONS_PER_SESSION, PROBE_TRIGGER_THRESHOLD

class FaceDetector:
    def __init__(self):
        self.face_detection = None
        self.face_cascade = None
        try:
            if 'mp_face_detection' in globals():
                self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        except Exception:
            pass
            
        if not self.face_detection:
            # Fallback to OpenCV Haar Cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces_count = 0
        
        if self.face_detection:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(img_rgb)
            faces_count = len(results.detections) if results.detections else 0
        elif self.face_cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detectMultiScale returns a tuple of rects
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces_count = len(faces)
            
        if faces_count != 1:
            cv2.putText(img, f"WARNING: {faces_count} Faces Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Face Tracking Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Configure page
st.set_page_config(page_title="HireBit - AI Interview Agent", layout="wide")

# Load heavy embedding model
embedding_model = load_embedding_model()

# Init State
init_session_state()

# --- Helpers ---
def process_answer(response_text):
    t_response = time.time()
    delta_t = t_response - st.session_state.t_question
    
    # 1. Text Metrics
    metrics = get_text_metrics(response_text)
    fk_grade = metrics["fk_score"]
    word_count = metrics["word_count"]
    jargon = metrics["jargon_candidates"]
    question_word_count = len(st.session_state.current_question.split())
    
    # 2. History processing
    mean_fk = np.mean(st.session_state.session_fk_scores) if st.session_state.session_fk_scores else fk_grade
    fk_delta = abs(fk_grade - mean_fk)
    
    # 3. Time processing
    time_results = calculate_time_ratio(word_count, question_word_count, delta_t)
    
    # 4. Cosine Drift
    drift_score, new_emb = calculate_cosine_drift(embedding_model, st.session_state.session_embeddings, response_text)
    
    # Update baseline logs
    st.session_state.session_fk_scores.append(fk_grade)
    st.session_state.session_embeddings.append(new_emb)
    
    # 5. Evaluate correctness and AI generation
    eval_result = evaluate_answer(st.session_state.job_field, st.session_state.current_question, response_text)
    ai_prob = eval_result.get("ai_generated_probability", 0)
    
    # 6. Compute Total Fraud Score
    score_result = compute_fraud_score(
        drift_score=drift_score,
        fk_delta=fk_delta,
        time_ratio=time_results["time_ratio"],
        delta_t=delta_t,
        word_count=word_count,
        candidate_level=st.session_state.candidate_level,
        ai_probability=ai_prob
    )
    
    fraud_score = score_result["fraud_score"]
    
    # Is this response for an actively generated Probe?
    is_probe_response = bool(st.session_state.pending_probe)
    probe_triggered_now = False
    
    # 6. Trigger Recovery Probe Logic
    if (not is_probe_response) and (fraud_score >= PROBE_TRIGGER_THRESHOLD):
        # We need to trigger a probe next
        probe_triggered_now = True
        st.session_state.recovery_probe_triggered = True
        
        # Select worst jargon to probe
        top_jargon = jargon[:5] if jargon else ["this concept"]
        st.session_state.pending_probe = generate_recovery_probe(top_jargon)
        
    # Generate schema
    flag_payload = generate_flag_json(
        session_id=st.session_state.session_id,
        candidate_level=st.session_state.candidate_level,
        response_index=len(st.session_state.responses) + 1,
        question_text=st.session_state.current_question,
        response_text=response_text,
        word_count=word_count,
        fk_grade=fk_grade,
        delta_t=delta_t,
        metrics_result=metrics,
        score_result=score_result,
        recovery_probe_triggered=probe_triggered_now,
        jargon_terms=jargon[:5] if probe_triggered_now else [],
        mean_fk=mean_fk,
        expected_min_time=time_results["expected_min_seconds"],
        target_model_name=TARGET_MODEL
    )
    
    if is_probe_response:
        # Update the payload with post-probe delta logic (simplified for MVP)
        # In full implementation, we'd compare this to the pre-probe payload.
        flag_payload["is_probe_response"] = True
        # Clear probe
        st.session_state.pending_probe = None
        st.session_state.recovery_probe_triggered = False

    st.session_state.flags.append(flag_payload)
    
    # Persist to database if configured
    save_flag_payload(flag_payload)
    
    # Log the full response object for UI
    st.session_state.responses.append({
        "question": st.session_state.current_question,
        "answer": response_text,
        "score_result": score_result,
        "eval_result": eval_result,
        "flag_payload": flag_payload,
        "delta_t": delta_t
    })

# --- Layout ---
tab_candidate, tab_recruiter = st.tabs(["Candidate View", "Recruiter View 🔒"])

with tab_candidate:
    st.title("HireBit Interview")
    
    total_qs = len(st.session_state.custom_questions) if hasattr(st.session_state, "custom_questions") and st.session_state.custom_questions else QUESTIONS_PER_SESSION
    
    if not st.session_state.started:
        st.write("Welcome to HireBit. To begin, please select your experience level.")
        job_field = st.text_input("Job Field / Domain (e.g. Software Engineer, Data Scientist):", value="Software Engineer")
        
        with st.expander("Recruiter Advanced Settings"):
            st.write("Provide custom questions directly (one per line). Overrides AI generation.")
            custom_qs = st.text_area("Custom Questions (optional):")
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Fresher (0-2 years)"):
                start_interview("fresher", job_field, custom_qs)
                st.rerun()
        with col2:
            if st.button("Senior (3+ years)"):
                start_interview("senior", job_field, custom_qs)
                st.rerun()
    
    elif len(st.session_state.responses) >= total_qs:
        st.success("Thank you. Your session is complete.")
        st.session_state.completed = True
        
        st.write("### End of Session - Mentor Review")
        if "mentor_review" not in st.session_state:
            with st.spinner("Compiling Mentor Feedback..."):
                review = generate_mentor_review(st.session_state.responses)
                st.session_state.mentor_review = review
        st.markdown(st.session_state.mentor_review)
        
        if st.button("Start New Session"):
            reset_session()
            st.rerun()
            
    else:
        # Screen Lock JS
        components.html("""
        <script>
            window.parent.document.addEventListener('visibilitychange', function() {
                if (window.parent.document.visibilityState === 'hidden') {
                    alert('WARNING: Screen changing detected! This violation has been logged.');
                }
            });
            window.parent.document.addEventListener('blur', function() {
                alert('WARNING: You left the interview window. This violation has been logged.');
            });
        </script>
        """, height=0)
        
        q_idx = len(st.session_state.responses) + 1
        colA, colB = st.columns([2, 1])
        
        with colA:
            st.subheader(f"Question {q_idx} of {total_qs}")
            
            # Load question
            if st.session_state.pending_probe:
                current_q = st.session_state.pending_probe
            else:
                # We generate a new question if not set for this step
                if getattr(st.session_state, 'q_idx_generated', -1) != q_idx:
                    with st.spinner("Preparing question..."):
                        if hasattr(st.session_state, "custom_questions") and st.session_state.custom_questions:
                            # Use static questions
                            if q_idx <= len(st.session_state.custom_questions):
                                current_q = st.session_state.custom_questions[q_idx - 1]
                            else:
                                current_q = "Thank you. Please hit submit to finish."
                        else:
                            # LLM dynamically generate question
                            if len(st.session_state.responses) > 0:
                                prev_q = st.session_state.responses[-1]["question"]
                                prev_a = st.session_state.responses[-1]["answer"]
                            else:
                                prev_q, prev_a = None, None
                            
                            current_q = generate_question(st.session_state.job_field, prev_q, prev_a)
                            
                        st.session_state.q_idx_generated = q_idx
                        st.session_state.cached_q = current_q
                else:
                    current_q = st.session_state.cached_q
                    
            set_current_question(current_q)
            
            st.info(current_q)
            
            # Using form to capture submit and time more accurately without triggering mid-typing
            with st.form("answer_form", clear_on_submit=True):
                answer_text = st.text_area("Your Answer:", height=200)
                submitted = st.form_submit_button("Submit Answer")
                if submitted:
                    if answer_text.strip():
                        with st.spinner("Processing..."):
                            process_answer(answer_text)
                        st.rerun()
                    else:
                        st.warning("Please provide an answer before submitting.")

        with colB:
            st.markdown("**Proctoring Camera**")
            webrtc_streamer(
                key="proctoring",
                video_processor_factory=FaceDetector,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

with tab_recruiter:
    st.title("Recruiter Dashboard")
    st.write(f"Session ID: `{st.session_state.session_id}`")
    st.write(f"Candidate Level: `{st.session_state.candidate_level}`")
    
    if st.session_state.responses:
        # Session Risk Summary
        final_score = st.session_state.responses[-1]["score_result"]["fraud_score"]
        risk = st.session_state.responses[-1]["score_result"]["risk_level"]
        
        # Calculate Employability Index
        total_correctness = 0
        valid_evals = 0
        for r in st.session_state.responses:
            if 'eval_result' in r and isinstance(r['eval_result'], dict) and 'correctness_score' in r['eval_result']:
                try:
                    total_correctness += int(r['eval_result']['correctness_score'])
                    valid_evals += 1
                except:
                    pass
                    
        avg_corr = (total_correctness / valid_evals) * 10 if valid_evals > 0 else 50
        employability_index = max(0, min(100, int(avg_corr - (final_score * 0.8))))
        
        col_risk, col_hire = st.columns(2)
        with col_risk:
            color = "green" if risk == "LOW" else "orange" if risk == "MODERATE" else "red"
            st.markdown(f"### Security Risk: <span style='color:{color}'>{risk}</span> (Score: {final_score})", unsafe_allow_html=True)
            
        with col_hire:
            hire_color = "green" if employability_index >= 70 else "orange" if employability_index >= 40 else "red"
            st.markdown(f"### Employability Index: <span style='color:{hire_color}'>{employability_index}%</span>", unsafe_allow_html=True)
            st.caption("Based on average correctness penalized by fraud score.")
        
        st.divider()

        # --- LIVE PROCTORING FEED ---
        st.subheader("🔴 Live Integrity Feed")
        proctor_data = get_session_history(st.session_state.session_id)
        proctor_events = [p for p in proctor_data if p.get("risk_level") == "PROCTOR_EVENT"]
        
        if proctor_events:
            # Stats from proctor events
            switches = len([e for e in proctor_events if "WINDOW_FOCUS" in e.get("flag", {}).get("reasoning_note", "")])
            clipboard = len([e for e in proctor_events if "CLIPBOARD" in e.get("flag", {}).get("reasoning_note", "")])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Window Tab Shifts", switches)
            c2.metric("Clipboard Events", clipboard)
            
            latest_status = proctor_events[-1]["flag"]["reasoning_note"]
            c3.markdown(f"**Latest Activity:**\n`{latest_status}`")
            
            with st.expander("View Full Violation History", expanded=True):
                for event in reversed(proctor_events):
                    ts = event.get("signals", {}).get("event", "UNKNOWN")
                    details = event.get("flag", {}).get("reasoning_note", "No details")
                    st.write(f"**{ts}**: {details}")
        else:
            st.info("Waiting for proctor badge activity...")

        st.divider()
        
        for idx, resp in enumerate(st.session_state.responses):
            flag = resp["flag_payload"]
            is_probe_trigger = flag["flag"]["triggered"]
            
            with st.expander(f"Response #{idx+1} — Score: {flag['fraud_score']} ({flag['risk_level']})"):
                st.markdown(f"**Q:** {resp['question']}")
                if flag.get("is_probe_response"):
                    st.caption("🟠 **RECOVERY PROBE RESPONSE**")
                    
                st.markdown(f"**A:** {resp['answer']}")
                st.markdown("---")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Delivery Time", f"{resp['delta_t']:.1f}s")
                col2.metric("Word Count", flag["response_metadata"]["word_count"])
                col3.metric("FK Grade", flag["response_metadata"]["flesch_kincaid_grade"])
                col4.metric("Correct", f"{resp.get('eval_result', {}).get('correctness_score', 'N/A')}/10")
                
                ai_prob = flag["signals"].get("ai_generation", {}).get("probability", "N/A")
                col5.metric("AI Prob", f"{ai_prob}%")
                
                st.caption(f"**Evaluation:** {resp.get('eval_result', {}).get('reasoning', '')}")
                ai_reasoning = resp.get('eval_result', {}).get('ai_detection_reasoning', '')
                if ai_reasoning and ai_reasoning != "N/A":
                    st.caption(f"**AI Detection:** {ai_reasoning}")
                
                if flag["fraud_score"] >= PROBE_TRIGGER_THRESHOLD:
                    st.warning(f"**FLAGGED:** {flag['flag']['reasoning_note']}")
                else:
                    st.success("Normal Baseline")
                    
                if is_probe_trigger:
                    st.error("🚨 RECOVERY PROBE TRIGGERED 🚨")
                    st.write(f"**Extracted Jargon:** {', '.join(flag['recovery_probe']['jargon_terms_extracted'])}")
                    
        st.divider()
        st.subheader("Audit JSON Export")
        st.json(st.session_state.flags)
    else:
        st.info("No responses recorded yet.")
