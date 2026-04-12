import os
from litellm import completion
from dotenv import load_dotenv
from config import MAX_TOKENS_QUESTION, MAX_TOKENS_PROBE, TEMPERATURE

# Load environment variables (API keys)
load_dotenv()

# Determine which model to use
TARGET_MODEL = os.getenv("TARGET_MODEL", "openai/gpt-4o")

def generate_question(role="Software Engineer", previous_q=None, previous_a=None):
    """
    Generates a generic domain question, acting conversational if context is provided.
    """
    if previous_q and previous_a:
        sys_prompt = f"You are a technical interviewer for a {role} position. The candidate just answered your previous question.\nPrevious Q: {previous_q}\nCandidate Answer: {previous_a}\n\nFirst, briefly validate, acknowledge, or gently course-correct their answer in 1-2 sentences. Then, smoothly transition and ask your NEXT conceptual interview question. Do not include standard greetings."
    else:
        sys_prompt = f"You are a technical interviewer for a {role} position. Ask ONE clear, conceptual interview question. Do not include greetings or preamble."
        
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "Proceed with the interview process."}
    ]
    
    response = completion(
        model=TARGET_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS_QUESTION + 100, # extra tokens for conversational buffer
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content.strip()

def generate_recovery_probe(jargon_terms):
    """
    Generates a dynamic recovery probe based on extracted jargon.
    """
    terms_str = ", ".join(jargon_terms)
    
    prompt = f"""You are an elite technical interviewer. 
The candidate just used these specific terms in their answer: [{terms_str}].

Generate exactly ONE follow-up question that:
(a) references one of these terms directly by name,
(b) asks for a concrete, specific implementation detail that cannot be answered without real understanding,
(c) cannot be answered by rephrasing the original term.

Output ONLY the question. No preamble."""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate the follow-up question."}
    ]
    
    response = completion(
        model=TARGET_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS_PROBE,
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content.strip()

def evaluate_answer(role, question, answer):
    import json
    prompt = f"""You are an expert {role}. 
Evaluate the candidate's answer to this question.
Question: {question}
Answer: {answer}

Respond ONLY with valid JSON containing:
"correctness_score": an integer from 0 to 10
"reasoning": a concise 1-sentence explanation of why
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        response = completion(
            model=TARGET_MODEL,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # Fallback if json parsing fails or model doesn't support it strictly
        print(f"Eval Error: {e}")
        return {"correctness_score": -1, "reasoning": "Evaluation failed or unsupported."}

def generate_mentor_review(qa_history):
    history_str = ""
    for i, qa in enumerate(qa_history):
        history_str += f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}\n\n"
        
    prompt = f"""You are a senior mentor giving final feedback after an interview.
Here is the transcript:
{history_str}
Write a constructive, encouraging review directly to the candidate. 
Highlight their strengths, give specific actionable tips on what they missed or could improve, and give a supportive closing. Use Markdown formatting."""
    messages = [{"role": "system", "content": "You are a helpful mentor."}, {"role": "user", "content": prompt}]
    
    response = completion(
        model=TARGET_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()
