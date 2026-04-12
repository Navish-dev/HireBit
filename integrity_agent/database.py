import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

def get_supabase_client() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Failed to initialize Supabase client: {e}")
        return None

def save_flag_payload(payload: dict):
    """
    Saves the audit JSON payload to the Supabase database.
    Expects a table named 'interview_flags'.
    """
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        # We assume the user creates a table 'interview_flags' in their Supabase project
        # containing at least a 'session_id' text column and a 'payload' jsonb column.
        data, count = supabase.table('interview_flags').insert({
            "session_id": payload["session_id"],
            "fraud_score": payload["fraud_score"],
            "risk_level": payload["risk_level"],
            "payload": payload
        }).execute()
        return True
    except Exception as e:
        print(f"Database save error: {e}")
        return False

def get_session_history(session_id: str):
    """
    Retrieves all payloads for a specific session.
    """
    supabase = get_supabase_client()
    if not supabase:
        return []
        
    try:
        response = supabase.table('interview_flags').select("*").eq('session_id', session_id).order('created_at', desc=False).execute()
        return [row['payload'] for row in response.data] if response.data else []
    except Exception as e:
        print(f"Database fetch error: {e}")
        return []
