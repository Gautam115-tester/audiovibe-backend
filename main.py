from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq  # ✅ Import Groq

# 1. Load Environment Variables
load_dotenv()

app = FastAPI()

# --- CONFIGURATION & SECRETS ---
# ✅ Load GROQ_API_KEY instead of GROK_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Validation
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("⚠️ CRITICAL: Supabase credentials missing.")

if not GROQ_API_KEY:
    print("⚠️ CRITICAL: GROQ_API_KEY missing.")

# Initialize Clients
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    print(f"⚠️ Failed to init Supabase: {e}")

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# --- SECURITY CONSTANTS ---
EXPECTED_INTEGRITY_HEADER = 'clean-device-v1'

# --- DATA MODELS ---
class SongRequest(BaseModel):
    artist: str
    title: str

class TrackUpload(BaseModel):
    title: str
    artist: str
    album: str
    audio_file_url: str
    cover_image_url: str
    duration_ms: int
    genres: List[str]
    tier_required: str

class PlayRecord(BaseModel):
    user_id: str
    track_id: str
    listen_time_ms: int
    total_duration_ms: int

# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "Audio Vibe Backend Active (Groq Edition)"}

# 1. PROXY: Get All Tracks
@app.get("/tracks")
async def get_tracks(x_app_integrity: str = Header(None)):
    if x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        raise HTTPException(status_code=403, detail="Security Check Failed")
    try:
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase Error: {str(e)}")

# 2. PROXY: Add New Track
@app.post("/tracks")
async def add_track(track: TrackUpload, x_app_integrity: str = Header(None)):
    if x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        raise HTTPException(status_code=403, detail="Security Check Failed")
    try:
        data = track.dict()
        response = supabase.table("music_tracks").insert(data).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert Failed: {str(e)}")

# 3. PROXY: Record Play Stats
@app.post("/record-play")
async def record_play(stat: PlayRecord, x_app_integrity: str = Header(None)):
    if x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        raise HTTPException(status_code=403, detail="Security Check Failed")
    try:
        completion_rate = 0.0
        if stat.total_duration_ms > 0:
            completion_rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms)

        response = supabase.rpc('upsert_listening_stat', {
            'p_user_id': stat.user_id,
            'p_track_id': stat.track_id,
            'p_listen_time_ms': stat.listen_time_ms,
            'p_completion_rate': completion_rate
        }).execute()
        
        return {"status": "recorded"}
    except Exception as e:
        print(f"Stats Error: {e}")
        return {"status": "error", "detail": str(e)}

# 4. AI ENRICHMENT (Using Groq)
@app.post("/enrich-metadata")
async def enrich_metadata(song: SongRequest, x_app_integrity: str = Header(None)):
    if x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        raise HTTPException(status_code=403, detail="Security Check Failed")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server Error: GROQ_API_KEY missing")

    prompt = (
        f"Analyze the song '{song.title}' by '{song.artist}'. "
        "Return ONLY a single string in this exact format: "
        "'Mood;Language;Genre'. "
        "Example: 'Energetic;English;Pop'. "
        "No intro text."
    )

    try:
        # ✅ Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a music metadata expert. Return only the requested format."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192", # Fast and efficient model
            temperature=0.3,
        )

        content = chat_completion.choices[0].message.content.strip()
        parts = content.split(';')
        
        if len(parts) < 3:
            return {
                "formatted": content, 
                "mood": "Unknown", 
                "language": "Unknown", 
                "genre": "Unknown"
            }

        return {
            "formatted": content,
            "mood": parts[0].strip(),
            "language": parts[1].strip(),
            "genre": parts[2].strip()
        }

    except Exception as e:
        print(f"Groq AI Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
