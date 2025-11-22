from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

app = FastAPI()

# --- CONFIGURATION & SECRETS ---
# These are loaded from the server's environment variables
GROK_API_KEY = os.getenv("GROK_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # The secret admin key

# Validation
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("CRITICAL WARNING: Supabase credentials not found.")

# Initialize Supabase Client with Admin privileges
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    print(f"Failed to init Supabase: {e}")

# --- DATA MODELS (Pydantic) ---
# These define what data the Flutter app sends to us

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
    """Health check endpoint to ensure server is running."""
    return {"status": "Audio Vibe Backend Active"}

# 1. PROXY: Get All Tracks
@app.get("/tracks")
async def get_tracks():
    try:
        # Select all tracks, ordered by newest first
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase Error: {str(e)}")

# 2. PROXY: Add New Track (Secure Admin Action)
@app.post("/tracks")
async def add_track(track: TrackUpload):
    try:
        # Convert Pydantic model to dictionary
        data = track.dict()
        # Insert into Supabase
        response = supabase.table("music_tracks").insert(data).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert Failed: {str(e)}")

# 3. PROXY: Record Play Stats
@app.post("/record-play")
async def record_play(stat: PlayRecord):
    try:
        # Calculate completion rate logic here to keep Flutter logic thin
        completion_rate = 0.0
        if stat.total_duration_ms > 0:
            completion_rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms)

        # Call the RPC function defined in Supabase SQL
        response = supabase.rpc('upsert_listening_stat', {
            'p_user_id': stat.user_id,
            'p_track_id': stat.track_id,
            'p_listen_time_ms': stat.listen_time_ms,
            'p_completion_rate': completion_rate
        }).execute()
        
        return {"status": "recorded"}
    except Exception as e:
        # Log error but don't crash app flow for stats
        print(f"Stats Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. AI ENRICHMENT (Grok Proxy)
@app.post("/enrich-metadata")
async def enrich_metadata(song: SongRequest):
    if not GROK_API_KEY:
        raise HTTPException(status_code=500, detail="Server Error: Grok Key missing")

    url = "https://api.x.ai/v1/chat/completions"
    
    # Strict prompt for formatting
    prompt = (
        f"Analyze the song '{song.title}' by '{song.artist}'. "
        "Return ONLY a single string in this exact format: "
        "'Mood;Language;Genre'. "
        "Use standard moods: Romantic, Sad, Party, Chill, Intense, Happy. "
        "Example response: 'Party;Punjabi;Bhangra'. "
        "No intro text."
    )

    payload = {
        "messages": [
            {"role": "system", "content": "You are a music metadata expert."},
            {"role": "user", "content": prompt}
        ],
        "model": "grok-beta",
        "stream": False,
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            # Parse Content
            content = data['choices'][0]['message']['content'].strip()
            parts = content.split(';')
            
            # Handle potential AI formatting errors gracefully
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
            print(f"AI Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
