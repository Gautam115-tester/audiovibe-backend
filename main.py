from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="AudioVibe Backend API", version="5.1.0 (Safe-Startup)")

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Global State for Clients
supabase: Optional[Client] = None
groq_client: Optional[Groq] = None
startup_error: Optional[str] = None

# 4. Safe Initialization (Prevents Crash on Boot)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

try:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase Credentials Missing in Environment Variables")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY Missing in Environment Variables")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ System Online: Clients Initialized")

except Exception as e:
    startup_error = str(e)
    print(f"❌ Startup Warning: {e}")
    print("Server will start in Maintenance Mode (Health Check Only).")

# 5. Config
EXPECTED_INTEGRITY_HEADER = 'clean-device-v1'
GROQ_MODELS = {"primary": "llama-3.3-70b-versatile"}
metadata_cache = {}

# 6. Models
class SongRequest(BaseModel):
    artist: str
    title: str
    album: Optional[str] = None

class TrackUpload(BaseModel):
    title: str
    artist: str
    album: str
    audio_file_url: str
    cover_image_url: Optional[str] = None
    duration_ms: int
    genres: List[str]
    tier_required: str = "free"

class PlayRecord(BaseModel):
    user_id: str
    track_id: str
    listen_time_ms: int
    total_duration_ms: int

# 7. Helper: Check Backend Readiness
def ensure_ready():
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Backend Not Ready: {startup_error}")
    if not supabase or not groq_client:
        raise HTTPException(status_code=503, detail="Backend Not Ready: Clients not initialized")

# 8. Endpoints
@app.get("/")
def read_root():
    if startup_error:
        return {"status": "Maintenance Mode", "error": startup_error}
    return {"status": "Active", "version": "5.1.0"}

@app.get("/health")
def health_check():
    """
    DEBUG ENDPOINT: Visit this to see why 'Login' or 'Internal' errors are happening.
    """
    return {
        "status": "unhealthy" if startup_error else "healthy",
        "startup_error": startup_error,
        "env_vars": {
            "GROQ_KEY_SET": bool(GROQ_API_KEY),
            "SUPABASE_URL_SET": bool(SUPABASE_URL),
            "SUPABASE_KEY_SET": bool(SUPABASE_SERVICE_ROLE_KEY)
        }
    }

@app.get("/tracks")
async def get_tracks(x_app_integrity: Optional[str] = Header(None)):
    ensure_ready()
    try:
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tracks")
async def add_track(track: TrackUpload, x_app_integrity: Optional[str] = Header(None)):
    ensure_ready()
    try:
        data = track.dict()
        if not data.get('tier_required'): data['tier_required'] = 'free'
        response = supabase.table("music_tracks").insert(data).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record-play")
async def record_play(stat: PlayRecord, x_app_integrity: Optional[str] = Header(None)):
    ensure_ready()
    try:
        completion_rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms) if stat.total_duration_ms > 0 else 0
        supabase.rpc('upsert_listening_stat', {
            'p_user_id': stat.user_id, 'p_track_id': stat.track_id, 
            'p_listen_time_ms': stat.listen_time_ms, 'p_completion_rate': completion_rate
        }).execute()
        return {"status": "recorded"}
    except Exception:
        return {"status": "error"}

@app.post("/enrich-metadata")
async def enrich_metadata(song: SongRequest, x_app_integrity: Optional[str] = Header(None)):
    ensure_ready()
    
    clean_title = song.title.strip()
    clean_artist = song.artist.strip()
    clean_album = song.album.strip() if song.album else ""
    
    cache_key = f"{clean_artist.lower()}:{clean_title.lower()}"
    if cache_key in metadata_cache:
        return metadata_cache[cache_key]

    if clean_album and clean_album.lower() != "unknown":
        search_query = f"Track: '{clean_title}' from Album/Movie: '{clean_album}' by Artist: '{clean_artist}'"
    else:
        search_query = f"Track: '{clean_title}' by Artist: '{clean_artist}'"

    print(f"🔍 Analyzing: {search_query}")

    # --- LAYMAN PROMPT ---
    prompt = (
        f"Analyze this music track: {search_query}.\n\n"
        "TASK 1: MOOD (Verification Matrix)\n"
        "   - High Energy + Angry/War -> 'Aggressive'\n"
        "   - High Energy + Happy/Dance -> 'Energetic'\n"
        "   - Low Energy + Love -> 'Romantic'\n"
        "   - Low Energy + Sad -> 'Melancholic'\n"
        "   - Spiritual -> 'Spiritual'\n\n"
        
        "TASK 2: INDUSTRY (Source)\n"
        "   - India (Hindi) -> 'Bollywood'\n"
        "   - India (South) -> 'Tollywood'\n"
        "   - India (Punjab) -> 'Punjabi'\n"
        "   - Western -> 'International'\n"
        "   - Independent -> 'Indie'\n\n"
        
        "TASK 3: GENRE (SIMPLIFY FOR LAYMAN)\n"
        "   - 'Party', 'Pop', 'Rock', 'Hip-Hop', 'Folk', 'Devotional', 'Classical', 'LoFi'\n\n"
        
        "REQUIRED OUTPUT (JSON):\n"
        "{\n"
        '  "mood": "One word from Task 1",\n'
        '  "language": "Primary language",\n'
        '  "industry": "One word from Task 2",\n'
        '  "genre": "One word from Task 3",\n'
        '  "reasoning": "Brief explanation"\n'
        "}"
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a music classifier. Output strict JSON."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"}
        )

        content = chat_completion.choices[0].message.content.strip()
        
        try:
            data = json.loads(content)
            mood = data.get("mood", "Neutral").title()
            language = data.get("language", "Unknown").title()
            industry = data.get("industry", "International").title()
            genre = data.get("genre", "Pop").title()
            
            reasoning = data.get("reasoning", "").lower()
            if "fight" in reasoning or "war" in reasoning or "aggressive" in reasoning:
                if mood == "Romantic": mood = "Aggressive"

        except json.JSONDecodeError:
            mood, language, industry, genre = "Neutral", "Unknown", "International", "Pop"

        result = {
            "formatted": f"{mood};{language};{industry};{genre}",
            "mood": mood,
            "language": language,
            "industry": industry,
            "genre": genre
        }
        
        metadata_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"Error: {e}")
        return {
            "formatted": "Neutral;Unknown;International;Pop",
            "mood": "Neutral",
            "language": "Unknown",
            "industry": "International",
            "genre": "Pop"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
