from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="AudioVibe Backend API", version="1.0.0")

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION & SECRETS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Validation
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("⚠️ CRITICAL: Supabase credentials missing.")
    raise RuntimeError("Supabase configuration missing")

if not GROQ_API_KEY:
    print("⚠️ CRITICAL: GROQ_API_KEY missing.")
    raise RuntimeError("Groq API key missing")

# Initialize Clients
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("✅ Supabase client initialized")
except Exception as e:
    print(f"❌ Failed to init Supabase: {e}")
    raise

# Initialize Groq Client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq client initialized")
except Exception as e:
    print(f"❌ Failed to init Groq: {e}")
    raise

# --- SECURITY CONSTANTS ---
EXPECTED_INTEGRITY_HEADER = 'clean-device-v1'

# ✅ UPDATED: Current supported Groq models
GROQ_MODELS = {
    "primary": "llama-3.3-70b-versatile",  # Best quality
    "fast": "llama-3.1-8b-instant",        # Faster
    "fallback": "mixtral-8x7b-32768"       # Alternative
}

# --- DATA MODELS ---
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

# --- ROUTES ---

@app.get("/")
def read_root():
    return {
        "status": "AudioVibe Backend Active (Groq Edition)",
        "version": "1.0.0",
        "groq_model": GROQ_MODELS["primary"],
        "endpoints": {
            "health": "GET /health",
            "enrich": "POST /enrich-metadata",
            "tracks": "GET /tracks",
            "add_track": "POST /tracks",
            "record_play": "POST /record-play",
            "test_groq": "GET /test-groq"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "supabase_configured": bool(SUPABASE_URL),
        "model": GROQ_MODELS["primary"]
    }

# ✅ Test Groq Connection
@app.get("/test-groq")
async def test_groq():
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say 'Hello from Groq!'"}
            ],
            model=GROQ_MODELS["primary"],
            max_tokens=20,
        )
        
        response = chat_completion.choices[0].message.content.strip()
        
        return {
            "success": True,
            "model": GROQ_MODELS["primary"],
            "response": response
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# 1. PROXY: Get All Tracks
@app.get("/tracks")
async def get_tracks(x_app_integrity: Optional[str] = Header(None)):
    # ✅ Made security check optional for development
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header: {x_app_integrity}")
    
    try:
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"❌ Error fetching tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Supabase Error: {str(e)}")

# 2. PROXY: Add New Track
@app.post("/tracks")
async def add_track(track: TrackUpload, x_app_integrity: Optional[str] = Header(None)):
    # ✅ Made security check optional for development
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header: {x_app_integrity}")
    
    try:
        data = track.dict()
        print(f"📤 Inserting track: {data['title']} by {data['artist']}")
        print(f"   Genres: {data['genres']}")
        print(f"   Tier: {data['tier_required']}")
        
        response = supabase.table("music_tracks").insert(data).execute()
        
        print(f"✅ Track inserted successfully: {response.data}")
        return {"status": "success", "data": response.data}
    except Exception as e:
        print(f"❌ Insert Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insert Failed: {str(e)}")

# 3. PROXY: Record Play Stats
@app.post("/record-play")
async def record_play(stat: PlayRecord, x_app_integrity: Optional[str] = Header(None)):
    # ✅ Made security check optional for development
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header: {x_app_integrity}")
    
    try:
        completion_rate = 0.0
        if stat.total_duration_ms > 0:
            completion_rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms)

        # Check if RPC function exists, otherwise use direct insert
        try:
            response = supabase.rpc('upsert_listening_stat', {
                'p_user_id': stat.user_id,
                'p_track_id': stat.track_id,
                'p_listen_time_ms': stat.listen_time_ms,
                'p_completion_rate': completion_rate
            }).execute()
        except Exception as rpc_error:
            print(f"⚠️ RPC failed, using direct insert: {rpc_error}")
            # Fallback to direct insert
            response = supabase.table("listening_stats").insert({
                "user_id": stat.user_id,
                "track_id": stat.track_id,
                "listen_time_ms": stat.listen_time_ms,
                "total_duration_ms": stat.total_duration_ms,
                "completed": completion_rate >= 0.9
            }).execute()
        
        print(f"📊 Stats recorded for track {stat.track_id}")
        return {"status": "recorded", "completion_rate": completion_rate}
    except Exception as e:
        print(f"❌ Stats Error: {e}")
        return {"status": "error", "detail": str(e)}

# 4. AI ENRICHMENT (Using Groq with UPDATED MODEL)
@app.post("/enrich-metadata")
async def enrich_metadata(song: SongRequest, x_app_integrity: Optional[str] = Header(None)):
    # ✅ Made security check optional for development
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header: {x_app_integrity}")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Server Error: GROQ_API_KEY missing")

    print(f"🎵 Enrichment request: '{song.title}' by '{song.artist}'")

    album_info = f" from album '{song.album}'" if song.album else ""
    
    prompt = (
        f"Analyze the song '{song.title}' by '{song.artist}'{album_info}. "
        "Provide ONLY these three items separated by semicolons:\n"
        "1. Primary mood (one word: e.g., Energetic, Melancholic, Upbeat, Chill, Aggressive, Romantic, Dark, Peaceful)\n"
        "2. Primary language (e.g., English, Spanish, Hindi, Korean, Japanese, French)\n"
        "3. Primary genre (e.g., Rock, Pop, Hip-Hop, Classical, Jazz, Electronic, Metal, R&B, Country)\n\n"
        "Format: Mood;Language;Genre\n"
        "Example: Energetic;English;Rock\n\n"
        "Respond with ONLY the three items separated by semicolons, nothing else:"
    )

    try:
        # ✅ Call Groq API with UPDATED MODEL
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a music analysis expert. Respond ONLY with the format requested, no explanation."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            model=GROQ_MODELS["primary"],  # ✅ FIXED: Use updated model
            temperature=0.3,
            max_tokens=50,
        )

        content = chat_completion.choices[0].message.content.strip()
        print(f"📦 Groq response: '{content}'")
        
        parts = content.split(';')
        
        # Validate and clean response
        if len(parts) < 3:
            print(f"⚠️ Incomplete response from Groq: {content}")
            return {
                "formatted": "Neutral;Unknown;General",
                "mood": "Neutral",
                "language": "Unknown",
                "genre": "General"
            }

        mood = parts[0].strip() or "Neutral"
        language = parts[1].strip() or "Unknown"
        genre = parts[2].strip() or "General"

        result = {
            "formatted": f"{mood};{language};{genre}",
            "mood": mood,
            "language": language,
            "genre": genre
        }
        
        print(f"✅ Returning: {result}")
        return result

    except Exception as e:
        print(f"❌ Groq AI Error: {e}")
        # Return fallback instead of error
        return {
            "formatted": "Neutral;Unknown;General",
            "mood": "Neutral",
            "language": "Unknown",
            "genre": "General",
            "error": str(e)
        }

# ✅ Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"❌ Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    print("═══════════════════════════════════════")
    print("🚀 AudioVibe Backend Server (Python)")
    print(f"🤖 Groq Model: {GROQ_MODELS['primary']}")
    print(f"✅ Groq API: {'Configured' if GROQ_API_KEY else '❌ Missing'}")
    print(f"✅ Supabase: {'Configured' if SUPABASE_URL else '❌ Missing'}")
    print("═══════════════════════════════════════")
    uvicorn.run(app, host="0.0.0.0", port=8000)
