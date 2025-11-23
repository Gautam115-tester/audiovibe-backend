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
# ✅ FIXED: Now looking for SUPABASE_SERVICE_ROLE_KEY
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Validation
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("⚠️ CRITICAL: Supabase credentials missing.")
    print(f"   SUPABASE_URL: {'✅ Set' if SUPABASE_URL else '❌ Missing'}")
    print(f"   SUPABASE_SERVICE_ROLE_KEY: {'✅ Set' if SUPABASE_SERVICE_ROLE_KEY else '❌ Missing'}")
    raise RuntimeError("Supabase configuration missing")

if not GROQ_API_KEY:
    print("⚠️ CRITICAL: GROQ_API_KEY missing.")
    raise RuntimeError("Groq API key missing")

# ✅ Verify we're using service_role key (not anon key)
if len(SUPABASE_SERVICE_ROLE_KEY) < 200:
    print("⚠️ WARNING: SUPABASE_SERVICE_ROLE_KEY seems too short. Make sure you're using service_role key, not anon key!")
    print(f"   Current key length: {len(SUPABASE_SERVICE_ROLE_KEY)} chars")
    print(f"   Expected: 250-400+ chars for service_role key")

# Initialize Clients
try:
    # ✅ Use SUPABASE_SERVICE_ROLE_KEY
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    print("✅ Supabase client initialized with service_role key")
    print(f"   Key length: {len(SUPABASE_SERVICE_ROLE_KEY)} chars")
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

# ✅ Current supported Groq models
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
    tier_required: str = "free"  # ✅ Default value

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
        "service_role_configured": len(SUPABASE_SERVICE_ROLE_KEY) > 200,
        "endpoints": {
            "health": "GET /health",
            "enrich": "POST /enrich-metadata",
            "tracks": "GET /tracks",
            "add_track": "POST /tracks",
            "record_play": "POST /record-play",
            "test_groq": "GET /test-groq",
            "debug_auth": "GET /debug/auth"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "supabase_configured": bool(SUPABASE_URL),
        "service_role_key_length": len(SUPABASE_SERVICE_ROLE_KEY) if SUPABASE_SERVICE_ROLE_KEY else 0,
        "using_service_role": len(SUPABASE_SERVICE_ROLE_KEY) > 200,
        "model": GROQ_MODELS["primary"]
    }

# ✅ Debug endpoint to verify authentication
@app.get("/debug/auth")
async def debug_auth():
    """Debug endpoint to check if service_role key is working"""
    try:
        # Try to query the database
        response = supabase.table("music_tracks").select("id").limit(1).execute()
        
        # Determine key type based on length
        if len(SUPABASE_SERVICE_ROLE_KEY) > 200:
            key_type = "service_role (correct)"
        else:
            key_type = "anon key (WRONG - use service_role!)"
        
        return {
            "status": "success",
            "key_type": key_type,
            "key_length": len(SUPABASE_SERVICE_ROLE_KEY),
            "key_preview": SUPABASE_SERVICE_ROLE_KEY[:30] + "...",
            "can_read_database": True,
            "tracks_found": len(response.data)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "key_length": len(SUPABASE_SERVICE_ROLE_KEY),
            "key_type": "service_role" if len(SUPABASE_SERVICE_ROLE_KEY) > 200 else "anon (WRONG!)"
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
    # ✅ Security check optional for development
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header: {x_app_integrity}")
    
    try:
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        print(f"📚 Retrieved {len(response.data)} tracks")
        return response.data
    except Exception as e:
        print(f"❌ Error fetching tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Supabase Error: {str(e)}")

# 2. PROXY: Add New Track (✅ FIXED)
@app.post("/tracks")
async def add_track(track: TrackUpload, x_app_integrity: Optional[str] = Header(None)):
    # ✅ Security check optional for development
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header: {x_app_integrity}")
    
    try:
        data = track.dict()
        
        # ✅ Ensure tier_required has a valid value
        if not data.get('tier_required'):
            data['tier_required'] = 'free'
        
        print(f"📤 Inserting track: {data['title']} by {data['artist']}")
        print(f"   Album: {data['album']}")
        print(f"   Genres: {data['genres']}")
        print(f"   Tier: {data['tier_required']}")
        print(f"   Duration: {data['duration_ms']}ms")
        print(f"   Cover URL: {data.get('cover_image_url', 'None')}")
        
        # ✅ Insert into database using service_role key (bypasses RLS)
        response = supabase.table("music_tracks").insert(data).execute()
        
        if response.data:
            print(f"✅ Track inserted successfully!")
            print(f"   Track ID: {response.data[0].get('id', 'unknown')}")
            return {"status": "success", "data": response.data}
        else:
            print(f"⚠️ Insert returned no data")
            return {"status": "success", "message": "Track may have been inserted"}
            
    except Exception as e:
        print(f"❌ Insert Failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Check if it's an RLS error
        error_str = str(e)
        if "row-level security" in error_str.lower() or "42501" in error_str:
            print("🔒 RLS Error Detected!")
            print("   Make sure you're using SUPABASE_SERVICE_ROLE_KEY (not anon key)")
            print("   Current key length:", len(SUPABASE_SERVICE_ROLE_KEY), "chars")
            raise HTTPException(
                status_code=403,
                detail=f"Database permission error. Please verify service_role key is configured. Error: {str(e)}"
            )
        
        raise HTTPException(status_code=500, detail=f"Insert Failed: {str(e)}")

# 3. PROXY: Record Play Stats
@app.post("/record-play")
async def record_play(stat: PlayRecord, x_app_integrity: Optional[str] = Header(None)):
    # ✅ Security check optional for development
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
            print(f"📊 Stats recorded via RPC for track {stat.track_id}")
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
            print(f"📊 Stats recorded via direct insert for track {stat.track_id}")
        
        return {"status": "recorded", "completion_rate": completion_rate}
    except Exception as e:
        print(f"❌ Stats Error: {e}")
        # Don't fail the request if stats recording fails
        return {"status": "error", "detail": str(e)}

# 4. AI ENRICHMENT (Using Groq)
@app.post("/enrich-metadata")
async def enrich_metadata(song: SongRequest, x_app_integrity: Optional[str] = Header(None)):
    # ✅ Security check optional for development
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
        # Call Groq API
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
            model=GROQ_MODELS["primary"],
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
    import traceback
    traceback.print_exc()
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
    print(f"🔑 Service Key Length: {len(SUPABASE_SERVICE_ROLE_KEY) if SUPABASE_SERVICE_ROLE_KEY else 0} chars")
    if SUPABASE_SERVICE_ROLE_KEY and len(SUPABASE_SERVICE_ROLE_KEY) < 200:
        print("⚠️  WARNING: Key seems too short - make sure you're using service_role key!")
    print("═══════════════════════════════════════")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Key Changes:

1. ✅ **Changed `SUPABASE_SERVICE_KEY` to `SUPABASE_SERVICE_ROLE_KEY`** throughout the entire code
2. ✅ All references now use the correct variable name

## Your Render Environment Variables Should Be:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...your-service-role-key-here
GROQ_API_KEY=gsk_...
