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

app = FastAPI(title="AudioVibe Backend API", version="2.0.0 (Matrix Logic)")

# 2. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. API Key Validation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("CRITICAL: Supabase credentials missing.")
    raise RuntimeError("Supabase configuration missing")

if not GROQ_API_KEY:
    print("CRITICAL: GROQ_API_KEY missing.")
    raise RuntimeError("Groq API key missing")

# 4. Initialize Clients
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    print("✅ Supabase initialized.")
except Exception as e:
    print(f"❌ Supabase init failed: {e}")
    raise

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq initialized.")
except Exception as e:
    print(f"❌ Groq init failed: {e}")
    raise

# 5. Constants & Config
EXPECTED_INTEGRITY_HEADER = 'clean-device-v1'

# We use the larger model for better logic/reasoning capabilities
GROQ_MODELS = {
    "primary": "llama-3.3-70b-versatile" 
}

metadata_cache = {}

# 6. Data Models
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

# 7. Standard Endpoints
@app.get("/")
def read_root():
    return {
        "status": "AudioVibe Backend Active",
        "logic_engine": "Text-Logic + Verification Matrix",
        "endpoints": ["/health", "/enrich-metadata", "/tracks", "/record-play"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "groq": "connected",
        "supabase": "connected"
    }

@app.get("/tracks")
async def get_tracks(x_app_integrity: Optional[str] = Header(None)):
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header: {x_app_integrity}")
    
    try:
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"Error fetching tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tracks")
async def add_track(track: TrackUpload, x_app_integrity: Optional[str] = Header(None)):
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header")
    
    try:
        data = track.dict()
        if not data.get('tier_required'):
            data['tier_required'] = 'free'
        
        response = supabase.table("music_tracks").insert(data).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        print(f"Insert Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record-play")
async def record_play(stat: PlayRecord, x_app_integrity: Optional[str] = Header(None)):
    try:
        completion_rate = 0.0
        if stat.total_duration_ms > 0:
            completion_rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms)

        try:
            supabase.rpc('upsert_listening_stat', {
                'p_user_id': stat.user_id,
                'p_track_id': stat.track_id,
                'p_listen_time_ms': stat.listen_time_ms,
                'p_completion_rate': completion_rate
            }).execute()
        except Exception:
            supabase.table("listening_stats").insert({
                "user_id": stat.user_id,
                "track_id": stat.track_id,
                "listen_time_ms": stat.listen_time_ms,
                "total_duration_ms": stat.total_duration_ms,
                "completed": completion_rate >= 0.9
            }).execute()
        
        return {"status": "recorded"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# 8. CORE LOGIC: Text-Logic + Verification Matrix
@app.post("/enrich-metadata")
async def enrich_metadata(song: SongRequest, x_app_integrity: Optional[str] = Header(None)):
    if x_app_integrity and x_app_integrity != EXPECTED_INTEGRITY_HEADER:
        print(f"⚠️ Invalid integrity header")

    # 1. Clean Inputs
    clean_title = song.title.strip()
    clean_artist = song.artist.strip()
    clean_album = song.album.strip() if song.album else ""

    # 2. Check Cache
    cache_key = f"{clean_artist.lower()}:{clean_title.lower()}"
    if cache_key in metadata_cache:
        print(f"⚡ Cache hit for: {clean_title}")
        return metadata_cache[cache_key]

    # 3. Smart Search Query Construction
    if clean_album and clean_album.lower() != "unknown":
        search_query = f"Track: '{clean_title}' from Album/Movie: '{clean_album}' by Artist: '{clean_artist}'"
    else:
        search_query = f"Track: '{clean_title}' by Artist: '{clean_artist}'"

    print(f"🔍 Analyzing: {search_query}")

    # 4. The Verification Matrix Prompt
    # This prompt forces the AI to check specific rules before deciding logic.
    prompt = (
        f"Analyze this music track: {search_query}.\n\n"
        "INTERNAL ANALYSIS STEPS (Mental Check):\n"
        "1. **Recall Audio**: Think about the tempo, instruments (Dhol, Guitar, Synth), and vocal style.\n"
        "2. **Check Energy**: Is it High (Fast/Loud) or Low (Slow/Quiet)?\n"
        "3. **Check Valence**: Is the emotion Positive (Happy/Love) or Negative (Angry/Sad)?\n\n"
        "VERIFICATION MATRIX (Apply these rules strictly):\n"
        "---------------------------------------------------\n"
        "| Energy | Valence  | Resulting Mood       |\n"
        "|--------|----------|----------------------|\n"
        "| High   | Negative | Aggressive / Intense | (e.g., War, Fight, Drill, Metal, Arjan Vailly)\n"
        "| High   | Positive | Energetic / Upbeat   | (e.g., Dance, Party, Celebration)\n"
        "| Low    | Positive | Romantic / Chill     | (e.g., Love Ballads, Lofi, Acoustic)\n"
        "| Low    | Negative | Melancholic / Sad    | (e.g., Heartbreak, Slow Sad Songs)\n"
        "| Any    | Spiritual| Spiritual            | (e.g., Sufi, Devotional, Qawwali)\n"
        "---------------------------------------------------\n\n"
        "Based on the matrix, provide the metadata in JSON format:\n"
        "{\n"
        '  "mood": "Select ONE word from the Resulting Mood column above",\n'
        '  "language": "Primary language (e.g., Hindi, Punjabi, English)",\n'
        '  "genre": "Specific Genre (e.g., Bollywood, Punjabi Folk, Hip-Hop)",\n'
        '  "reasoning": "Briefly explain based on Energy and Valence"\n'
        "}"
    )

    try:
        # 5. Call AI
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "You are a music expert. You ALWAYS use the Verification Matrix to determine mood. You DO NOT assume all Bollywood songs are Romantic."
                },
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.1, # Low temperature = Strict adherence to logic
            max_tokens=200,
            response_format={"type": "json_object"}
        )

        content = chat_completion.choices[0].message.content.strip()
        print(f"🤖 AI Output: {content}")

        # 6. Parse Response
        try:
            data = json.loads(content)
            mood = data.get("mood", "Neutral").title()
            language = data.get("language", "Unknown").title()
            genre = data.get("genre", "General").title()
            
            # 7. Final Safety Net (The "Arjan Vailly" Clause)
            # Even if AI fails the matrix, we catch known high-aggression keywords
            reasoning = data.get("reasoning", "").lower()
            if "fight" in reasoning or "war" in reasoning or "aggressive" in reasoning:
                if mood == "Romantic":
                    mood = "Aggressive"

        except json.JSONDecodeError:
            print("❌ JSON parse failed, using fallback")
            mood, language, genre = "Neutral", "Unknown", "General"

        result = {
            "formatted": f"{mood};{language};{genre}",
            "mood": mood,
            "language": language,
            "genre": genre
        }
        
        metadata_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"🔥 Error: {e}")
        return {
            "formatted": "Neutral;Unknown;General",
            "mood": "Neutral",
            "language": "Unknown",
            "genre": "General",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    print("="*40)
    print("AudioVibe Backend: Matrix Logic Loaded")
    print("="*40)
    uvicorn.run(app, host="0.0.0.0", port=8000)
