from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq

# 1. Load Env
load_dotenv()

app = FastAPI(title="AudioVibe Backend API", version="5.0.0 (Layman Genres)")

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Auth
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not GROQ_API_KEY:
    raise RuntimeError("CRITICAL: Missing Keys")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Init Failed: {e}")
    raise

# 4. Config
EXPECTED_INTEGRITY_HEADER = 'clean-device-v1'
GROQ_MODELS = {"primary": "llama-3.3-70b-versatile"}
metadata_cache = {}

# 5. Models
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

# 6. Standard Endpoints
@app.get("/")
def read_root():
    return {"status": "Active", "version": "5.0.0"}

@app.get("/tracks")
async def get_tracks(x_app_integrity: Optional[str] = Header(None)):
    try:
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tracks")
async def add_track(track: TrackUpload, x_app_integrity: Optional[str] = Header(None)):
    try:
        data = track.dict()
        if not data.get('tier_required'): data['tier_required'] = 'free'
        response = supabase.table("music_tracks").insert(data).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record-play")
async def record_play(stat: PlayRecord, x_app_integrity: Optional[str] = Header(None)):
    try:
        completion_rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms) if stat.total_duration_ms > 0 else 0
        supabase.rpc('upsert_listening_stat', {
            'p_user_id': stat.user_id, 'p_track_id': stat.track_id, 
            'p_listen_time_ms': stat.listen_time_ms, 'p_completion_rate': completion_rate
        }).execute()
        return {"status": "recorded"}
    except Exception:
        return {"status": "error"}

# 7. ENRICH METADATA (Simplified for Layman)
@app.post("/enrich-metadata")
async def enrich_metadata(song: SongRequest, x_app_integrity: Optional[str] = Header(None)):
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

    # --- SIMPLIFIED LAYMAN PROMPT ---
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
        "   Map the specific style to ONE of these simple categories ONLY:\n"
        "   - 'Party' (Use for: EDM, House, Item Songs, Mass Beats, Club, Disco)\n"
        "   - 'Pop' (Use for: General Hits, Filmy, Easy Listening)\n"
        "   - 'Rock' (Use for: Metal, Alternative, Heavy Guitar)\n"
        "   - 'Hip-Hop' (Use for: Rap, Trap, Drill, R&B)\n"
        "   - 'Folk' (Use for: Traditional, Village, Acoustic, Desi)\n"
        "   - 'Devotional' (Use for: Sufi, Bhajan, Gospel, Qawwali)\n"
        "   - 'Classical' (Use for: Orchestra, Sitar, Ragas)\n"
        "   - 'LoFi' (Use for: Slow, Chill, Study)\n\n"
        
        "REQUIRED OUTPUT (JSON):\n"
        "{\n"
        '  "mood": "One word from Task 1",\n'
        '  "language": "Primary language",\n'
        '  "industry": "One word from Task 2",\n'
        '  "genre": "One word from Task 3 (The Simple Category)",\n'
        '  "reasoning": "Brief explanation"\n'
        "}"
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a music classifier for everyday users. You simplify complex genres into basic categories."},
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
            
            # Safety Check for Mood
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
