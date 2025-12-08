from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from collections import defaultdict
import hashlib
import os
import json
import requests
import time
from dotenv import load_dotenv
from supabase import create_client, Client

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES & SETUP
# ============================================================
load_dotenv()

app = FastAPI(title="AudioVibe Secure API", version="7.1.0-FreeTier")

# ============================================================
# 🔐 SECURITY CONFIGURATION
# ============================================================

# ✅ MUST MATCH Flutter's SecurityService._appSecret
APP_INTEGRITY_SECRET = os.getenv("APP_INTEGRITY_SECRET", "DONOTTOUCHAPI")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

# ⚠️ ALLOWED APP VERSIONS
ALLOWED_APP_VERSIONS = ["1.0", "1.0.0", "1.0.1", "1.1.0"]

# 🚫 Rate Limiting Configuration
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100  # requests per window per device
rate_limit_storage = defaultdict(list)

# ============================================================
# 2. INITIALIZE CLIENTS (SUPABASE ONLY)
# ============================================================

supabase: Optional[Client] = None
startup_error: Optional[str] = None

# In-memory cache to save API costs/time
metadata_cache = {}

try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase Credentials Missing in Environment Variables")
    
    if not LASTFM_API_KEY:
        print("⚠️ Warning: LASTFM_API_KEY is missing. Mood/Genre detection will be limited.")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    print("✅ System Online: Clients Initialized Successfully")

except Exception as e:
    startup_error = str(e)
    print(f"❌ Startup Warning: {e}")
    print("⚠️ Server started in MAINTENANCE MODE (Health Check Only).")

# ============================================================
# 3. PYDANTIC MODELS
# ============================================================

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

# ============================================================
# 🛡️ MIDDLEWARE: SECURITY & LOGGING
# ============================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # 1. ✅ ALLOW PUBLIC ENDPOINTS
    if request.url.path in ["/health", "/", "/docs", "/openapi.json", "/favicon.ico"]:
        return await call_next(request)

    # 2. ✅ ALLOW CORS PREFLIGHT
    if request.method == "OPTIONS":
        return await call_next(request)

    # 3. Extract Headers
    timestamp = request.headers.get("x-app-timestamp")
    integrity_hash = request.headers.get("x-app-integrity")
    device_id = request.headers.get("x-device-id")
    app_version = request.headers.get("x-app-version")

    # Debug Printing
    print(f"\n🔍 SEC_CHECK: {request.url.path} | Device: {device_id} | Ver: {app_version}")

    # 4. Validate Presence
    if not all([timestamp, integrity_hash, device_id, app_version]):
        return JSONResponse(status_code=403, content={"detail": "Missing security headers"})

    # 5. Validate App Version
    if app_version not in ALLOWED_APP_VERSIONS:
        return JSONResponse(status_code=403, content={"detail": f"Unsupported app version: {app_version}"})

    # 6. Validate Timestamp (Replay Attack Protection)
    try:
        request_time = int(timestamp)
        current_time = int(time.time())
        time_diff = abs(current_time - request_time)

        if time_diff > 300: # 5 minutes tolerance
            return JSONResponse(status_code=403, content={"detail": "Request expired"})
    except ValueError:
        return JSONResponse(status_code=403, content={"detail": "Invalid timestamp format"})

    # 7. Validate Integrity Hash
    payload = f"{APP_INTEGRITY_SECRET}{timestamp}{device_id}{app_version}"
    expected_hash = hashlib.sha256(payload.encode()).hexdigest()

    if integrity_hash != expected_hash:
        return JSONResponse(status_code=403, content={"detail": "Invalid security signature"})

    # 8. Rate Limiting
    current_time = time.time()
    rate_limit_storage[device_id] = [t for t in rate_limit_storage[device_id] if current_time - t < RATE_LIMIT_WINDOW]
    
    if len(rate_limit_storage[device_id]) >= RATE_LIMIT_MAX_REQUESTS:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    rate_limit_storage[device_id].append(current_time)

    response = await call_next(request)
    return response

# ============================================================
# 📡 CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 5. HELPER FUNCTIONS (FREE APIS)
# ============================================================

def ensure_ready():
    """Stops the request if the backend failed to initialize"""
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Backend Maintenance: {startup_error}")
    if not supabase:
        raise HTTPException(status_code=503, detail="Backend Initializing...")

def get_lastfm_tags(artist, track):
    """Free Last.fm tags for mood/genre detection"""
    if not LASTFM_API_KEY:
        return {"mood": "Neutral", "genre": "Pop", "tags": []}
        
    try:
        # Step 1: Search for track to get details
        url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&artist={artist}&track={track}&api_key={LASTFM_API_KEY}&format=json&autocorrect=1"
        resp = requests.get(url, timeout=3).json()
        
        tags = []
        if "track" in resp and "toptags" in resp["track"]:
             tag_objs = resp["track"]["toptags"].get("tag", [])
             # Handle single tag or list of tags
             if isinstance(tag_objs, dict): tag_objs = [tag_objs]
             tags = [t["name"].lower() for t in tag_objs]

        # Mappings
        mood_map = {
            "romantic": ["love", "romance", "romantic", "ballad"],
            "sad": ["sad", "melancholy", "heartbreak", "emotional"],
            "energetic": ["dance", "party", "upbeat", "energy", "workout"],
            "chill": ["chill", "relax", "acoustic", "lo-fi", "lofi", "ambient"]
        }
        
        genre_map = {
            "pop": ["pop"],
            "rock": ["rock", "metal", "indie"],
            "hip-hop": ["hip-hop", "rap", "trap"],
            "folk": ["folk", "country"],
            "bollywood": ["bollywood", "filmi", "hindi"],
            "devotional": ["devotional", "bhajan", "spiritual"]
        }

        # Determine Mood
        detected_mood = "Neutral"
        for mood, keywords in mood_map.items():
            if any(k in tags for k in keywords):
                detected_mood = mood.title()
                break
        
        # Determine Genre
        detected_genre = "Pop"
        for genre, keywords in genre_map.items():
            if any(k in tags for k in keywords):
                detected_genre = genre.title()
                break

        return {"mood": detected_mood, "genre": detected_genre, "tags": tags[:10]}
    except Exception as e:
        print(f"⚠️ Last.fm Error: {e}")
        return {"mood": "Neutral", "genre": "Pop", "tags": []}

def detect_language_from_lyrics(artist, title):
    """Free lyrics language detection via Lyrics.ovh"""
    try:
        # Lyrics.ovh is a free public API
        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
        response = requests.get(url, timeout=4)
        
        if response.status_code != 200:
            return {"lang": "Unknown", "industry": None}

        lyrics = response.json().get("lyrics", "")
        if not lyrics: return {"lang": "Unknown", "industry": None}

        # Character Set Detection logic
        # Devanagari (Hindi/Marathi)
        if any("\u0900" <= char <= "\u097F" for char in lyrics):
            return {"lang": "Hindi", "industry": "Bollywood"}
        
        # Telugu
        if any("\u0C00" <= char <= "\u0C7F" for char in lyrics):
            return {"lang": "Telugu", "industry": "Tollywood"}
            
        # Gurmukhi (Punjabi)
        if any("\u0A00" <= char <= "\u0A7F" for char in lyrics):
            return {"lang": "Punjabi", "industry": "Punjabi"}
            
        # Tamil
        if any("\u0B80" <= char <= "\u0BFF" for char in lyrics):
            return {"lang": "Tamil", "industry": "Kollywood"}
        
        # Malayalam
        if any("\u0D00" <= char <= "\u0D7F" for char in lyrics):
            return {"lang": "Malayalam", "industry": "Mollywood"}

        # English (ASCII check)
        if any(ord(c) < 128 for c in lyrics[:100]):
            return {"lang": "English", "industry": "International"}

        return {"lang": "Unknown", "industry": None}
    except Exception:
        return {"lang": "Unknown", "industry": None}

def search_musicbrainz(artist: str, title: str, album: Optional[str] = None) -> dict:
    try:
        base_url = "https://musicbrainz.org/ws/2/recording/"
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album and album.lower() != "unknown":
            query_parts.append(f'release:"{album}"')
        
        headers = {"User-Agent": "AudioVibe/7.0 (contact@audiovibe.app)"}
        response = requests.get(base_url, params={"query": " AND ".join(query_parts), "fmt": "json", "limit": 3}, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("recordings"):
                rec = data["recordings"][0]
                labels = []
                for rel in rec.get("releases", [])[:3]:
                    if "label-info" in rel:
                        labels.extend([l.get("label", {}).get("name") for l in rel["label-info"] if l.get("label")])
                return {"found": True, "labels": list(set(labels))[:3], "tags": [t["name"] for t in rec.get("tags", [])][:5]}
    except Exception as e:
        print(f"MusicBrainz Error: {e}")
    return {"found": False, "labels": [], "tags": []}

def detect_industry_python(artist, album, mb_data, lyrics_industry, lastfm_tags):
    """
    Combines Signals to determine Industry:
    1. Lyrics Script (Strongest Signal)
    2. Last.fm Tags (Strong Signal)
    3. Keyword Matching (Fallback)
    """
    # 1. Strongest: Lyrics Script
    if lyrics_industry: 
        return lyrics_industry

    # 2. Strong: Last.fm Tags
    tags_str = " ".join(lastfm_tags).lower()
    if "bollywood" in tags_str: return "Bollywood"
    if "telugu" in tags_str or "tollywood" in tags_str: return "Tollywood"
    if "punjabi" in tags_str: return "Punjabi"
    if "tamil" in tags_str: return "Kollywood"
    if "k-pop" in tags_str or "kpop" in tags_str: return "K-Pop"

    # 3. Fallback: Python Keyword Matching
    txt = f"{artist} {album} {' '.join(mb_data.get('labels', []))}".lower()
    if "bollywood" in txt or "t-series" in txt or "zee music" in txt: return "Bollywood"
    if "tollywood" in txt or "aditya music" in txt: return "Tollywood"
    if "punjabi" in txt or "speed records" in txt: return "Punjabi"
    
    return "International"

# ============================================================
# 📌 API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    status = "Maintenance Mode" if startup_error else "Active"
    return {"status": status, "version": "7.1.0-FreeTier", "error": startup_error}

@app.get("/health")
def health_check():
    return {"status": "healthy" if not startup_error else "unhealthy", "error": startup_error}

@app.post("/enrich-metadata")
async def enrich_metadata(
    song: SongRequest,
    x_app_timestamp: str = Header(...),
    x_app_integrity: str = Header(...),
    x_device_id: str = Header(...),
    x_app_version: str = Header(...)
):
    ensure_ready()
    clean_title, clean_artist = song.title.strip(), song.artist.strip()
    clean_album = song.album.strip() if song.album else ""

    # Check Cache
    cache_key = f"{clean_artist.lower()}:{clean_title.lower()}:{clean_album.lower()}"
    if cache_key in metadata_cache:
        print(f"💾 Returning Cached Metadata for: {clean_title}")
        return metadata_cache[cache_key]

    print(f"🔍 Analyzing New Track: {clean_title} by {clean_artist}")
    
    # 1. PARALLEL-ISH EXECUTION (Sequential for now, but fast)
    # Fetch data from MusicBrainz
    mb_data = search_musicbrainz(clean_artist, clean_title, clean_album)
    
    # Fetch data from Last.fm (Mood/Genre/Tags)
    lastfm_data = get_lastfm_tags(clean_artist, clean_title)
    
    # Fetch data from Lyrics.ovh (Language/Industry)
    lyrics_data = detect_language_from_lyrics(clean_artist, clean_title)

    # 2. SYNTHESIZE RESULTS
    # Determine Language
    language = lyrics_data["lang"]
    if language == "Unknown":
        # Fallback language detection based on industry hints
        if "bollywood" in lastfm_data["tags"]: language = "Hindi"
        elif "punjabi" in lastfm_data["tags"]: language = "Punjabi"
        else: language = "English" # Default fallback

    # Determine Industry
    final_industry = detect_industry_python(
        clean_artist, 
        clean_album, 
        mb_data, 
        lyrics_data["industry"], 
        lastfm_data["tags"]
    )

    # Use Last.fm mood/genre
    mood = lastfm_data["mood"]
    genre = lastfm_data["genre"]

    # ✅ Construct Result
    result = {
        "formatted": f"{mood};{language};{final_industry};{genre}",
        "mood": mood, 
        "language": language, 
        "industry": final_industry, 
        "genre": genre,
        "sources_used": {
            "musicbrainz": mb_data.get("found", False),
            "lastfm": bool(lastfm_data["tags"]),
            "lyrics": lyrics_data["lang"] != "Unknown"
        }
    }
    
    # Save to cache
    metadata_cache[cache_key] = result
    
    return result

# ============================================================
# TRACK MANAGEMENT ENDPOINTS
# ============================================================

@app.get("/tracks")
async def get_tracks(
    x_app_timestamp: str = Header(...),
    x_app_integrity: str = Header(...),
    x_device_id: str = Header(...),
    x_app_version: str = Header(...)
):
    ensure_ready()
    try:
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"❌ Database Error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/tracks")
async def add_track(
    track: TrackUpload,
    x_app_timestamp: str = Header(...),
    x_app_integrity: str = Header(...),
    x_device_id: str = Header(...),
    x_app_version: str = Header(...)
):
    ensure_ready()
    try:
        data = track.dict()
        response = supabase.table("music_tracks").insert(data).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to make it accessible to your Flutter Emulator/Device
    uvicorn.run(app, host="0.0.0.0", port=8000)
