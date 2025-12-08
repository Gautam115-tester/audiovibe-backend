from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # <--- IMPORTED FOR MIDDLEWARE FIX
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import os
import json
import requests
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()

app = FastAPI(title="AudioVibe Secure API", version="7.0.2-Stable")

# ============================================================
# 🔐 SECURITY CONFIGURATION
# ============================================================

# ✅ MUST MATCH Flutter's SecurityService._appSecret
APP_INTEGRITY_SECRET = os.getenv("APP_INTEGRITY_SECRET", "DONOTTOUCHAPI")

# ⚠️ ALLOWED APP VERSIONS (Update when you release new versions)
ALLOWED_APP_VERSIONS = ["1.0.0", "1.0.1", "1.1.0"]

# 🚫 Rate Limiting Configuration
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100  # requests per window per device
rate_limit_storage = defaultdict(list)

# ============================================================
# 2. INITIALIZE CLIENTS (SUPABASE & GROQ)
# ============================================================

supabase: Optional[Client] = None
groq_client: Optional[Groq] = None
startup_error: Optional[str] = None

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

# ============================================================
# 3. CONFIGURATION
# ============================================================

EXPECTED_INTEGRITY_HEADER = 'clean-device-v1'
GROQ_MODELS = {"primary": "llama-3.3-70b-versatile"}
metadata_cache = {}

# ============================================================
# 4. PYDANTIC MODELS
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

class PlayRecord(BaseModel):
    user_id: str
    track_id: str
    listen_time_ms: int
    total_duration_ms: int

# ============================================================
# 🛡️ MIDDLEWARE: Security Validation (FIXED)
# ============================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # 1. ✅ ALLOW PUBLIC ENDPOINTS
    # This prevents blocking the health check or Swagger UI
    if request.url.path in ["/health", "/", "/docs", "/openapi.json", "/favicon.ico"]:
        return await call_next(request)

    # 2. ✅ ALLOW CORS PREFLIGHT (OPTIONS)
    # Browsers send this before the actual request. It never has headers.
    if request.method == "OPTIONS":
        return await call_next(request)

    # 3. Extract Headers
    timestamp = request.headers.get("x-app-timestamp")
    integrity_hash = request.headers.get("x-app-integrity")
    device_id = request.headers.get("x-device-id")
    app_version = request.headers.get("x-app-version")

    # 4. Validate Presence
    if not all([timestamp, integrity_hash, device_id, app_version]):
        # ⚠️ FIX: Return JSONResponse instead of raising exception to prevent crash
        return JSONResponse(
            status_code=403,
            content={"detail": "Missing security headers"}
        )

    # 5. Validate App Version
    if app_version not in ALLOWED_APP_VERSIONS:
        return JSONResponse(
            status_code=403,
            content={"detail": "Unsupported app version. Please update."}
        )

    # 6. Validate Timestamp
    try:
        request_time = int(timestamp)
        current_time = int(time.time())
        time_diff = abs(current_time - request_time)

        if time_diff > 300:
            return JSONResponse(
                status_code=403,
                content={"detail": "Request expired or invalid timestamp"}
            )
    except ValueError:
        return JSONResponse(
            status_code=403,
            content={"detail": "Invalid timestamp format"}
        )

    # 7. Validate Integrity Hash
    payload = f"{APP_INTEGRITY_SECRET}{timestamp}{device_id}{app_version}"
    expected_hash = hashlib.sha256(payload.encode()).hexdigest()

    if integrity_hash != expected_hash:
        print(f"❌ Hash Mismatch!")
        print(f"   Expected: {expected_hash}")
        print(f"   Received: {integrity_hash}")
        return JSONResponse(
            status_code=403,
            content={"detail": "Invalid security signature"}
        )

    # 8. Rate Limiting
    current_time = time.time()
    device_requests = rate_limit_storage[device_id]

    # Clean old requests
    device_requests[:] = [
        req_time for req_time in device_requests
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]

    # Check limit
    if len(device_requests) >= RATE_LIMIT_MAX_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )

    device_requests.append(current_time)

    # ✅ Proceed
    response = await call_next(request)
    return response

# ============================================================
# 📡 CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",  # ⚠️ Replace with your actual domain
        "http://localhost:*",      # Development only
        "*"                        # Fallback
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def ensure_ready():
    """Check if backend is ready to process requests"""
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Backend Not Ready: {startup_error}")
    if not supabase or not groq_client:
        raise HTTPException(status_code=503, detail="Backend Not Ready: Clients not initialized")

def search_musicbrainz(artist: str, title: str, album: Optional[str] = None) -> dict:
    try:
        base_url = "https://musicbrainz.org/ws/2/recording/"
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album and album.lower() != "unknown":
            query_parts.append(f'release:"{album}"')
        
        query = " AND ".join(query_parts)
        params = {"query": query, "fmt": "json", "limit": 3}
        headers = {"User-Agent": "AudioVibe/7.0 (contact@audiovibe.app)"}
        
        response = requests.get(base_url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("recordings"):
                recording = data["recordings"][0]
                releases = recording.get("releases", [])
                tags = [tag["name"] for tag in recording.get("tags", [])]
                
                labels = []
                release_types = []
                for release in releases[:3]:
                    if "label-info" in release:
                        labels.extend([li.get("label", {}).get("name") for li in release["label-info"]])
                    release_types.append(release.get("status", ""))
                
                return {
                    "found": True,
                    "release_type": release_types[0] if release_types else "Unknown",
                    "labels": list(set(labels))[:3],
                    "tags": tags[:5]
                }
    except Exception as e:
        print(f"MusicBrainz error: {e}")
    
    return {"found": False, "release_type": "Unknown", "labels": [], "tags": []}

def search_wikipedia(artist: str, album: str) -> dict:
    try:
        if not album or album.lower() == "unknown":
            return {"is_soundtrack": False, "is_film_album": False, "industry_hints": []}
        
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "list": "search",
            "srsearch": f"{album} {artist} album soundtrack",
            "format": "json", "srlimit": 3
        }
        
        response = requests.get(search_url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("query", {}).get("search", [])
            
            for result in results:
                snippet = result.get("snippet", "").lower()
                title = result.get("title", "").lower()
                
                is_soundtrack = any(word in snippet or word in title 
                                   for word in ["soundtrack", "film", "movie", "cinema"])
                
                industry_hints = []
                if any(word in snippet or word in title for word in ["bollywood", "hindi film", "mumbai"]):
                    industry_hints.append("Bollywood")
                if any(word in snippet or word in title for word in ["tollywood", "telugu film", "tamil film"]):
                    industry_hints.append("Tollywood")
                if any(word in snippet or word in title for word in ["punjabi music", "punjabi album"]):
                    industry_hints.append("Punjabi")
                
                if is_soundtrack or industry_hints:
                    return {"is_soundtrack": is_soundtrack, "is_film_album": is_soundtrack, "industry_hints": industry_hints}
    except Exception as e:
        print(f"Wikipedia error: {e}")
    
    return {"is_soundtrack": False, "is_film_album": False, "industry_hints": []}

def detect_industry_enhanced(artist: str, title: str, album: Optional[str], 
                            musicbrainz_data: dict, wiki_data: dict) -> str:
    artist_lower = artist.lower()
    album_lower = album.lower() if album else ""
    
    if wiki_data.get("industry_hints"):
        return wiki_data["industry_hints"][0]
    
    is_film_music = (wiki_data.get("is_soundtrack", False) or 
                    any(word in album_lower for word in ["soundtrack", "ost", "film", "movie"]))
    
    combined_text = f"{artist_lower} {album_lower} {' '.join(musicbrainz_data.get('labels', [])).lower()}"
    
    if "bollywood" in combined_text or "hindi" in combined_text: return "Bollywood"
    if "tollywood" in combined_text or "telugu" in combined_text: return "Tollywood"
    if "punjabi" in combined_text: return "Punjabi"
    
    western = ["atlantic", "columbia", "universal", "warner", "epic", "interscope"]
    if any(ind in combined_text for ind in western): return "International"
    
    indian_names = ["kumar", "singh", "sharma", "khan", "rao", "reddy"]
    if any(name in artist_lower for name in indian_names): return "Indie"
    
    return "International"

# ============================================================
# 📌 API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    if startup_error:
        return {"status": "Maintenance Mode", "error": startup_error}
    return {
        "status": "Active",
        "version": "7.0.2-Stable",
        "message": "Secure API with Enhanced Metadata Detection"
    }

@app.get("/health")
def health_check():
    """Public endpoint for monitoring (no auth required)"""
    return {
        "status": "unhealthy" if startup_error else "healthy",
        "timestamp": int(time.time()),
        "startup_error": startup_error
    }

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
        raise HTTPException(status_code=500, detail=str(e))

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
        if not data.get('tier_required'): data['tier_required'] = 'free'
        response = supabase.table("music_tracks").insert(data).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record-play")
async def record_play(
    stat: PlayRecord,
    x_app_timestamp: str = Header(...),
    x_app_integrity: str = Header(...),
    x_device_id: str = Header(...),
    x_app_version: str = Header(...)
):
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
async def enrich_metadata(
    song: SongRequest,
    x_app_timestamp: str = Header(...),
    x_app_integrity: str = Header(...),
    x_device_id: str = Header(...),
    x_app_version: str = Header(...)
):
    """
    Enhanced metadata enrichment with Markdown cleaning and Debugging
    """
    ensure_ready()
    
    clean_title = song.title.strip()
    clean_artist = song.artist.strip()
    clean_album = song.album.strip() if song.album else ""
    
    cache_key = f"{clean_artist.lower()}:{clean_title.lower()}:{clean_album.lower()}"
    if cache_key in metadata_cache:
        print(f"💾 Cache hit: {cache_key}")
        return metadata_cache[cache_key]

    print(f"🔍 Analyzing: '{clean_title}' by '{clean_artist}'")
    
    # 1. External Search
    musicbrainz_data = search_musicbrainz(clean_artist, clean_title, clean_album)
    wiki_data = search_wikipedia(clean_artist, clean_album)
    
    # 2. Industry Detection
    industry = detect_industry_enhanced(
        clean_artist, clean_title, clean_album,
        musicbrainz_data, wiki_data
    )
    
    # 3. Context for AI
    context_info = f"Artist: {clean_artist}, Title: {clean_title}"
    if clean_album and clean_album.lower() != "unknown":
        context_info += f", Album: {clean_album}"
    if musicbrainz_data.get("found"):
        context_info += f", Labels: {', '.join(musicbrainz_data['labels'][:2])}"
    
    # 4. Groq Classification
    prompt = (
        f"Analyze: {context_info}\nDetected Industry: {industry}\n\n"
        "TASK 1: MOOD (Aggressive, Energetic, Romantic, Melancholic, Chill, Uplifting)\n"
        "TASK 2: LANGUAGE (Hindi, Telugu, Punjabi, English, etc.)\n"
        "TASK 3: GENRE (Party, Pop, Hip-Hop, Folk, Devotional, LoFi, EDM, Jazz)\n\n"
        "OUTPUT: Return ONLY raw JSON. Do not use markdown.\n"
        "{ \"mood\": \"...\", \"language\": \"...\", \"genre\": \"...\" }"
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a music classifier. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.1,
            max_tokens=150,
            response_format={"type": "json_object"} 
        )

        content = chat_completion.choices[0].message.content.strip()
        print(f"🤖 Raw AI Response: {content}")

        # 🛠️ FIX: Strip Markdown Code Blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        mood = data.get("mood", "Neutral").title()
        language = data.get("language", "Unknown").title()
        genre = data.get("genre", "Pop").title()

    except Exception as e:
        print(f"⚠️ Groq Processing Error: {e}")
        mood, language, genre = "Neutral", "Unknown", "Pop"

    result = {
        "formatted": f"{mood};{language};{industry};{genre}",
        "mood": mood, "language": language, "industry": industry, "genre": genre,
        "metadata_sources": {"musicbrainz_found": musicbrainz_data.get("found", False)}
    }
    
    metadata_cache[cache_key] = result
    print(f"✅ Result: {result['formatted']}")
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
