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
from groq import Groq

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES & SETUP
# ============================================================
load_dotenv()

app = FastAPI(title="AudioVibe Secure API", version="7.0.6-Final")

# ============================================================
# 🔐 SECURITY CONFIGURATION
# ============================================================

# ✅ MUST MATCH Flutter's SecurityService._appSecret
APP_INTEGRITY_SECRET = os.getenv("APP_INTEGRITY_SECRET", "DONOTTOUCHAPI")

# ⚠️ ALLOWED APP VERSIONS
# ✅ FIX: Added "1.0" because your Android logs showed the phone sending "1.0"
ALLOWED_APP_VERSIONS = ["1.0", "1.0.0", "1.0.1", "1.1.0"]

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
GROQ_MODELS = {"primary": "llama-3.3-70b-versatile"}

# In-memory cache to save API costs
metadata_cache = {}

try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase Credentials Missing in Environment Variables")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY Missing in Environment Variables")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
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

class PlayRecord(BaseModel):
    user_id: str
    track_id: str
    listen_time_ms: int
    total_duration_ms: int

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
        print("   ⚠️ REJECTED: Missing Headers")
        return JSONResponse(status_code=403, content={"detail": "Missing security headers"})

    # 5. Validate App Version
    if app_version not in ALLOWED_APP_VERSIONS:
        print(f"   ⚠️ REJECTED: Unsupported Version '{app_version}'")
        return JSONResponse(status_code=403, content={"detail": f"Unsupported app version: {app_version}. Allowed: {ALLOWED_APP_VERSIONS}"})

    # 6. Validate Timestamp (Replay Attack Protection)
    try:
        request_time = int(timestamp)
        current_time = int(time.time())
        time_diff = abs(current_time - request_time)

        if time_diff > 300: # 5 minutes tolerance
            print(f"   ⚠️ REJECTED: Expired Timestamp (Diff: {time_diff}s)")
            return JSONResponse(status_code=403, content={"detail": "Request expired"})
    except ValueError:
        return JSONResponse(status_code=403, content={"detail": "Invalid timestamp format"})

    # 7. Validate Integrity Hash
    # PAYLOAD = Secret + Timestamp + DeviceID + Version
    payload = f"{APP_INTEGRITY_SECRET}{timestamp}{device_id}{app_version}"
    expected_hash = hashlib.sha256(payload.encode()).hexdigest()

    if integrity_hash != expected_hash:
        print(f"   ❌ REJECTED: Hash Mismatch")
        print(f"      Expected: {expected_hash}")
        print(f"      Received: {integrity_hash}")
        return JSONResponse(status_code=403, content={"detail": "Invalid security signature"})

    # 8. Rate Limiting
    current_time = time.time()
    # Remove requests older than the window
    rate_limit_storage[device_id] = [t for t in rate_limit_storage[device_id] if current_time - t < RATE_LIMIT_WINDOW]
    
    if len(rate_limit_storage[device_id]) >= RATE_LIMIT_MAX_REQUESTS:
        print(f"   ⚠️ REJECTED: Rate Limit Exceeded for {device_id}")
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    rate_limit_storage[device_id].append(current_time)

    # ✅ Proceed
    response = await call_next(request)
    return response

# ============================================================
# 📡 CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development/Flutter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def ensure_ready():
    """Stops the request if the backend failed to initialize"""
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Backend Maintenance: {startup_error}")
    if not supabase or not groq_client:
        raise HTTPException(status_code=503, detail="Backend Initializing...")

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
                        labels.extend([l.get("label", {}).get("name") for l in rel["label-info"]])
                return {"found": True, "labels": list(set(labels))[:3], "tags": [t["name"] for t in rec.get("tags", [])][:5]}
    except Exception as e:
        print(f"MusicBrainz Error: {e}")
    return {"found": False, "labels": [], "tags": []}

def search_wikipedia(artist: str, album: str) -> dict:
    try:
        if not album or album.lower() == "unknown": return {"industry_hints": [], "is_soundtrack": False}
        response = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": f"{album} {artist} album", "format": "json", "srlimit": 3}, timeout=5)
        if response.status_code == 200:
            results = response.json().get("query", {}).get("search", [])
            hints = []
            is_soundtrack = False
            for r in results:
                txt = (r.get("snippet", "") + r.get("title", "")).lower()
                if "bollywood" in txt or "hindi" in txt: hints.append("Bollywood")
                if "tollywood" in txt or "telugu" in txt: hints.append("Tollywood")
                if "punjabi" in txt: hints.append("Punjabi")
                
                # Check for soundtrack keywords
                if "soundtrack" in txt or "ost" in txt:
                    is_soundtrack = True

            if hints or is_soundtrack: 
                return {"industry_hints": list(set(hints)), "is_soundtrack": is_soundtrack}
    except Exception: pass
    return {"industry_hints": [], "is_soundtrack": False}

def detect_industry(artist, album, mb_data, wiki_data):
    if wiki_data.get("industry_hints"): return wiki_data["industry_hints"][0]
    txt = f"{artist} {album} {' '.join(mb_data.get('labels', []))}".lower()
    if "bollywood" in txt: return "Bollywood"
    if "tollywood" in txt: return "Tollywood"
    if "punjabi" in txt: return "Punjabi"
    return "International"

# ============================================================
# 📌 API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    status = "Maintenance Mode" if startup_error else "Active"
    return {"status": status, "version": "7.0.6-Final", "error": startup_error}

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
    
    # 1. External Search
    mb_data = search_musicbrainz(clean_artist, clean_title, clean_album)
    wiki_data = search_wikipedia(clean_artist, clean_album)
    
    # 2. Basic Python Industry Detection (Keyword based)
    detected_industry_python = detect_industry(clean_artist, clean_album, mb_data, wiki_data)
    
    context = f"Artist: {clean_artist}, Title: {clean_title}, Album: {clean_album}, PythonDetectedIndustry: {detected_industry_python}"
    if mb_data['found']: context += f", Labels: {', '.join(mb_data['labels'])}"
    
    # 3. GROQ PROMPT (Updated to ask for Industry)
    prompt = (
        f"Analyze: {context}\n"
        "TASK 1: MOOD (Aggressive, Energetic, Romantic, Melancholic, Chill, Uplifting)\n"
        "TASK 2: LANGUAGE (Hindi, Telugu, Punjabi, English, Tamil, Malayalam, etc.)\n"
        "TASK 3: GENRE (Party, Pop, Hip-Hop, Folk, Devotional, LoFi, EDM, Jazz)\n"
        "TASK 4: INDUSTRY (Bollywood, Tollywood, Mollywood, Kollywood, Punjabi, International, Indie)\n"
        "OUTPUT: Return ONLY raw JSON.\n"
        "{ \"mood\": \"...\", \"language\": \"...\", \"genre\": \"...\", \"industry\": \"...\" }"
    )

    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": "Return valid JSON only. Do not use Markdown."}, {"role": "user", "content": prompt}],
            model=GROQ_MODELS["primary"], temperature=0.1, response_format={"type": "json_object"}
        )
        content = chat.choices[0].message.content.strip()
        print(f"🤖 AI Raw: {content}")

        # 🛠️ ROBUST MARKDOWN STRIPPER
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
            else:
                content = content.replace("```json", "").replace("```", "")
        
        content = content.strip()
        
        data = json.loads(content)
        mood = data.get("mood", "Neutral").title()
        language = data.get("language", "Unknown").title()
        genre = data.get("genre", "Pop").title()
        
        # ✅ SMART INDUSTRY LOGIC
        # If Python thought it was "International" (default), but AI says "Punjabi", trust the AI.
        ai_industry = data.get("industry", "International").title()
        
        if detected_industry_python == "International" and ai_industry != "International":
            final_industry = ai_industry
        else:
            final_industry = detected_industry_python

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON Parse Error: {e} | Content: {content}")
        mood, language, genre, final_industry = "Neutral", "Unknown", "Pop", detected_industry_python
    except Exception as e:
        print(f"⚠️ Groq API Error: {e}")
        mood, language, genre, final_industry = "Neutral", "Unknown", "Pop", detected_industry_python

    # ✅ Construct Result matching what Flutter expects
    result = {
        "formatted": f"{mood};{language};{final_industry};{genre}",
        "mood": mood, 
        "language": language, 
        "industry": final_industry, 
        "genre": genre,
        "sources_used": {
            "musicbrainz": mb_data.get("found", False),
            "is_soundtrack": wiki_data.get("is_soundtrack", False),
            "wiki_hints": len(wiki_data.get("industry_hints", [])) > 0,
            "ai_industry_override": (final_industry != detected_industry_python)
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
