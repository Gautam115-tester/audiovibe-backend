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
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES & SETUP
# ============================================================
load_dotenv()

app = FastAPI(title="AudioVibe Secure API", version="7.2.0-MultiSearchProvider")

# ============================================================
# 🔐 SECURITY CONFIGURATION
# ============================================================

APP_INTEGRITY_SECRET = os.getenv("APP_INTEGRITY_SECRET", "DONOTTOUCHAPI")
ALLOWED_APP_VERSIONS = ["1.0", "1.0.0", "1.0.1", "1.1.0"]
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 100
rate_limit_storage = defaultdict(list)

# ============================================================
# 2. INITIALIZE CLIENTS
# ============================================================

supabase: Optional[Client] = None
groq_client: Optional[Groq] = None
startup_error: Optional[str] = None
GROQ_MODELS = {"primary": "llama-3.3-70b-versatile"}

# Search Provider API Keys
SEARCH_PROVIDERS = {
    "serpapi": os.getenv("SERPAPI_KEY"),
    "valueserp": os.getenv("VALUESERP_KEY"),
    "serper": os.getenv("SERPER_API_KEY"),
    "scraperapi": os.getenv("SCRAPERAPI_KEY"),
    "lastfm": os.getenv("LASTFM_API_KEY"),
}

# In-memory cache
metadata_cache = {}

try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase Credentials Missing")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY Missing")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # Log available search providers
    enabled_providers = [k for k, v in SEARCH_PROVIDERS.items() if v]
    print("✅ System Online: Clients Initialized")
    print(f"🔍 Search Providers Enabled: {', '.join(enabled_providers) if enabled_providers else 'None (Fallback Mode)'}")

except Exception as e:
    startup_error = str(e)
    print(f"❌ Startup Warning: {e}")

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
# 🛡️ SECURITY MIDDLEWARE
# ============================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path in ["/health", "/", "/docs", "/openapi.json", "/favicon.ico"]:
        return await call_next(request)
    
    if request.method == "OPTIONS":
        return await call_next(request)

    timestamp = request.headers.get("x-app-timestamp")
    integrity_hash = request.headers.get("x-app-integrity")
    device_id = request.headers.get("x-device-id")
    app_version = request.headers.get("x-app-version")

    print(f"\n🔍 SEC_CHECK: {request.url.path} | Device: {device_id} | Ver: {app_version}")

    if not all([timestamp, integrity_hash, device_id, app_version]):
        print("   ⚠️ REJECTED: Missing Headers")
        return JSONResponse(status_code=403, content={"detail": "Missing security headers"})

    if app_version not in ALLOWED_APP_VERSIONS:
        print(f"   ⚠️ REJECTED: Unsupported Version '{app_version}'")
        return JSONResponse(status_code=403, content={"detail": f"Unsupported app version: {app_version}"})

    try:
        request_time = int(timestamp)
        current_time = int(time.time())
        time_diff = abs(current_time - request_time)

        if time_diff > 300:
            print(f"   ⚠️ REJECTED: Expired Timestamp (Diff: {time_diff}s)")
            return JSONResponse(status_code=403, content={"detail": "Request expired"})
    except ValueError:
        return JSONResponse(status_code=403, content={"detail": "Invalid timestamp format"})

    payload = f"{APP_INTEGRITY_SECRET}{timestamp}{device_id}{app_version}"
    expected_hash = hashlib.sha256(payload.encode()).hexdigest()

    if integrity_hash != expected_hash:
        print(f"   ❌ REJECTED: Hash Mismatch")
        return JSONResponse(status_code=403, content={"detail": "Invalid security signature"})

    current_time = time.time()
    rate_limit_storage[device_id] = [t for t in rate_limit_storage[device_id] if current_time - t < RATE_LIMIT_WINDOW]
    
    if len(rate_limit_storage[device_id]) >= RATE_LIMIT_MAX_REQUESTS:
        print(f"   ⚠️ REJECTED: Rate Limit Exceeded for {device_id}")
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    rate_limit_storage[device_id].append(current_time)

    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🔍 SEARCH PROVIDER IMPLEMENTATIONS
# ============================================================

def search_serpapi(artist: str, title: str, album: Optional[str] = None) -> dict:
    """SerpAPI - Best quality, 100 free/month then $50/5k"""
    api_key = SEARCH_PROVIDERS.get("serpapi")
    if not api_key:
        return {"success": False, "context": "", "provider": "serpapi_disabled"}
    
    query = f'"{title}" "{artist}"'
    if album and album.lower() not in ["unknown", "single", ""]:
        query += f' "{album}"'
    query += " song music"
    
    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": api_key, "num": 10},
            timeout=5
        )
        
        if response.status_code != 200:
            return {"success": False, "context": "", "provider": "serpapi_error"}
        
        data = response.json()
        context_parts = []
        
        for result in data.get("organic_results", [])[:10]:
            context_parts.append(result.get("snippet", ""))
            context_parts.append(result.get("title", ""))
        
        kg = data.get("knowledge_graph", {})
        if kg.get("description"):
            context_parts.insert(0, kg["description"])
        
        context = " | ".join(filter(None, context_parts))
        
        return {
            "success": len(context) > 50,
            "context": context,
            "provider": "serpapi",
            "result_count": len(data.get("organic_results", []))
        }
    except Exception as e:
        print(f"SerpAPI Error: {e}")
        return {"success": False, "context": "", "provider": "serpapi_exception"}

def search_valueserp(artist: str, title: str, album: Optional[str] = None) -> dict:
    """ValueSerp - Best free tier, 100 free/month"""
    api_key = SEARCH_PROVIDERS.get("valueserp")
    if not api_key:
        return {"success": False, "context": "", "provider": "valueserp_disabled"}
    
    query = f'"{title}" "{artist}"'
    if album and album.lower() not in ["unknown", "single", ""]:
        query += f' "{album}"'
    query += " song music"
    
    try:
        response = requests.get(
            "https://api.valueserp.com/search",
            params={"q": query, "api_key": api_key, "num": 10, "location": "United States"},
            timeout=5
        )
        
        if response.status_code != 200:
            return {"success": False, "context": "", "provider": "valueserp_error"}
        
        data = response.json()
        context_parts = []
        
        for result in data.get("organic_results", [])[:10]:
            context_parts.append(result.get("snippet", ""))
            context_parts.append(result.get("title", ""))
        
        context = " | ".join(filter(None, context_parts))
        
        return {
            "success": len(context) > 50,
            "context": context,
            "provider": "valueserp",
            "result_count": len(data.get("organic_results", []))
        }
    except Exception as e:
        print(f"ValueSerp Error: {e}")
        return {"success": False, "context": "", "provider": "valueserp_exception"}

def search_serper(artist: str, title: str, album: Optional[str] = None) -> dict:
    """Serper.dev - 2,500 free/month"""
    api_key = SEARCH_PROVIDERS.get("serper")
    if not api_key:
        return {"success": False, "context": "", "provider": "serper_disabled"}
    
    query = f'"{title}" "{artist}"'
    if album and album.lower() not in ["unknown", "single", ""]:
        query += f' "{album}"'
    query += " song music"
    
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 10},
            timeout=5
        )
        
        if response.status_code != 200:
            return {"success": False, "context": "", "provider": "serper_error"}
        
        data = response.json()
        context_parts = []
        
        for result in data.get("organic", [])[:10]:
            context_parts.append(result.get("snippet", ""))
            context_parts.append(result.get("title", ""))
        
        kg = data.get("knowledgeGraph", {})
        if kg.get("description"):
            context_parts.insert(0, kg["description"])
        
        context = " | ".join(filter(None, context_parts))
        
        return {
            "success": len(context) > 50,
            "context": context,
            "provider": "serper",
            "result_count": len(data.get("organic", []))
        }
    except Exception as e:
        print(f"Serper Error: {e}")
        return {"success": False, "context": "", "provider": "serper_exception"}

def search_duckduckgo(artist: str, title: str, album: Optional[str] = None) -> dict:
    """DuckDuckGo - 100% FREE, no API key needed"""
    query = f"{title} {artist}"
    if album and album.lower() not in ["unknown", "single", ""]:
        query += f" {album}"
    query += " song music"
    
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=5
        )
        
        if response.status_code != 200:
            return {"success": False, "context": "", "provider": "duckduckgo_error"}
        
        data = response.json()
        context_parts = []
        
        if data.get("Abstract"):
            context_parts.append(data["Abstract"])
        
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                context_parts.append(topic["Text"])
        
        context = " | ".join(context_parts)
        
        return {
            "success": len(context) > 50,
            "context": context,
            "provider": "duckduckgo",
            "result_count": len(context_parts)
        }
    except Exception as e:
        print(f"DuckDuckGo Error: {e}")
        return {"success": False, "context": "", "provider": "duckduckgo_exception"}

def search_lastfm(artist: str, title: str, album: Optional[str] = None) -> dict:
    """Last.fm - FREE with unlimited requests (music-specific)"""
    api_key = SEARCH_PROVIDERS.get("lastfm")
    if not api_key:
        return {"success": False, "context": "", "provider": "lastfm_disabled"}
    
    try:
        response = requests.get(
            "http://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getInfo",
                "artist": artist,
                "track": title,
                "api_key": api_key,
                "format": "json"
            },
            timeout=5
        )
        
        if response.status_code != 200:
            return {"success": False, "context": "", "provider": "lastfm_error"}
        
        data = response.json()
        track = data.get("track", {})
        context_parts = []
        
        wiki = track.get("wiki", {})
        if wiki.get("summary"):
            # Remove HTML tags
            import re
            summary = re.sub('<[^<]+?>', '', wiki["summary"])
            context_parts.append(summary)
        
        tags = [tag["name"] for tag in track.get("toptags", {}).get("tag", [])[:5]]
        if tags:
            context_parts.append(f"Genres: {', '.join(tags)}")
        
        album_data = track.get("album", {})
        if album_data.get("title"):
            context_parts.append(f"Album: {album_data['title']}")
        
        context = " | ".join(context_parts)
        
        return {
            "success": len(context) > 30,
            "context": context,
            "provider": "lastfm",
            "result_count": len(context_parts)
        }
    except Exception as e:
        print(f"Last.fm Error: {e}")
        return {"success": False, "context": "", "provider": "lastfm_exception"}

# ============================================================
# 🎯 SMART MULTI-PROVIDER SEARCH
# ============================================================

def search_with_cascade(artist: str, title: str, album: Optional[str] = None) -> dict:
    """
    Try providers in order of preference until one succeeds.
    Priority: Paid > Free Music-Specific > Free General
    """
    providers = [
        ("SerpAPI", search_serpapi),
        ("ValueSerp", search_valueserp),
        ("Serper", search_serper),
        ("Last.fm", search_lastfm),
        ("DuckDuckGo", search_duckduckgo),
    ]
    
    for name, provider_func in providers:
        try:
            result = provider_func(artist, title, album)
            if result["success"]:
                print(f"✅ Using {name} for search")
                return result
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")
            continue
    
    # All providers failed
    print("❌ All search providers failed")
    return {"success": False, "context": "", "provider": "all_failed"}

# ============================================================
# FALLBACK METHODS
# ============================================================

def search_musicbrainz(artist: str, title: str, album: Optional[str] = None) -> dict:
    try:
        base_url = "https://musicbrainz.org/ws/2/recording/"
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album and album.lower() != "unknown":
            query_parts.append(f'release:"{album}"')
        
        headers = {"User-Agent": "AudioVibe/7.2 (contact@audiovibe.app)"}
        response = requests.get(
            base_url, 
            params={"query": " AND ".join(query_parts), "fmt": "json", "limit": 3}, 
            headers=headers, 
            timeout=5
        )
        
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

def detect_industry_fallback(artist, album, mb_data) -> str:
    txt = f"{artist} {album} {' '.join(mb_data.get('labels', []))}".lower()
    if "bollywood" in txt or "t-series" in txt: return "Bollywood"
    if "tollywood" in txt or "telugu" in txt: return "Tollywood"
    if "kollywood" in txt or "tamil" in txt: return "Kollywood"
    if "punjabi" in txt or "speed records" in txt: return "Punjabi"
    return "International"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_ready():
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Backend Maintenance: {startup_error}")
    if not supabase or not groq_client:
        raise HTTPException(status_code=503, detail="Backend Initializing...")

# ============================================================
# 📌 API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    enabled = [k for k, v in SEARCH_PROVIDERS.items() if v]
    return {
        "status": "Maintenance Mode" if startup_error else "Active",
        "version": "7.2.0-MultiSearchProvider",
        "search_providers_enabled": enabled if enabled else ["fallback_mode"],
        "error": startup_error
    }

@app.get("/health")
def health_check():
    enabled = [k for k, v in SEARCH_PROVIDERS.items() if v]
    return {
        "status": "healthy" if not startup_error else "unhealthy",
        "search_providers": enabled,
        "error": startup_error
    }

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
        print(f"💾 Cache Hit: {clean_title}")
        return metadata_cache[cache_key]

    print(f"🔍 Analyzing: {clean_title} by {clean_artist}")
    
    # PHASE 1: Try search providers (cascade)
    search_result = search_with_cascade(clean_artist, clean_title, clean_album)
    
    # PHASE 2: Fallback metadata
    mb_data = search_musicbrainz(clean_artist, clean_title, clean_album)
    fallback_industry = detect_industry_fallback(clean_artist, clean_album, mb_data)
    
    # PHASE 3: Build AI Context
    context_parts = [
        f"Artist: {clean_artist}",
        f"Title: {clean_title}",
        f"Album: {clean_album}",
    ]
    
    if search_result["success"]:
        context_parts.append(f"Search Results: {search_result['context'][:1500]}")
    
    if mb_data['found']:
        context_parts.append(f"Labels: {', '.join(mb_data['labels'])}")
    
    context_parts.append(f"Industry Hint: {fallback_industry}")
    
    final_context = "\n".join(context_parts)
    
    # PHASE 4: AI Analysis
    prompt = (
        f"Analyze this song:\n\n{final_context}\n\n"
        "Return JSON with: mood (Aggressive/Energetic/Romantic/Melancholic/Chill/Uplifting), "
        "language (Hindi/Punjabi/Telugu/Tamil/English/etc), "
        "genre (Party/Pop/Hip-Hop/Folk/Devotional/LoFi/EDM/Jazz/Classical/Rock), "
        "industry (Bollywood/Tollywood/Kollywood/Punjabi/Indie/International).\n"
        'Format: {"mood": "...", "language": "...", "genre": "...", "industry": "..."}'
    )

    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        content = chat.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(content)
        mood = data.get("mood", "Neutral").title()
        language = data.get("language", "Unknown").title()
        genre = data.get("genre", "Pop").title()
        ai_industry = data.get("industry", "International").title()
        
        final_industry = ai_industry if ai_industry != "International" else fallback_industry

    except Exception as e:
        print(f"⚠️ AI Error: {e}")
        mood, language, genre, final_industry = "Neutral", "Unknown", "Pop", fallback_industry

    result = {
        "formatted": f"{mood};{language};{final_industry};{genre}",
        "mood": mood,
        "language": language,
        "industry": final_industry,
        "genre": genre,
        "sources_used": {
            "search_provider": search_result.get("provider", "none"),
            "search_success": search_result["success"],
            "musicbrainz": mb_data["found"]
        }
    }
    
    metadata_cache[cache_key] = result
    print(f"✅ Result: {result['formatted']}")
    
    return result

# ============================================================
# OTHER ENDPOINTS
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
        return {"status": "recorded"}
    except Exception as e:
        return {"status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
