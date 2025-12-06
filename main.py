from fastapi import FastAPI, HTTPException, Header, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
import os
import json
import httpx
import hashlib
import time
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
from contextlib import asynccontextmanager
import asyncio

# ============================================================================
# 1. CONFIGURATION & LOGGING
# ============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    APP_SECRET: str = os.getenv("APP_INTEGRITY_SECRET", "DONOTTOUCHAPI")
    UPLOAD_API_KEY: str = os.getenv("UPLOAD_API_KEY", "DONOTTOUCHAPI")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # Rate Limits
    RATE_LIMIT_TRACKS: str = "60/minute"
    RATE_LIMIT_STREAM: str = "30/minute"
    RATE_LIMIT_ENRICH: str = "20/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    
    # Timeouts & Expiry
    EXTERNAL_API_TIMEOUT: int = 8
    METADATA_CACHE_TTL: int = 3600
    SHORT_STREAM_EXPIRY: int = 600  # 10 mins
    LONG_STREAM_EXPIRY: int = 7200  # 2 hours
    LONG_TRACK_THRESHOLD: int = 900000 # 15 mins
    INTEGRITY_WINDOW: int = 60
    
    # AI Settings
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 100

config = Config()

class AppState:
    supabase: Optional[Client] = None
    redis_client: Optional[redis.Redis] = None
    http_client: Optional[httpx.AsyncClient] = None
    groq_client: Optional[Groq] = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting AudioVibe API v16.0 (Merged)...")
    try:
        # 1. Supabase
        if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("Supabase credentials missing")
        state.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
        logger.info("✅ Supabase connected")
        
        # 2. Groq AI
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing")
        state.groq_client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("✅ Groq AI connected")
        
        # 3. Redis (Optional)
        try:
            state.redis_client = redis.from_url(config.REDIS_URL, decode_responses=True, socket_connect_timeout=5)
            state.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable: {e}")
            state.redis_client = None
        
        # 4. HTTP Client
        state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.EXTERNAL_API_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        logger.info("✅ All services initialized")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    logger.info("👋 Shutting down...")
    if state.http_client:
        await state.http_client.aclose()
    if state.redis_client:
        state.redis_client.close()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="AudioVibe Secure API", version="16.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# ============================================================================
# 2. MODELS
# ============================================================================

class SongRequest(BaseModel):
    artist: str = Field(..., min_length=1, max_length=200)
    title: str = Field(..., min_length=1, max_length=200)
    album: Optional[str] = Field(None, max_length=200)
    
    @field_validator('artist', 'title', 'album')
    @classmethod
    def clean_whitespace(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v else v

class TrackUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    artist: str = Field(..., min_length=1, max_length=200)
    album: str = Field(default="Unknown Album", max_length=200)
    audio_file_url: str = Field(..., min_length=1)
    cover_image_url: Optional[str] = None
    duration_ms: int = Field(default=0, ge=0)
    genres: Union[List[str], str] = Field(default_factory=list)
    tier_required: str = Field(default="free")
    
    @field_validator('tier_required')
    @classmethod
    def validate_tier(cls, v: str) -> str:
        valid_tiers = ['free', 'pro', 'ultra_pro']
        v_lower = v.lower().strip()
        if v_lower not in valid_tiers:
            return 'free'
        return v_lower
    
    @field_validator('genres')
    @classmethod
    def validate_genres(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, str):
            if ',' in v:
                return [g.strip() for g in v.split(',') if g.strip()]
            elif ';' in v:
                return [g.strip() for g in v.split(';') if g.strip()]
            else:
                return [v.strip()] if v.strip() else []
        elif isinstance(v, list):
            return [str(g).strip() for g in v if str(g).strip()]
        return []

class PlayRecord(BaseModel):
    user_id: str = Field(..., min_length=1)
    track_id: str = Field(..., min_length=1)
    listen_time_ms: int = Field(..., ge=0)
    total_duration_ms: int = Field(..., gt=0)

# ============================================================================
# 3. SECURITY DEPENDENCIES
# ============================================================================

def verify_app_integrity(
    x_app_integrity: str = Header(..., description="SHA256 integrity hash"),
    x_app_timestamp: str = Header(..., description="Unix timestamp in seconds")
) -> Dict[str, Any]:
    try:
        request_time = int(x_app_timestamp)
        current_time = int(time.time())
        time_diff = abs(current_time - request_time)
        
        if time_diff > config.INTEGRITY_WINDOW:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Request expired")
        
        expected_hash = hashlib.sha256(
            f"{config.APP_SECRET}{x_app_timestamp}".encode()
        ).hexdigest()
        
        if x_app_integrity != expected_hash:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Integrity failed")
        
        return {"timestamp": request_time, "verified": True}
        
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid timestamp")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Security check failed")

def verify_upload_api_key(x_api_key: str = Header(..., description="Upload API Key")) -> Dict[str, Any]:
    if not x_api_key or x_api_key != config.UPLOAD_API_KEY:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return {"authenticated": True}

async def verify_upload_auth(
    x_api_key: Optional[str] = Header(None),
    x_app_integrity: Optional[str] = Header(None),
    x_app_timestamp: Optional[str] = Header(None)
) -> Dict[str, Any]:
    if x_api_key:
        return verify_upload_api_key(x_api_key)
    if x_app_integrity and x_app_timestamp:
        return verify_app_integrity(x_app_integrity, x_app_timestamp)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

# ============================================================================
# 4. HELPERS
# ============================================================================

def ensure_ready():
    if not state.supabase or not state.groq_client:
        raise HTTPException(503, "Services unavailable")

def clean_json_response(content: str) -> str:
    """Helper to remove markdown from AI response"""
    if not content:
        return "{}"
    content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'```\s*', '', content)
    return content.strip()

async def get_cached_metadata(key: str) -> Optional[Dict]:
    if state.redis_client:
        try:
            data = state.redis_client.get(f"metadata:{key}")
            return json.loads(data) if data else None
        except Exception:
            pass
    return None

async def set_cached_metadata(key: str, value: Dict) -> None:
    if state.redis_client:
        try:
            state.redis_client.setex(f"metadata:{key}", config.METADATA_CACHE_TTL, json.dumps(value))
        except Exception:
            pass

async def search_musicbrainz(artist: str, title: str, album: Optional[str] = None) -> Dict:
    try:
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album and album.lower() not in ["unknown", "unknown album", ""]:
            query_parts.append(f'release:"{album}"')
        
        params = {"query": " AND ".join(query_parts), "fmt": "json", "limit": 2}
        headers = {"User-Agent": "AudioVibe/16.0 (contact@audiovibe.app)"}
        
        response = await state.http_client.get(
            "https://musicbrainz.org/ws/2/recording/", params=params, headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if recordings := data.get("recordings"):
                rec = recordings[0]
                return {
                    "found": True,
                    "tags": [t["name"] for t in rec.get("tags", [])][:5],
                    "labels": [
                        l.get("label", {}).get("name") 
                        for r in rec.get("releases", [])[:2] 
                        for l in r.get("label-info", []) 
                        if l.get("label", {}).get("name")
                    ]
                }
    except Exception as e:
        logger.error(f"MusicBrainz error: {e}")
    return {"found": False, "tags": [], "labels": []}

async def search_wikipedia(artist: str, album: str) -> Dict:
    try:
        if not album or album.lower() in ["unknown", "unknown album", ""]:
            return {"hints": [], "is_soundtrack": False}
        
        params = {
            "action": "query", "list": "search",
            "srsearch": f"{album} {artist} album", "format": "json", "srlimit": 2
        }
        
        response = await state.http_client.get("https://en.wikipedia.org/w/api.php", params=params)
        
        if response.status_code == 200:
            data = response.json()
            hints = []
            is_soundtrack = False
            for res in data.get("query", {}).get("search", []):
                txt = (res.get("snippet", "") + res.get("title", "")).lower()
                if any(x in txt for x in ["soundtrack", "film", "movie"]): is_soundtrack = True
                if "bollywood" in txt: hints.append("Bollywood")
                if "punjabi" in txt: hints.append("Punjabi")
                if "telugu" in txt or "tollywood" in txt: hints.append("Tollywood")
                if "tamil" in txt: hints.append("Kollywood")
            return {"hints": list(set(hints)), "is_soundtrack": is_soundtrack}
    except Exception:
        pass
    return {"hints": [], "is_soundtrack": False}

def detect_industry_enhanced(artist: str, title: str, album: Optional[str], mb_data: Dict, wiki_data: Dict) -> str:
    """Robust Industry Detection Logic"""
    artist_lower = artist.lower()
    
    if wiki_data.get("hints"):
        return wiki_data["hints"][0]

    combined_text = f"{artist_lower} {' '.join(mb_data.get('labels', [])).lower()} {' '.join(mb_data.get('tags', [])).lower()}"
    
    if any(x in combined_text for x in ["bollywood", "hindi", "t-series", "zee music", "tips"]):
        return "Bollywood"
    if any(x in combined_text for x in ["punjabi", "speed records", "white hill"]):
        return "Punjabi"
    if any(x in combined_text for x in ["tollywood", "telugu", "aditya music"]):
        return "Tollywood"
    if any(x in combined_text for x in ["atlantic", "columbia", "universal", "interscope"]):
        return "International"
    if any(name in artist_lower for name in ["kumar", "singh", "sharma", "khan"]):
        return "Bollywood"

    return "International"

# ============================================================================
# 5. ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"status": "online", "version": "16.0.0", "mode": "secure_paginated"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/tracks/metadata", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_TRACKS)
async def get_tracks_metadata(request: Request, page: int = 1, limit: int = 50, search: Optional[str] = None):
    """
    MERGED ENDPOINT:
    Attempts Album-Based Pagination first (from your snippet).
    Falls back to Standard Pagination if view is missing.
    """
    ensure_ready()
    try:
        page = max(1, int(page))
        limit = min(max(1, int(limit)), config.MAX_PAGE_SIZE)
        start = (page - 1) * limit
        end = start + limit - 1
        
        try:
            # --- ATTEMPT 1: Album-Based Pagination (from your v10 code) ---
            album_query = state.supabase.table("unique_albums").select("album, artist", count="exact")
            
            if search and search.strip():
                term = f"%{search.strip()}%"
                album_query = album_query.or_(f"album.ilike.{term},artist.ilike.{term}")
            
            album_query = album_query.order("created_at", desc=True).range(start, end)
            albums_response = album_query.execute()
            
            # If no albums found or empty
            if not albums_response.data:
                return {
                    "status": "success", "page": page, "total_albums": 0, "tracks": []
                }

            album_names = [album['album'] for album in albums_response.data]
            
            # Fetch tracks for these albums
            tracks_query = state.supabase.table("music_tracks").select(
                "id, title, artist, album, cover_image_url, duration_ms, genres, tier_required, created_at"
            ).in_("album", album_names).order("created_at", desc=True)
            
            tracks_response = tracks_query.execute()
            
            return {
                "status": "success",
                "page": page,
                "total_albums": albums_response.count,
                "tracks": tracks_response.data,
                "mode": "album_grouped"
            }

        except Exception:
            # --- ATTEMPT 2: Standard Pagination (Fallback) ---
            # If 'unique_albums' view doesn't exist, this runs
            logger.info("⚠️ Falling back to standard pagination")
            
            query = state.supabase.table("music_tracks").select(
                "id, title, artist, album, cover_image_url, duration_ms, genres, tier_required, created_at",
                count="exact"
            )
            
            if search and search.strip():
                term = f"%{search.strip()}%"
                query = query.or_(f"title.ilike.{term},artist.ilike.{term},album.ilike.{term}")
            
            query = query.order("created_at", desc=True).range(start, end)
            response = query.execute()
            
            return {
                "status": "success",
                "page": page,
                "total_count": response.count,
                "tracks": response.data,
                "mode": "standard"
            }
            
    except Exception as e:
        logger.error(f"Metadata error: {e}")
        raise HTTPException(500, "Failed to fetch tracks")

@app.get("/tracks/stream/{track_id}", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_STREAM)
async def get_stream_url(request: Request, track_id: str):
    """
    SECURE STREAM ENDPOINT
    Includes 'download' parameter fix for just_audio compatibility.
    """
    ensure_ready()
    try:
        response = state.supabase.table("music_tracks").select(
            "audio_file_url, duration_ms, title"
        ).eq("id", track_id).execute()
        
        if not response.data:
            raise HTTPException(404, "Track not found")
        
        track = response.data[0]
        duration = track.get('duration_ms', 0)
        is_long = duration > config.LONG_TRACK_THRESHOLD
        expiry = config.LONG_STREAM_EXPIRY if is_long else config.SHORT_STREAM_EXPIRY
        
        raw_url = track['audio_file_url']
        if "/music_files/" in raw_url:
            path = raw_url.split("/music_files/")[1]
        else:
            path = raw_url
            
        # FIX: Ensure just_audio sees .mp3 extension
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', track.get('title', 'track'))
        download_filename = f"{safe_title}.mp3"

        signed = state.supabase.storage.from_("music_files").create_signed_url(
            path, expiry, options={'download': download_filename}
        )
        
        return {
            "stream_url": signed["signedURL"],
            "expires_in": expiry,
            "track_id": track_id
        }
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(500, "Failed to generate stream URL")

@app.post("/tracks", dependencies=[Depends(verify_upload_auth)])
@limiter.limit(config.RATE_LIMIT_UPLOAD)
async def add_track(request: Request, track: TrackUpload):
    ensure_ready()
    try:
        data = track.model_dump()
        response = state.supabase.table("music_tracks").insert(data).execute()
        if response.data:
            return {"status": "success", "data": response.data[0]}
        raise Exception("Database insert failed")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, "Upload failed")

@app.post("/record-play", dependencies=[Depends(verify_app_integrity)])
async def record_play(stat: PlayRecord):
    ensure_ready()
    try:
        rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms) if stat.total_duration_ms > 0 else 0
        state.supabase.rpc('upsert_listening_stat', {
            'p_user_id': stat.user_id, 'p_track_id': stat.track_id,
            'p_listen_time_ms': stat.listen_time_ms, 'p_completion_rate': rate
        }).execute()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Record play error: {e}")
        return {"status": "error"}

@app.post("/enrich-metadata", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_ENRICH)
async def enrich_metadata(request: Request, song: SongRequest):
    """
    Merged AI Logic:
    - Uses strict Apple Music genres (from v15)
    - Uses clean_json_response (from v10) for safety
    """
    ensure_ready()
    
    cache_key = f"v5:{song.artist}:{song.title}:{song.album or ''}".lower()
    if cached := await get_cached_metadata(cache_key):
        return cached
    
    try:
        # 1. Async Context Gathering
        mb_task = search_musicbrainz(song.artist, song.title, song.album)
        wiki_task = search_wikipedia(song.artist, song.album or "")
        mb_data, wiki_data = await asyncio.gather(mb_task, wiki_task, return_exceptions=True)
        
        if isinstance(mb_data, Exception): mb_data = {"found": False}
        if isinstance(wiki_data, Exception): wiki_data = {"hints": []}
        
        industry = detect_industry_enhanced(song.artist, song.title, song.album, mb_data, wiki_data)

        external_context = f"Detected Industry: {industry}. "
        if mb_data.get("found"):
            external_context += f"Tags: {', '.join(mb_data.get('tags', []))}. "
        if wiki_data.get("hints"):
            external_context += f"Keywords: {', '.join(wiki_data['hints'])}."

        # 2. Strict Apple Music Prompt
        prompt = f"""
        Analyze song: "{song.title}" by "{song.artist}" (Album: {song.album}).
        Context: {external_context}

        Task: Return a JSON object with strictly these fields.

        1. "genre": Choose EXACTLY ONE from this Apple Music style list. Do not invent new ones.
           [Pop, Hip-Hop/Rap, Bollywood, Punjabi Pop, R&B/Soul, Alternative, Rock, 
            Indian Pop, Dance, Electronic, Singer/Songwriter, Classical, Jazz, K-Pop, 
            Latin, Metal, Country, Blues, Devotional, Reggae]

        2. "mood": Choose EXACTLY ONE: 
           [Uplifting, Chill, Energetic, Melancholic, Romantic, Aggressive, Focus, Party]

        3. "language": Detect primary language (e.g., Hindi, English, Punjabi, Spanish).

        JSON OUTPUT ONLY:
        """

        # 3. Call Groq
        chat_completion = state.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a music taxonomist. Output strict JSON."},
                {"role": "user", "content": prompt}
            ],
            model=config.GROQ_MODEL,
            temperature=0.2,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        # 4. Clean & Parse JSON (Merged Helper)
        raw_content = chat_completion.choices[0].message.content.strip()
        cleaned_content = clean_json_response(raw_content)
        ai_data = json.loads(cleaned_content)
        
        # 5. Fallback Logic
        genre = ai_data.get("genre", "Pop").strip()
        valid_genres = [
            "Pop", "Hip-Hop/Rap", "Bollywood", "Punjabi Pop", "R&B/Soul", "Alternative", 
            "Rock", "Indian Pop", "Dance", "Electronic", "Singer/Songwriter", 
            "Classical", "Jazz", "K-Pop", "Latin", "Metal", "Country", "Blues", "Devotional"
        ]
        
        if genre not in valid_genres:
            if industry == "Bollywood": genre = "Bollywood"
            elif industry == "Punjabi": genre = "Punjabi Pop"
            elif industry == "Tollywood": genre = "Indian Pop"
            else: genre = "Pop"

        result = {
            "mood": ai_data.get("mood", "Neutral").title(),
            "language": ai_data.get("language", "Unknown").title(),
            "industry": industry,
            "genre": genre,
            "formatted": f"{ai_data.get('mood')};{genre}",
            "is_cached": False
        }
        
        await set_cached_metadata(cache_key, {**result, "is_cached": True})
        return result

    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        return {
            "mood": "Neutral", "language": "Unknown", 
            "industry": "Unknown", "genre": "Pop", 
            "formatted": "Neutral;Pop", "error": True
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
