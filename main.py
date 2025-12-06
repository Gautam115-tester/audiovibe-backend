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
# CONFIGURATION & LOGGING
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
    APP_SECRET: str = os.getenv("APP_INTEGRITY_SECRET", "")
    UPLOAD_API_KEY: str = os.getenv("UPLOAD_API_KEY", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "").split(",")
    RATE_LIMIT_TRACKS: str = "60/minute"
    RATE_LIMIT_STREAM: str = "30/minute"
    RATE_LIMIT_ENRICH: str = "20/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    EXTERNAL_API_TIMEOUT: int = 8
    METADATA_CACHE_TTL: int = 3600
    SHORT_STREAM_EXPIRY: int = 480
    LONG_STREAM_EXPIRY: int = 5400
    LONG_TRACK_THRESHOLD: int = 900000
    INTEGRITY_WINDOW: int = 60
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
    logger.info("🚀 Starting AudioVibe API v13.0...")
    try:
        if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("Supabase credentials missing")
        state.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
        logger.info("✅ Supabase connected")
        
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing")
        state.groq_client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("✅ Groq AI connected")
        
        try:
            state.redis_client = redis.from_url(config.REDIS_URL, decode_responses=True, socket_connect_timeout=5)
            state.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable: {e}")
            state.redis_client = None
        
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
app = FastAPI(title="AudioVibe Secure API", version="13.0.0", lifespan=lifespan)
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
# MODELS
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
            logger.warning(f"Invalid tier '{v}', defaulting to 'free'")
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
    
    @field_validator('album', 'title', 'artist')
    @classmethod
    def clean_fields(cls, v: str) -> str:
        return v.strip() if v else v

class PlayRecord(BaseModel):
    user_id: str = Field(..., min_length=1)
    track_id: str = Field(..., min_length=1)
    listen_time_ms: int = Field(..., ge=0)
    total_duration_ms: int = Field(..., gt=0)

# ============================================================================
# SECURITY DEPENDENCIES
# ============================================================================

def verify_app_integrity(
    x_app_integrity: str = Header(..., description="SHA256 integrity hash"),
    x_app_timestamp: str = Header(..., description="Unix timestamp in seconds")
) -> Dict[str, Any]:
    """Verify request integrity using timestamp-based hash"""
    try:
        request_time = int(x_app_timestamp)
        current_time = int(time.time())
        time_diff = abs(current_time - request_time)
        
        if time_diff > config.INTEGRITY_WINDOW:
            logger.warning(f"⚠️  Expired request: diff={time_diff}s")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Request expired. Time difference: {time_diff}s. Max: {config.INTEGRITY_WINDOW}s"
            )
        
        expected_hash = hashlib.sha256(
            f"{config.APP_SECRET}{x_app_timestamp}".encode()
        ).hexdigest()
        
        if x_app_integrity != expected_hash:
            logger.warning(f"⚠️  Integrity check failed")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Integrity verification failed"
            )
        
        return {"timestamp": request_time, "verified": True, "time_diff": time_diff}
        
    except ValueError as e:
        logger.error(f"❌ Invalid timestamp: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid timestamp format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Integrity error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Security verification failed"
        )

def verify_upload_api_key(
    x_api_key: str = Header(..., description="Upload API Key")
) -> Dict[str, Any]:
    """Verify API key for upload operations"""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if x_api_key != config.UPLOAD_API_KEY:
        logger.warning(f"⚠️  Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return {"authenticated": True, "method": "api_key"}

async def verify_upload_auth(
    x_api_key: Optional[str] = Header(None),
    x_app_integrity: Optional[str] = Header(None),
    x_app_timestamp: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Flexible upload auth: API key OR integrity headers"""
    if x_api_key:
        try:
            return verify_upload_api_key(x_api_key)
        except HTTPException:
            pass
    
    if x_app_integrity and x_app_timestamp:
        try:
            return verify_app_integrity(x_app_integrity, x_app_timestamp)
        except HTTPException:
            pass
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )

# ============================================================================
# HELPERS
# ============================================================================

def ensure_ready():
    if not state.supabase:
        raise HTTPException(503, "Database unavailable")
    if not state.groq_client:
        raise HTTPException(503, "AI service unavailable")

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
    """Search MusicBrainz for metadata - FIXED VERSION"""
    try:
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album and album.lower() not in ["unknown", "unknown album", ""]:
            query_parts.append(f'release:"{album}"')
        
        params = {"query": " AND ".join(query_parts), "fmt": "json", "limit": 3}
        headers = {"User-Agent": "AudioVibe/13.0"}
        
        response = await state.http_client.get(
            "https://musicbrainz.org/ws/2/recording/",
            params=params,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if recordings := data.get("recordings"):
                recording = recordings[0]
                tags = [tag["name"] for tag in recording.get("tags", [])]
                labels = []
                for release in recording.get("releases", [])[:3]:
                    if label_info := release.get("label-info"):
                        labels.extend([li.get("label", {}).get("name") for li in label_info if li.get("label", {}).get("name")])
                return {"found": True, "labels": list(set(labels))[:3], "tags": tags[:5]}
    except Exception as e:
        logger.error(f"MusicBrainz error: {e}")
    return {"found": False, "labels": [], "tags": []}

async def search_wikipedia(artist: str, album: str) -> Dict:
    """Search Wikipedia for album info - FIXED VERSION"""
    try:
        if not album or album.lower() in ["unknown", "unknown album", ""]:
            return {"is_soundtrack": False, "is_film_album": False, "industry_hints": []}
        
        params = {
            "action": "query", "list": "search",
            "srsearch": f"{album} {artist} album",
            "format": "json", "srlimit": 3
        }
        
        response = await state.http_client.get("https://en.wikipedia.org/w/api.php", params=params)
        
        if response.status_code == 200:
            data = response.json()
            for result in data.get("query", {}).get("search", []):
                combined = f"{result.get('snippet', '')} {result.get('title', '')}".lower()
                is_soundtrack = any(w in combined for w in ["soundtrack", "film", "movie", "cinema"])
                hints = []
                if any(w in combined for w in ["bollywood", "hindi film", "mumbai"]):
                    hints.append("Bollywood")
                if any(w in combined for w in ["tollywood", "telugu", "tamil"]):
                    hints.append("Tollywood")
                if "punjabi" in combined:
                    hints.append("Punjabi")
                if is_soundtrack or hints:
                    return {"is_soundtrack": is_soundtrack, "is_film_album": is_soundtrack, "industry_hints": hints}
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
    return {"is_soundtrack": False, "is_film_album": False, "industry_hints": []}

def detect_industry_enhanced(artist: str, title: str, album: Optional[str], mb_data: Dict, wiki_data: Dict) -> str:
    """Enhanced industry detection - WORKING VERSION FROM OLD CODE"""
    if hints := wiki_data.get("industry_hints"):
        return hints[0]
    
    combined = f"{artist} {album or ''} {' '.join(mb_data.get('labels',[]))}".lower()
    
    if any(k in combined for k in ["bollywood", "hindi", "mumbai", "t-series", "zee music"]):
        return "Bollywood"
    if any(k in combined for k in ["tollywood", "telugu", "tamil", "kollywood"]):
        return "Tollywood"
    if "punjabi" in combined:
        return "Punjabi"
    
    western_labels = ["atlantic", "columbia", "universal", "warner", "epic"]
    if any(label in combined for label in western_labels):
        return "International"
    
    if "independent" in combined or "indie" in combined:
        return "Indie"
    
    return "International"

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "AudioVibe Secure API",
        "version": "13.0.0",
        "status": "online",
        "security": "enabled",
        "fixes": ["AI enrichment restored", "Pagination added", "Security enhanced"],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "13.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "supabase": bool(state.supabase),
            "groq": bool(state.groq_client),
            "redis": bool(state.redis_client)
        }
    }

@app.get("/tracks/metadata", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_TRACKS)
async def get_tracks_metadata(
    request: Request,
    page: int = 1,
    limit: int = 50,
    search: Optional[str] = None
):
    """Get tracks with pagination - SECURED"""
    ensure_ready()
    
    try:
        page = max(1, int(page))
        limit = min(max(1, int(limit)), config.MAX_PAGE_SIZE)
    except (ValueError, TypeError):
        raise HTTPException(422, "Invalid page or limit")
    
    try:
        start = (page - 1) * limit
        end = start + limit - 1
        
        query = state.supabase.table("music_tracks").select(
            "id, title, artist, album, cover_image_url, duration_ms, genres, tier_required, created_at",
            count="exact"
        )
        
        if search and search.strip():
            search_term = f"%{search.strip()}%"
            query = query.or_(f"title.ilike.{search_term},artist.ilike.{search_term},album.ilike.{search_term}")
        
        query = query.order("created_at", desc=True).range(start, end)
        response = query.execute()
        
        logger.info(f"📋 Fetched {len(response.data)} tracks (page {page})")
        
        return {
            "status": "success",
            "page": page,
            "limit": limit,
            "total_count": response.count,
            "count": len(response.data),
            "tracks": response.data
        }
    except Exception as e:
        logger.error(f"❌ Metadata error: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to fetch tracks: {str(e)}")

@app.get("/tracks/stream/{track_id}", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_STREAM)
async def get_stream_url(request: Request, track_id: str):
    """Generate signed stream URL - SECURED"""
    ensure_ready()
    
    try:
        response = state.supabase.table("music_tracks").select(
            "audio_file_url, duration_ms, title, artist"
        ).eq("id", track_id).execute()
        
        if not response.data:
            raise HTTPException(404, "Track not found")
        
        track = response.data[0]
        duration_ms = track.get('duration_ms', 0)
        is_long = not duration_ms or duration_ms == 0 or duration_ms > config.LONG_TRACK_THRESHOLD
        expiry = config.LONG_STREAM_EXPIRY if is_long else config.SHORT_STREAM_EXPIRY
        
        raw_url = track['audio_file_url']
        path = raw_url.split("/music_files/")[1] if "/music_files/" in raw_url else raw_url
        signed = state.supabase.storage.from_("music_files").create_signed_url(path, expiry)
        
        logger.info(f"🎵 Stream URL: {track.get('title')} by {track.get('artist')}")
        
        return {
            "stream_url": signed["signedURL"],
            "expires_in": expiry,
            "expires_at": int(time.time()) + expiry,
            "track_id": track_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Stream error: {e}")
        raise HTTPException(500, "Failed to generate stream URL")

@app.post("/tracks", dependencies=[Depends(verify_upload_auth)])
@limiter.limit(config.RATE_LIMIT_UPLOAD)
async def add_track(request: Request, track: TrackUpload):
    """Upload track - SECURED (API key OR integrity)"""
    ensure_ready()
    
    try:
        data = track.model_dump()
        logger.info(f"📤 Uploading: '{data['title']}' by {data['artist']}")
        
        response = state.supabase.table("music_tracks").insert(data).execute()
        
        if response.data:
            track_id = response.data[0].get('id', 'unknown')
            logger.info(f"✅ Upload successful! ID: {track_id}")
            return {
                "status": "success",
                "message": "Track uploaded successfully",
                "data": response.data[0]
            }
        else:
            raise Exception("No data returned from database")
            
    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/record-play", dependencies=[Depends(verify_app_integrity)])
async def record_play(stat: PlayRecord):
    """Record play statistics - SECURED"""
    ensure_ready()
    
    try:
        rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms) if stat.total_duration_ms > 0 else 0
        
        state.supabase.rpc('upsert_listening_stat', {
            'p_user_id': stat.user_id,
            'p_track_id': stat.track_id,
            'p_listen_time_ms': stat.listen_time_ms,
            'p_completion_rate': rate
        }).execute()
        
        logger.info(f"📊 Play recorded: user={stat.user_id}, track={stat.track_id}, rate={rate*100:.1f}%")
        return {"status": "success", "completion_rate": round(rate * 100, 2)}
        
    except Exception as e:
        logger.error(f"❌ Record play error: {e}")
        return {"status": "error", "message": "Failed to record play"}

@app.post("/enrich-metadata", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_ENRICH)
async def enrich_metadata(request: Request, song: SongRequest):
    """
    AI metadata enrichment - FIXED & WORKING
    Returns: mood, language, industry, genre
    """
    ensure_ready()
    
    cache_key = f"{song.artist}:{song.title}:{song.album or ''}".lower()
    
    if cached := await get_cached_metadata(cache_key):
        logger.info(f"✅ Cache hit: {cache_key}")
        return cached
    
    try:
        # Fetch external data
        mb_task = search_musicbrainz(song.artist, song.title, song.album)
        wiki_task = search_wikipedia(song.artist, song.album or "")
        mb_data, wiki_data = await asyncio.gather(mb_task, wiki_task, return_exceptions=True)
        
        if isinstance(mb_data, Exception):
            mb_data = {"found": False, "labels": [], "tags": []}
        if isinstance(wiki_data, Exception):
            wiki_data = {"is_soundtrack": False, "is_film_album": False, "industry_hints": []}
        
        industry = detect_industry_enhanced(song.artist, song.title, song.album, mb_data, wiki_data)
        
        # Build context
        context = f"Song: '{song.title}' by {song.artist}"
        if song.album:
            context += f" from album '{song.album}'"
        if mb_data.get("found"):
            context += f", Labels: {', '.join(mb_data['labels'][:2])}"
        if wiki_data.get("is_film_album"):
            context += ", Type: Film Soundtrack"
        
        # AI prompt - WORKING VERSION FROM OLD CODE
        prompt = f"""Analyze: {context}
Detected Industry: {industry}

TASK 1: MOOD
   Options: Aggressive, Energetic, Romantic, Melancholic, Spiritual, Chill, Uplifting

TASK 2: LANGUAGE
   Detect primary language (Hindi, Telugu, Tamil, Punjabi, English, etc.)

TASK 3: GENRE (Simple categories)
   Options: Party, Pop, Rock, Hip-Hop, Folk, Devotional, Classical, LoFi, EDM, Jazz

OUTPUT JSON:
{{
  "mood": "...",
  "language": "...",
  "genre": "..."
}}"""

        try:
            logger.info(f"🤖 Enriching: '{song.title}' by {song.artist}")
            
            chat_completion = state.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a music metadata classifier. Output strict JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model=config.GROQ_MODEL,
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            content = chat_completion.choices[0].message.content.strip()
            ai_data = json.loads(content)
            
            mood = ai_data.get("mood", "Neutral").strip().title()
            language = ai_data.get("language", "Unknown").strip().title()
            genre = ai_data.get("genre", "Pop").strip().title()
            
            logger.info(f"✅ AI: mood={mood}, language={language}, genre={genre}")
            
        except Exception as e:
            logger.error(f"❌ AI error: {e}")
            mood, language, genre = "Neutral", "Unknown", "Pop"
        
        result = {
            "mood": mood,
            "language": language,
            "industry": industry,
            "genre": genre,
            "formatted": f"{mood};{language};{industry};{genre}",
            "sources_used": {
                "musicbrainz": mb_data.get("found", False),
                "wikipedia": wiki_data.get("is_film_album", False),
                "ai": True
            }
        }
        
        await set_cached_metadata(cache_key, result)
        logger.info(f"✅ Enrichment complete: {result['formatted']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Enrichment error: {e}", exc_info=True)
        raise HTTPException(500, f"Enrichment failed: {str(e)}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
