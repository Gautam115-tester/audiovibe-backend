from fastapi import FastAPI, HTTPException, Header, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import os
import json
import httpx
import hashlib
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from functools import lru_cache
import redis
from contextlib import asynccontextmanager

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Class
class Config:
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    APP_SECRET: str = os.getenv("APP_INTEGRITY_SECRET", "change_this_secret")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    
    # Rate Limiting
    RATE_LIMIT_TRACKS: str = "30/minute"
    RATE_LIMIT_STREAM: str = "20/minute"
    RATE_LIMIT_ENRICH: str = "10/minute"
    
    # Timeouts (seconds)
    EXTERNAL_API_TIMEOUT: int = 5
    REQUEST_TIMEOUT: int = 30
    
    # Cache TTL (seconds)
    METADATA_CACHE_TTL: int = 3600  # 1 hour
    
    # Stream Expiration
    SHORT_STREAM_EXPIRY: int = 480   # 8 minutes
    LONG_STREAM_EXPIRY: int = 5400   # 90 minutes
    LONG_TRACK_THRESHOLD: int = 900000  # 15 minutes in ms
    
    # Integrity Check
    INTEGRITY_WINDOW: int = 30  # seconds
    
    # Models
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

config = Config()

# ============================================================================
# LIFESPAN & GLOBAL STATE
# ============================================================================

class AppState:
    supabase: Optional[Client] = None
    redis_client: Optional[redis.Redis] = None
    http_client: Optional[httpx.AsyncClient] = None
    groq_client: Optional[Groq] = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting AudioVibe API...")
    
    try:
        # Initialize Supabase
        if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("Supabase credentials missing")
        state.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
        
        # Initialize Groq
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing")
        state.groq_client = Groq(api_key=config.GROQ_API_KEY)
        
        # Initialize Redis (optional)
        try:
            state.redis_client = redis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            state.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable, using in-memory cache: {e}")
            state.redis_client = None
        
        # Initialize HTTP Client with connection pooling
        state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.EXTERNAL_API_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")
    if state.http_client:
        await state.http_client.aclose()
    if state.redis_client:
        state.redis_client.close()

# ============================================================================
# APP INITIALIZATION
# ============================================================================

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="AudioVibe Secure API",
    version="9.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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
    
    @validator('artist', 'title', 'album')
    def clean_whitespace(cls, v):
        return v.strip() if v else v

class TrackUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    artist: str = Field(..., min_length=1, max_length=200)
    album: str = Field(..., max_length=200)
    audio_file_url: str = Field(..., min_length=1)
    cover_image_url: Optional[str] = None
    duration_ms: int = Field(..., ge=0)
    genres: List[str] = Field(default_factory=list)
    tier_required: str = Field(default="free", regex="^(free|premium)$")

class PlayRecord(BaseModel):
    user_id: str = Field(..., min_length=1)
    track_id: str = Field(..., min_length=1)
    listen_time_ms: int = Field(..., ge=0)
    total_duration_ms: int = Field(..., gt=0)

# ============================================================================
# SECURITY DEPENDENCIES
# ============================================================================

def verify_app_integrity(
    x_app_integrity: str = Header(..., description="SHA256 hash for integrity"),
    x_app_timestamp: str = Header(..., description="Unix timestamp")
) -> Dict[str, Any]:
    """Verify request integrity and prevent replay attacks"""
    try:
        request_time = int(x_app_timestamp)
        current_time = int(time.time())
        
        # Check timestamp freshness
        if abs(current_time - request_time) > config.INTEGRITY_WINDOW:
            logger.warning(f"Expired request: timestamp={request_time}, current={current_time}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Request expired"
            )
        
        # Verify integrity hash
        expected_hash = hashlib.sha256(
            f"{config.APP_SECRET}{x_app_timestamp}".encode()
        ).hexdigest()
        
        if x_app_integrity != expected_hash:
            logger.warning("Integrity check failed")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Integrity verification failed"
            )
        
        return {"timestamp": request_time, "verified": True}
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid timestamp format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Integrity check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Security verification failed"
        )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_ready():
    """Verify all critical services are available"""
    if not state.supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable"
        )
    if not state.groq_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service unavailable"
        )

async def get_cached_metadata(key: str) -> Optional[Dict]:
    """Get metadata from Redis or memory"""
    if state.redis_client:
        try:
            data = state.redis_client.get(f"metadata:{key}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
    return None

async def set_cached_metadata(key: str, value: Dict) -> None:
    """Set metadata in Redis with TTL"""
    if state.redis_client:
        try:
            state.redis_client.setex(
                f"metadata:{key}",
                config.METADATA_CACHE_TTL,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")

async def search_musicbrainz(artist: str, title: str, album: Optional[str] = None) -> Dict:
    """Search MusicBrainz API asynchronously"""
    try:
        base_url = "https://musicbrainz.org/ws/2/recording/"
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        
        if album and album.lower() not in ["unknown", ""]:
            query_parts.append(f'release:"{album}"')
        
        query = " AND ".join(query_parts)
        params = {"query": query, "fmt": "json", "limit": 3}
        headers = {"User-Agent": "AudioVibe/9.0 (contact@audiovibe.app)"}
        
        response = await state.http_client.get(
            base_url,
            params=params,
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if recordings := data.get("recordings"):
                recording = recordings[0]
                releases = recording.get("releases", [])
                tags = [tag["name"] for tag in recording.get("tags", [])]
                
                labels = []
                release_types = []
                
                for release in releases[:3]:
                    if label_info := release.get("label-info"):
                        labels.extend([
                            li.get("label", {}).get("name")
                            for li in label_info
                            if li.get("label", {}).get("name")
                        ])
                    if rel_type := release.get("status"):
                        release_types.append(rel_type)
                
                return {
                    "found": True,
                    "release_type": release_types[0] if release_types else "Unknown",
                    "labels": list(set(labels))[:3],
                    "tags": tags[:5]
                }
    except httpx.TimeoutException:
        logger.warning("MusicBrainz API timeout")
    except Exception as e:
        logger.error(f"MusicBrainz error: {e}")
    
    return {"found": False, "release_type": "Unknown", "labels": [], "tags": []}

async def search_wikipedia(artist: str, album: str) -> Dict:
    """Search Wikipedia API asynchronously"""
    try:
        if not album or album.lower() in ["unknown", ""]:
            return {"is_soundtrack": False, "is_film_album": False, "industry_hints": []}
        
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{album} {artist} album soundtrack",
            "format": "json",
            "srlimit": 3
        }
        
        response = await state.http_client.get(search_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("query", {}).get("search", [])
            
            for result in results:
                snippet = result.get("snippet", "").lower()
                title = result.get("title", "").lower()
                combined = f"{snippet} {title}"
                
                is_soundtrack = any(
                    word in combined
                    for word in ["soundtrack", "film", "movie", "cinema"]
                )
                
                industry_hints = []
                if any(word in combined for word in ["bollywood", "hindi film", "mumbai"]):
                    industry_hints.append("Bollywood")
                if any(word in combined for word in ["tollywood", "telugu film", "tamil film"]):
                    industry_hints.append("Tollywood")
                if any(word in combined for word in ["punjabi music", "punjabi album"]):
                    industry_hints.append("Punjabi")
                
                if is_soundtrack or industry_hints:
                    return {
                        "is_soundtrack": is_soundtrack,
                        "is_film_album": is_soundtrack,
                        "industry_hints": industry_hints
                    }
    except httpx.TimeoutException:
        logger.warning("Wikipedia API timeout")
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
    
    return {"is_soundtrack": False, "is_film_album": False, "industry_hints": []}

def detect_industry_enhanced(
    artist: str,
    title: str,
    album: Optional[str],
    musicbrainz_data: Dict,
    wiki_data: Dict
) -> str:
    """Detect music industry with enhanced logic"""
    if industry_hints := wiki_data.get("industry_hints"):
        return industry_hints[0]
    
    artist_lower = artist.lower()
    album_lower = album.lower() if album else ""
    labels = " ".join(musicbrainz_data.get("labels", [])).lower()
    tags = " ".join(musicbrainz_data.get("tags", [])).lower()
    combined = f"{artist_lower} {album_lower} {labels} {tags}"
    
    industry_keywords = {
        "Bollywood": ["bollywood", "hindi", "mumbai"],
        "Tollywood": ["tollywood", "telugu", "tamil"],
        "Punjabi": ["punjabi"],
    }
    
    for industry, keywords in industry_keywords.items():
        if any(keyword in combined for keyword in keywords):
            return industry
    
    return "International"

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "9.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service status"""
    services = {
        "database": False,
        "redis": False,
        "ai": False
    }
    
    # Check Supabase
    try:
        if state.supabase:
            state.supabase.table("music_tracks").select("id").limit(1).execute()
            services["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    # Check Redis
    if state.redis_client:
        try:
            state.redis_client.ping()
            services["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
    
    # Check Groq
    services["ai"] = state.groq_client is not None
    
    all_healthy = all(services.values()) or (services["database"] and services["ai"])
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/tracks/metadata", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_TRACKS)
async def get_tracks_metadata(request: Request):
    """Get tracks metadata (no audio URLs)"""
    ensure_ready()
    
    try:
        response = state.supabase.table("music_tracks").select(
            "id, title, artist, album, cover_image_url, duration_ms, genres, tier_required, created_at"
        ).order("created_at", desc=True).execute()
        
        return {
            "status": "success",
            "count": len(response.data),
            "tracks": response.data
        }
    except Exception as e:
        logger.error(f"Error fetching tracks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch tracks"
        )

@app.get("/tracks/stream/{track_id}", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_STREAM)
async def get_stream_url(request: Request, track_id: str):
    """Generate secure signed URL for streaming"""
    ensure_ready()
    
    try:
        # Fetch track details
        response = state.supabase.table("music_tracks").select(
            "audio_file_url, duration_ms, title"
        ).eq("id", track_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Track not found"
            )
        
        track = response.data[0]
        duration_ms = track.get('duration_ms', 0)
        
        # Dynamic expiration based on duration
        is_long_content = (
            not duration_ms or
            duration_ms == 0 or
            duration_ms > config.LONG_TRACK_THRESHOLD
        )
        
        expires_in = (
            config.LONG_STREAM_EXPIRY if is_long_content
            else config.SHORT_STREAM_EXPIRY
        )
        
        # Extract file path
        raw_url = track['audio_file_url']
        if "/music_files/" in raw_url:
            file_path = raw_url.split("/music_files/")[1]
        else:
            file_path = raw_url
        
        # Generate signed URL
        signed_response = state.supabase.storage.from_("music_files").create_signed_url(
            file_path,
            expires_in
        )
        
        return {
            "stream_url": signed_response["signedURL"],
            "expires_in": expires_in,
            "expires_at": int(time.time()) + expires_in,
            "track_id": track_id,
            "content_type": "series" if is_long_content else "music"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream URL generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate stream URL"
        )

@app.post("/tracks", dependencies=[Depends(verify_app_integrity)])
async def add_track(track: TrackUpload):
    """
    Add new track (Admin only - should add role check)
    TODO: Implement proper admin authentication
    """
    ensure_ready()
    
    try:
        data = track.dict()
        response = state.supabase.table("music_tracks").insert(data).execute()
        
        logger.info(f"Track added: {track.title} by {track.artist}")
        
        return {
            "status": "success",
            "message": "Track added successfully",
            "data": response.data[0] if response.data else None
        }
    except Exception as e:
        logger.error(f"Error adding track: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add track"
        )

@app.post("/record-play", dependencies=[Depends(verify_app_integrity)])
async def record_play(stat: PlayRecord):
    """Record listening statistics"""
    ensure_ready()
    
    try:
        completion_rate = (
            min(1.0, stat.listen_time_ms / stat.total_duration_ms)
            if stat.total_duration_ms > 0 else 0
        )
        
        state.supabase.rpc('upsert_listening_stat', {
            'p_user_id': stat.user_id,
            'p_track_id': stat.track_id,
            'p_listen_time_ms': stat.listen_time_ms,
            'p_completion_rate': completion_rate
        }).execute()
        
        return {
            "status": "success",
            "completion_rate": round(completion_rate * 100, 2)
        }
    except Exception as e:
        logger.error(f"Error recording play: {e}")
        return {"status": "error", "message": "Failed to record play"}

@app.post("/enrich-metadata", dependencies=[Depends(verify_app_integrity)])
@limiter.limit(config.RATE_LIMIT_ENRICH)
async def enrich_metadata(request: Request, song: SongRequest):
    """AI-powered metadata enrichment"""
    ensure_ready()
    
    # Generate cache key
    cache_key = f"{song.artist.lower()}:{song.title.lower()}:{(song.album or '').lower()}"
    
    # Check cache
    cached = await get_cached_metadata(cache_key)
    if cached:
        logger.info(f"Cache hit for: {cache_key}")
        return cached
    
    try:
        # Fetch external data concurrently
        musicbrainz_task = search_musicbrainz(song.artist, song.title, song.album)
        wiki_task = search_wikipedia(song.artist, song.album or "")
        
        musicbrainz_data, wiki_data = await asyncio.gather(
            musicbrainz_task,
            wiki_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(musicbrainz_data, Exception):
            logger.error(f"MusicBrainz error: {musicbrainz_data}")
            musicbrainz_data = {"found": False, "release_type": "Unknown", "labels": [], "tags": []}
        
        if isinstance(wiki_data, Exception):
            logger.error(f"Wikipedia error: {wiki_data}")
            wiki_data = {"is_soundtrack": False, "is_film_album": False, "industry_hints": []}
        
        # Detect industry
        industry = detect_industry_enhanced(
            song.artist, song.title, song.album,
            musicbrainz_data, wiki_data
        )
        
        # AI enrichment
        context_info = f"Artist: {song.artist}, Title: {song.title}"
        if song.album:
            context_info += f", Album: {song.album}"
        
        prompt = (
            f"Analyze: {context_info}\n"
            f"Detected Industry: {industry}\n"
            "OUTPUT JSON: {\"mood\": \"...\", \"language\": \"...\", \"genre\": \"...\"}"
        )
        
        try:
            chat_completion = state.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a music metadata classifier. Output strict JSON only with keys: mood, language, genre."
                    },
                    {"role": "user", "content": prompt}
                ],
                model=config.GROQ_MODEL,
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            content = chat_completion.choices[0].message.content.strip()
            ai_data = json.loads(content)
            
            mood = ai_data.get("mood", "Neutral").title()
            language = ai_data.get("language", "Unknown").title()
            genre = ai_data.get("genre", "Pop").title()
            
        except Exception as e:
            logger.error(f"Groq AI error: {e}")
            mood, language, genre = "Neutral", "Unknown", "Pop"
        
        result = {
            "formatted": f"{mood};{language};{industry};{genre}",
            "mood": mood,
            "language": language,
            "industry": industry,
            "genre": genre,
            "sources_used": {
                "musicbrainz": musicbrainz_data.get("found", False),
                "wikipedia": wiki_data.get("is_film_album", False),
                "ai": True
            }
        }
        
        # Cache result
        await set_cached_metadata(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Metadata enrichment error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enrich metadata"
        )

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
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
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
    import asyncio
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
