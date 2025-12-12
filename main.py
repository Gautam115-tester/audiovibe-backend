from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from collections import defaultdict
import hashlib
import os
import json
import requests
import time
import re
import base64
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES & SETUP
# ============================================================
load_dotenv()
app = FastAPI(title="AudioVibe Secure API", version="9.0.0-SpotifyPowered")

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

# Enhanced cache with TTL
metadata_cache = {}
CACHE_TTL = 86400  # 24 hours

# Spotify Token Cache
spotify_token_cache = {"token": None, "expires_at": 0}

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
    print("✅ System Online: Spotify-Powered Metadata Engine Ready")
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
# 🛡️ MIDDLEWARE (SAME AS BEFORE)
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
    
    if not all([timestamp, integrity_hash, device_id, app_version]):
        return JSONResponse(status_code=403, content={"detail": "Missing security headers"})
    
    if app_version not in ALLOWED_APP_VERSIONS:
        return JSONResponse(status_code=403, content={"detail": f"Unsupported version: {app_version}"})
    
    try:
        request_time = int(timestamp)
        current_time = int(time.time())
        if abs(current_time - request_time) > 300:
            return JSONResponse(status_code=403, content={"detail": "Request expired"})
    except ValueError:
        return JSONResponse(status_code=403, content={"detail": "Invalid timestamp"})
    
    payload = f"{APP_INTEGRITY_SECRET}{timestamp}{device_id}{app_version}"
    expected_hash = hashlib.sha256(payload.encode()).hexdigest()
    if integrity_hash != expected_hash:
        return JSONResponse(status_code=403, content={"detail": "Invalid signature"})
    
    current_time = time.time()
    rate_limit_storage[device_id] = [t for t in rate_limit_storage[device_id] 
                                     if current_time - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_storage[device_id]) >= RATE_LIMIT_MAX_REQUESTS:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    rate_limit_storage[device_id].append(current_time)
    
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🎵 SPOTIFY API INTEGRATION
# ============================================================

def get_spotify_token() -> Optional[str]:
    """Get Spotify access token using Client Credentials flow"""
    global spotify_token_cache
    
    # Return cached token if still valid
    if spotify_token_cache["token"] and time.time() < spotify_token_cache["expires_at"]:
        return spotify_token_cache["token"]
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("⚠️ Spotify credentials not configured")
        return None
    
    try:
        # Encode credentials
        auth_str = f"{client_id}:{client_secret}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()
        
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={
                "Authorization": f"Basic {b64_auth}",
                "Content-Type": "application/x-www-form-urlencoded"
            },
            data={"grant_type": "client_credentials"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data["access_token"]
            expires_in = data["expires_in"]
            
            # Cache token with 5-minute buffer
            spotify_token_cache["token"] = token
            spotify_token_cache["expires_at"] = time.time() + expires_in - 300
            
            print("✅ Spotify token refreshed")
            return token
        else:
            print(f"❌ Spotify auth failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Spotify token error: {e}")
        return None

def search_spotify(artist: str, title: str, album: Optional[str] = None) -> Dict:
    """Search Spotify for track metadata with audio features"""
    token = get_spotify_token()
    if not token:
        return {"found": False}
    
    try:
        # Build search query
        query_parts = [f'track:"{title}"', f'artist:"{artist}"']
        if album and album.lower() not in ["unknown", "single", ""]:
            query_parts.append(f'album:"{album}"')
        
        query = " ".join(query_parts)
        
        # Search for track
        search_response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "q": query,
                "type": "track",
                "limit": 5,
                "market": "IN"  # Indian market for better local results
            },
            timeout=10
        )
        
        if search_response.status_code != 200:
            print(f"Spotify search failed: {search_response.status_code}")
            return {"found": False}
        
        search_data = search_response.json()
        tracks = search_data.get("tracks", {}).get("items", [])
        
        if not tracks:
            return {"found": False}
        
        # Get the best match (first result)
        track = tracks[0]
        track_id = track["id"]
        
        # Get audio features (valence, energy, danceability, etc.)
        features_response = requests.get(
            f"https://api.spotify.com/v1/audio-features/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        
        # Get artist details for genres
        artist_id = track["artists"][0]["id"]
        artist_response = requests.get(
            f"https://api.spotify.com/v1/artists/{artist_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        
        audio_features = {}
        if features_response.status_code == 200:
            audio_features = features_response.json()
        
        artist_data = {}
        if artist_response.status_code == 200:
            artist_data = artist_response.json()
        
        return {
            "found": True,
            "track": {
                "id": track_id,
                "name": track["name"],
                "popularity": track.get("popularity", 0),
                "explicit": track.get("explicit", False),
                "duration_ms": track.get("duration_ms", 0)
            },
            "audio_features": {
                "valence": audio_features.get("valence", 0.5),  # 0-1 (happiness)
                "energy": audio_features.get("energy", 0.5),  # 0-1 (intensity)
                "danceability": audio_features.get("danceability", 0.5),  # 0-1
                "acousticness": audio_features.get("acousticness", 0.5),  # 0-1
                "instrumentalness": audio_features.get("instrumentalness", 0),  # 0-1
                "speechiness": audio_features.get("speechiness", 0),  # 0-1
                "tempo": audio_features.get("tempo", 120),  # BPM
                "key": audio_features.get("key", -1),  # 0-11 (C, C#, D, etc.)
                "mode": audio_features.get("mode", 1)  # 0=minor, 1=major
            },
            "artist": {
                "name": track["artists"][0]["name"],
                "genres": artist_data.get("genres", []),
                "popularity": artist_data.get("popularity", 0)
            },
            "album": {
                "name": track["album"]["name"],
                "release_date": track["album"].get("release_date", ""),
                "type": track["album"].get("album_type", "")
            }
        }
        
    except Exception as e:
        print(f"Spotify API error: {e}")
        return {"found": False}

# ============================================================
# 🎯 ENHANCED HELPER FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    """Remove special chars, brackets, feat. annotations"""
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    text = re.sub(r'feat\.|ft\.|featuring', '', text, flags=re.IGNORECASE)
    return text.strip()

def detect_language_from_script(text: str) -> str:
    """Detect language from Unicode script"""
    if re.search(r'[\u0900-\u097F]', text):  # Devanagari
        return "Hindi"
    elif re.search(r'[\u0C00-\u0C7F]', text):  # Telugu
        return "Telugu"
    elif re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
        return "Tamil"
    elif re.search(r'[\u0A00-\u0A7F]', text):  # Gurmukhi
        return "Punjabi"
    elif re.search(r'[\u0D00-\u0D7F]', text):  # Malayalam
        return "Malayalam"
    return "Unknown"

def detect_mood_from_audio_features(features: Dict) -> str:
    """Detect mood using Spotify's audio features"""
    valence = features.get("valence", 0.5)
    energy = features.get("energy", 0.5)
    danceability = features.get("danceability", 0.5)
    acousticness = features.get("acousticness", 0.5)
    tempo = features.get("tempo", 120)
    
    # Aggressive: High energy, low valence, fast tempo
    if energy > 0.75 and valence < 0.4 and tempo > 140:
        return "Aggressive"
    
    # Energetic: High energy, high danceability
    if energy > 0.7 and danceability > 0.65:
        return "Energetic"
    
    # Romantic: Moderate energy, moderate valence, acoustic
    if 0.3 < energy < 0.6 and 0.4 < valence < 0.7 and acousticness > 0.3:
        return "Romantic"
    
    # Melancholic: Low valence, low energy
    if valence < 0.35 and energy < 0.5:
        return "Melancholic"
    
    # Chill: Low energy, moderate valence
    if energy < 0.5 and 0.4 < valence < 0.7:
        return "Chill"
    
    # Uplifting: High valence, moderate-high energy
    if valence > 0.7 and energy > 0.5:
        return "Uplifting"
    
    return "Neutral"

def detect_genre_from_spotify(spotify_data: Dict) -> str:
    """Detect genre from Spotify artist genres and audio features"""
    genres = spotify_data.get("artist", {}).get("genres", [])
    features = spotify_data.get("audio_features", {})
    
    genres_text = " ".join(genres).lower()
    
    # Check explicit genre matches
    if any(g in genres_text for g in ["hip hop", "rap", "trap"]):
        return "Hip-Hop"
    
    if any(g in genres_text for g in ["edm", "house", "techno", "electronic", "dance"]):
        return "EDM"
    
    if any(g in genres_text for g in ["folk", "traditional", "acoustic"]):
        return "Folk"
    
    if any(g in genres_text for g in ["devotional", "bhajan", "spiritual"]):
        return "Devotional"
    
    if any(g in genres_text for g in ["lo-fi", "lofi", "chill"]):
        return "LoFi"
    
    if any(g in genres_text for g in ["rock", "metal", "alternative"]):
        return "Rock"
    
    if any(g in genres_text for g in ["jazz", "blues"]):
        return "Jazz"
    
    if any(g in genres_text for g in ["classical", "raga"]):
        return "Classical"
    
    # Use audio features as fallback
    danceability = features.get("danceability", 0)
    energy = features.get("energy", 0)
    
    if danceability > 0.75 and energy > 0.7:
        return "Party"
    
    return "Pop"

def search_lastfm(artist: str, title: str) -> Dict:
    """Last.fm API - Better than MusicBrainz for metadata"""
    try:
        api_key = os.getenv("LASTFM_API_KEY", "")
        if not api_key:
            return {"found": False}
        
        response = requests.get(
            "http://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getInfo",
                "api_key": api_key,
                "artist": artist,
                "track": title,
                "format": "json"
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if "track" in data:
                track = data["track"]
                return {
                    "found": True,
                    "tags": [t["name"] for t in track.get("toptags", {}).get("tag", [])[:5]],
                    "listeners": track.get("listeners", 0),
                    "playcount": track.get("playcount", 0)
                }
    except Exception as e:
        print(f"Last.fm Error: {e}")
    return {"found": False}

def search_musicbrainz_enhanced(artist: str, title: str, album: Optional[str] = None) -> dict:
    """Enhanced MusicBrainz with better parsing"""
    try:
        base_url = "https://musicbrainz.org/ws/2/recording/"
        
        clean_artist = clean_text(artist)
        clean_title = clean_text(title)
        
        query_parts = [f'recording:"{clean_title}"', f'artist:"{clean_artist}"']
        if album and album.lower() not in ["unknown", "single"]:
            query_parts.append(f'release:"{clean_text(album)}"')
        
        headers = {"User-Agent": "AudioVibe/9.0 (contact@audiovibe.app)"}
        response = requests.get(
            base_url,
            params={
                "query": " AND ".join(query_parts),
                "fmt": "json",
                "limit": 5
            },
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("recordings"):
                rec = data["recordings"][0]
                
                labels = []
                for rel in rec.get("releases", [])[:5]:
                    if "label-info" in rel:
                        labels.extend([l.get("label", {}).get("name") 
                                     for l in rel["label-info"] if l.get("label")])
                
                tags = [t["name"].lower() for t in rec.get("tags", [])][:10]
                
                return {
                    "found": True,
                    "labels": list(set(labels))[:5],
                    "tags": tags
                }
    except Exception as e:
        print(f"MusicBrainz Error: {e}")
    return {"found": False, "labels": [], "tags": []}

def detect_industry_advanced(artist: str, album: str, title: str, 
                            spotify_data: Dict, mb_data: dict, lastfm_data: dict) -> str:
    """Enhanced industry detection with Spotify genres"""
    combined_text = f"{artist} {album} {title}".lower()
    
    # Get Spotify genres
    spotify_genres = " ".join(spotify_data.get("artist", {}).get("genres", [])).lower()
    
    # Check MusicBrainz labels
    labels_text = " ".join(mb_data.get("labels", [])).lower()
    
    # Check Last.fm tags
    tags_text = " ".join(lastfm_data.get("tags", [])).lower()
    
    # Full context
    full_context = f"{combined_text} {spotify_genres} {labels_text} {tags_text}"
    
    # Language detection from script
    script_lang = detect_language_from_script(f"{artist} {title}")
    
    # Bollywood signals
    if any(k in full_context for k in ["bollywood", "hindi film", "filmi", "t-series", "zee music", 
                                       "yrf", "eros", "sony music india"]):
        return "Bollywood"
    if script_lang == "Hindi" and any(k in full_context for k in ["film", "movie", "soundtrack", "ost"]):
        return "Bollywood"
    
    # Telugu/Tollywood
    if any(k in full_context for k in ["tollywood", "telugu", "aditya music", "lahari", "telugu film"]):
        return "Tollywood"
    if script_lang == "Telugu":
        return "Tollywood"
    
    # Tamil/Kollywood
    if any(k in full_context for k in ["kollywood", "tamil", "sony music south", "think music", "tamil film"]):
        return "Kollywood"
    if script_lang == "Tamil":
        return "Kollywood"
    
    # Punjabi
    if any(k in full_context for k in ["punjabi", "speed records", "white hill", "jatt", "punjabi pop"]):
        return "Punjabi"
    if script_lang == "Punjabi":
        return "Punjabi"
    
    # Malayalam
    if script_lang == "Malayalam" or "malayalam" in full_context:
        return "Mollywood"
    
    # Indie (if indie genres in Spotify)
    if any(k in spotify_genres for k in ["indie", "independent"]) or "indie" in labels_text:
        return "Indie"
    
    # Default to International for Western content
    return "International"

def analyze_with_groq_enhanced(artist: str, title: str, album: str, 
                              spotify_data: Dict, mb_data: dict, 
                              lastfm_data: dict, python_hint: str,
                              spotify_mood: str, spotify_genre: str) -> dict:
    """Enhanced GROQ analysis with Spotify audio features"""
    
    # Build rich context
    context_parts = [
        f"Artist: {artist}",
        f"Title: {title}",
        f"Album: {album}",
        f"Python Industry Hint: {python_hint}",
        f"Spotify Audio Analysis Mood: {spotify_mood}",
        f"Spotify Audio Analysis Genre: {spotify_genre}"
    ]
    
    if spotify_data.get("found"):
        genres = spotify_data.get("artist", {}).get("genres", [])
        if genres:
            context_parts.append(f"Spotify Artist Genres: {', '.join(genres[:5])}")
        
        features = spotify_data.get("audio_features", {})
        context_parts.append(
            f"Audio Features: Valence={features.get('valence', 0):.2f} "
            f"(happiness), Energy={features.get('energy', 0):.2f}, "
            f"Danceability={features.get('danceability', 0):.2f}"
        )
    
    if mb_data.get("found"):
        context_parts.append(f"Record Labels: {', '.join(mb_data['labels'][:3])}")
        context_parts.append(f"MB Tags: {', '.join(mb_data['tags'][:5])}")
    
    if lastfm_data.get("found"):
        context_parts.append(f"Last.fm Tags: {', '.join(lastfm_data['tags'][:5])}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Analyze this song and return ONLY valid JSON:

SONG DATA:
{context}

CLASSIFICATION RULES:

1. MOOD: Use Spotify Audio Analysis as PRIMARY source. Pick ONE:
   [Aggressive, Energetic, Romantic, Melancholic, Chill, Uplifting]

2. LANGUAGE (lyrics language):
   - Hindi songs → Hindi (even if title is English)
   - Punjabi songs → Punjabi
   - Telugu → Telugu, Tamil → Tamil
   - Western artists → English
   Examples: "Tum Hi Ho" → Hindi, "Brown Munde" → Punjabi

3. GENRE: Use Spotify Analysis + Artist Genres. Pick ONE:
   [Party, Pop, Hip-Hop, Folk, Devotional, LoFi, EDM, Rock, Jazz, Classical]

4. INDUSTRY: TRUST Python Hint for Indian content:
   - Bollywood (Hindi film), Tollywood (Telugu), Kollywood (Tamil)
   - Punjabi (Punjabi pop/film), Mollywood (Malayalam)
   - Indie (independent), International (Western)

OUTPUT (raw JSON):
{{"mood": "{spotify_mood}", "language": "Hindi", "genre": "{spotify_genre}", "industry": "{python_hint}"}}

Adjust ONLY if context strongly contradicts Spotify analysis.
"""

    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.15,
            response_format={"type": "json_object"}
        )
        
        content = chat.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        if content.startswith("json"):
            content = content[4:].strip()
        
        data = json.loads(content)
        
        return {
            "mood": data.get("mood", spotify_mood).title(),
            "language": data.get("language", "Unknown").title(),
            "genre": data.get("genre", spotify_genre).title(),
            "industry": data.get("industry", python_hint).title()
        }
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON Parse Error: {e}")
        return {
            "mood": spotify_mood,
            "language": "Unknown",
            "genre": spotify_genre,
            "industry": python_hint
        }
    except Exception as e:
        print(f"⚠️ Groq Error: {e}")
        return {
            "mood": spotify_mood,
            "language": "Unknown",
            "genre": spotify_genre,
            "industry": python_hint
        }

def ensure_ready():
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Backend Maintenance: {startup_error}")
    if not supabase or not groq_client:
        raise HTTPException(status_code=503, detail="Backend Initializing")

# ============================================================
# 📌 ENHANCED API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    status = "Maintenance Mode" if startup_error else "Active"
    spotify_status = "✅ Configured" if os.getenv("SPOTIFY_CLIENT_ID") else "⚠️ Not Configured"
    
    return {
        "status": status,
        "version": "9.0.0-SpotifyPowered",
        "features": [
            "🎵 Spotify Audio Features",
            "🎭 AI Mood Detection",
            "🌍 Multi-Source Intelligence",
            "📊 Real-time Audio Analysis"
        ],
        "integrations": {
            "spotify": spotify_status,
            "lastfm": "✅" if os.getenv("LASTFM_API_KEY") else "⚠️",
            "musicbrainz": "✅",
            "groq": "✅"
        },
        "error": startup_error
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if not startup_error else "unhealthy",
        "spotify_token": "valid" if spotify_token_cache.get("token") else "not_cached",
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
    
    clean_title = clean_text(song.title)
    clean_artist = clean_text(song.artist)
    clean_album = clean_text(song.album) if song.album else ""
    
    # Check cache with TTL
    cache_key = f"{clean_artist.lower()}:{clean_title.lower()}:{clean_album.lower()}"
    if cache_key in metadata_cache:
        cached = metadata_cache[cache_key]
        if time.time() - cached.get("cached_at", 0) < CACHE_TTL:
            print(f"💾 Cache Hit: {clean_title}")
            return cached["data"]
    
    print(f"🎵 Spotify-Powered Analysis: {clean_title} by {clean_artist}")
    
    # 🎵 PRIMARY: Spotify (best audio features)
    spotify_data = search_spotify(clean_artist, clean_title, clean_album)
    
    # 📡 SECONDARY: MusicBrainz + Last.fm
    mb_data = search_musicbrainz_enhanced(clean_artist, clean_title, clean_album)
    lastfm_data = search_lastfm(clean_artist, clean_title)
    
    # 🎭 Extract Spotify-based predictions
    spotify_mood = "Neutral"
    spotify_genre = "Pop"
    
    if spotify_data.get("found"):
        spotify_mood = detect_mood_from_audio_features(spotify_data.get("audio_features", {}))
        spotify_genre = detect_genre_from_spotify(spotify_data)
    
    # 🌍 Advanced industry detection
    python_hint = detect_industry_advanced(
        clean_artist, clean_album, clean_title,
        spotify_data, mb_data, lastfm_data
    )
    
    # 🤖 AI refinement with Spotify context
    ai_result = analyze_with_groq_enhanced(
        clean_artist, clean_title, clean_album,
        spotify_data, mb_data, lastfm_data,
        python_hint, spotify_mood, spotify_genre
    )
    
    # 📊 Confidence scoring
    confidence_score = 0
    sources_used = []
    
    if spotify_data.get("found"):
        confidence_score += 50
        sources_used.append("spotify")
    if mb_data.get("found"):
        confidence_score += 20
        sources_used.append("musicbrainz")
    if lastfm_data.get("found"):
        confidence_score += 15
        sources_used.append("lastfm")
    confidence_score += 15  # AI analysis
    
    # Final result
    result = {
        "formatted": f"{ai_result['mood']};{ai_result['language']};{ai_result['industry']};{ai_result['genre']}",
        "mood": ai_result["mood"],
        "language": ai_result["language"],
        "industry": ai_result["industry"],
        "genre": ai_result["genre"],
        "confidence_score": confidence_score,
        "sources_used": sources_used,
        "spotify_insights": {
            "audio_mood": spotify_mood,
            "audio_genre": spotify_genre,
            "popularity": spotify_data.get("track", {}).get("popularity", 0) if spotify_data.get("found") else 0,
            "audio_features": spotify_data.get("audio_features", {}) if spotify_data.get("found") else {}
        },
        "debug": {
            "mb_labels": mb_data.get("labels", [])[:3],
            "lastfm_tags": lastfm_data.get("tags", [])[:3],
            "python_hint": python_hint
        }
    }
    
    # Cache with timestamp
    metadata_cache[cache_key] = {
        "data": result,
        "cached_at": time.time()
    }
    
    return result

# ============================================================
# OTHER ENDPOINTS (SAME AS BEFORE)
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
        response = supabase.table("music_tracks").insert(track.dict()).execute()
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
