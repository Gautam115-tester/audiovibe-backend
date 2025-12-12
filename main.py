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
# LOAD ENVIRONMENT & SETUP
# ============================================================
load_dotenv()
app = FastAPI(title="AudioVibe API", version="11.0.0-PureAI")

APP_INTEGRITY_SECRET = os.getenv("APP_INTEGRITY_SECRET", "DONOTTOUCHAPI")
ALLOWED_APP_VERSIONS = ["1.0", "1.0.0", "1.0.1", "1.1.0"]
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 100
rate_limit_storage = defaultdict(list)

supabase: Optional[Client] = None
groq_client: Optional[Groq] = None
startup_error: Optional[str] = None
GROQ_MODELS = {"primary": "llama-3.3-70b-versatile"}

metadata_cache = {}
CACHE_TTL = 86400
spotify_token_cache = {"token": None, "expires_at": 0}

try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ System Online: Pure AI-Powered Detection")
except Exception as e:
    startup_error = str(e)
    print(f"❌ Startup Warning: {e}")

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
# MIDDLEWARE
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
        return JSONResponse(status_code=403, content={"detail": "Missing headers"})
    
    if app_version not in ALLOWED_APP_VERSIONS:
        return JSONResponse(status_code=403, content={"detail": "Unsupported version"})
    
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

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
                   allow_methods=["*"], allow_headers=["*"])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    text = re.sub(r'feat\.|ft\.|featuring', '', text, flags=re.IGNORECASE)
    return text.strip()

def get_spotify_token() -> Optional[str]:
    global spotify_token_cache
    
    if spotify_token_cache["token"] and time.time() < spotify_token_cache["expires_at"]:
        return spotify_token_cache["token"]
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("⚠️ Spotify credentials missing")
        return None
    
    try:
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
            
            spotify_token_cache["token"] = token
            spotify_token_cache["expires_at"] = time.time() + expires_in - 300
            
            print("✅ Spotify token refreshed")
            return token
            
    except Exception as e:
        print(f"❌ Spotify token error: {e}")
    return None

def search_spotify(artist: str, title: str) -> Dict:
    """Comprehensive Spotify search with multiple strategies"""
    token = get_spotify_token()
    if not token:
        print("⚠️ Spotify unavailable")
        return {"found": False}
    
    try:
        # Try multiple query strategies
        queries = [
            f'track:"{title}" artist:"{artist}"',
            f'"{title}" "{artist}"',
            f'{title} {artist}',
            f'{artist} {title}'
        ]
        
        best_match = None
        
        for query in queries:
            try:
                response = requests.get(
                    "https://api.spotify.com/v1/search",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "q": query,
                        "type": "track",
                        "limit": 5,
                        "market": "IN"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    tracks = data.get("tracks", {}).get("items", [])
                    
                    if tracks:
                        best_match = tracks[0]
                        print(f"✅ Spotify found: {best_match['name']} by {best_match['artists'][0]['name']}")
                        break
                        
            except Exception as e:
                print(f"Query attempt failed: {e}")
                continue
        
        if not best_match:
            print("⚠️ Spotify: No results found")
            return {"found": False}
        
        track_id = best_match["id"]
        
        # Get audio features
        audio_features = {}
        try:
            features_response = requests.get(
                f"https://api.spotify.com/v1/audio-features/{track_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5
            )
            if features_response.status_code == 200:
                audio_features = features_response.json()
        except Exception as e:
            print(f"Audio features error: {e}")
        
        # Get artist details
        artist_data = {}
        try:
            artist_id = best_match["artists"][0]["id"]
            artist_response = requests.get(
                f"https://api.spotify.com/v1/artists/{artist_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5
            )
            if artist_response.status_code == 200:
                artist_data = artist_response.json()
        except Exception as e:
            print(f"Artist data error: {e}")
        
        return {
            "found": True,
            "track": {
                "name": best_match["name"],
                "popularity": best_match.get("popularity", 0),
                "explicit": best_match.get("explicit", False),
                "duration_ms": best_match.get("duration_ms", 0)
            },
            "audio_features": {
                "valence": audio_features.get("valence", 0.5),
                "energy": audio_features.get("energy", 0.5),
                "danceability": audio_features.get("danceability", 0.5),
                "acousticness": audio_features.get("acousticness", 0.5),
                "instrumentalness": audio_features.get("instrumentalness", 0),
                "speechiness": audio_features.get("speechiness", 0),
                "tempo": audio_features.get("tempo", 120),
                "key": audio_features.get("key", -1),
                "mode": audio_features.get("mode", 1),
                "loudness": audio_features.get("loudness", -5)
            },
            "artist": {
                "name": best_match["artists"][0]["name"],
                "genres": artist_data.get("genres", []),
                "popularity": artist_data.get("popularity", 0),
                "followers": artist_data.get("followers", {}).get("total", 0)
            },
            "album": {
                "name": best_match["album"]["name"],
                "release_date": best_match["album"].get("release_date", ""),
                "type": best_match["album"].get("album_type", "")
            }
        }
        
    except Exception as e:
        print(f"Spotify search error: {e}")
        return {"found": False}

def search_lastfm(artist: str, title: str) -> Dict:
    """Last.fm API for additional metadata"""
    api_key = os.getenv("LASTFM_API_KEY", "")
    if not api_key:
        print("⚠️ Last.fm API key missing")
        return {"found": False}
    
    try:
        response = requests.get(
            "http://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getInfo",
                "api_key": api_key,
                "artist": artist,
                "track": title,
                "format": "json"
            },
            timeout=8
        )
        
        if response.status_code == 200:
            data = response.json()
            if "track" in data:
                track = data["track"]
                tags = [t["name"] for t in track.get("toptags", {}).get("tag", [])[:10]]
                
                print(f"✅ Last.fm found: {len(tags)} tags")
                
                return {
                    "found": True,
                    "tags": tags,
                    "listeners": int(track.get("listeners", 0)),
                    "playcount": int(track.get("playcount", 0)),
                    "album": track.get("album", {}).get("title", ""),
                    "summary": track.get("wiki", {}).get("summary", "")[:500]
                }
        
        print("⚠️ Last.fm: Track not found")
        return {"found": False}
        
    except Exception as e:
        print(f"Last.fm error: {e}")
        return {"found": False}

def search_musicbrainz(artist: str, title: str) -> Dict:
    """MusicBrainz for record labels and release info"""
    try:
        clean_artist = clean_text(artist)
        clean_title = clean_text(title)
        
        query = f'recording:"{clean_title}" AND artist:"{clean_artist}"'
        
        response = requests.get(
            "https://musicbrainz.org/ws/2/recording/",
            params={
                "query": query,
                "fmt": "json",
                "limit": 3
            },
            headers={"User-Agent": "AudioVibe/11.0 (support@audiovibe.app)"},
            timeout=8
        )
        
        if response.status_code == 200:
            data = response.json()
            recordings = data.get("recordings", [])
            
            if recordings:
                rec = recordings[0]
                
                labels = []
                for rel in rec.get("releases", [])[:5]:
                    if "label-info" in rel:
                        for label_info in rel["label-info"]:
                            if label_info.get("label"):
                                labels.append(label_info["label"].get("name", ""))
                
                tags = [t["name"] for t in rec.get("tags", [])[:10]]
                
                print(f"✅ MusicBrainz: {len(labels)} labels, {len(tags)} tags")
                
                return {
                    "found": True,
                    "labels": list(set(labels))[:5],
                    "tags": tags,
                    "recording_id": rec.get("id", "")
                }
        
        print("⚠️ MusicBrainz: No results")
        return {"found": False}
        
    except Exception as e:
        print(f"MusicBrainz error: {e}")
        return {"found": False}

def analyze_with_groq(artist: str, title: str, album: str, 
                      spotify_data: Dict, lastfm_data: Dict, mb_data: Dict) -> Dict:
    """
    GROQ AI analyzes ALL data sources to provide accurate metadata.
    This is the SINGLE source of truth for classification.
    """
    
    if not groq_client:
        print("❌ GROQ client unavailable")
        return {
            "mood": "Neutral",
            "language": "Unknown",
            "industry": "International",
            "genre": "Pop"
        }
    
    # Build comprehensive context
    context_parts = [
        f"=== SONG INFORMATION ===",
        f"Artist: {artist}",
        f"Title: {title}",
        f"Album: {album if album else 'Unknown'}",
        ""
    ]
    
    # Add Spotify data
    if spotify_data.get("found"):
        context_parts.append("=== SPOTIFY DATA ===")
        context_parts.append(f"Track Name: {spotify_data['track']['name']}")
        context_parts.append(f"Popularity: {spotify_data['track']['popularity']}/100")
        
        genres = spotify_data.get("artist", {}).get("genres", [])
        if genres:
            context_parts.append(f"Artist Genres: {', '.join(genres[:10])}")
        
        features = spotify_data.get("audio_features", {})
        context_parts.append(f"Audio Features:")
        context_parts.append(f"  - Valence (happiness): {features.get('valence', 0):.2f} (0=sad, 1=happy)")
        context_parts.append(f"  - Energy: {features.get('energy', 0):.2f} (0=calm, 1=intense)")
        context_parts.append(f"  - Danceability: {features.get('danceability', 0):.2f}")
        context_parts.append(f"  - Acousticness: {features.get('acousticness', 0):.2f}")
        context_parts.append(f"  - Tempo: {features.get('tempo', 0):.0f} BPM")
        context_parts.append(f"  - Speechiness: {features.get('speechiness', 0):.2f}")
        context_parts.append(f"  - Loudness: {features.get('loudness', 0):.1f} dB")
        context_parts.append("")
    
    # Add Last.fm data
    if lastfm_data.get("found"):
        context_parts.append("=== LAST.FM DATA ===")
        tags = lastfm_data.get("tags", [])
        if tags:
            context_parts.append(f"Tags: {', '.join(tags[:15])}")
        context_parts.append(f"Listeners: {lastfm_data.get('listeners', 0):,}")
        context_parts.append(f"Playcount: {lastfm_data.get('playcount', 0):,}")
        context_parts.append("")
    
    # Add MusicBrainz data
    if mb_data.get("found"):
        context_parts.append("=== MUSICBRAINZ DATA ===")
        labels = mb_data.get("labels", [])
        if labels:
            context_parts.append(f"Record Labels: {', '.join(labels)}")
        mb_tags = mb_data.get("tags", [])
        if mb_tags:
            context_parts.append(f"MB Tags: {', '.join(mb_tags)}")
        context_parts.append("")
    
    context = "\n".join(context_parts)
    
    # Enhanced prompt with detailed instructions
    prompt = f"""{context}

=== YOUR TASK ===
Analyze ALL the data above and classify this song into these 4 categories:

1. MOOD (Choose ONE that best fits):
   - Aggressive: Dark, intense, violent themes, mafia/gang content, heavy bass, high energy + low happiness
   - Energetic: High energy, danceable, party vibes, club music, fast tempo
   - Romantic: Love songs, emotional, moderate energy, acoustic elements
   - Melancholic: Sad, breakup, introspective, low valence/happiness
   - Chill: Relaxed, lo-fi, calm, low energy
   - Uplifting: Happy, motivational, high valence, inspiring
   
   **Decision guide for MOOD:**
   - If valence < 0.4 AND energy > 0.7 → Aggressive
   - If valence < 0.35 → Melancholic
   - If energy > 0.75 AND danceability > 0.7 → Energetic
   - If valence > 0.7 → Uplifting
   - If energy < 0.5 → Chill
   - If acousticness > 0.4 AND 0.4 < valence < 0.7 → Romantic

2. LANGUAGE (The actual lyrics language, NOT the title language):
   - Hindi: Bollywood songs, Hindi film music, mainstream Indian pop
   - Punjabi: Punjabi artists (Honey Singh, Sidhu, AP Dhillon, Badshah), even if title is English
   - Telugu: Tollywood, South Indian Telugu cinema
   - Tamil: Kollywood, Tamil cinema
   - Malayalam: Malayalam cinema
   - English: Western/International artists only
   - Unknown: Only if absolutely no information available
   
   **Decision guide for LANGUAGE:**
   - If artist genres contain "punjabi", "desi hip hop" → Punjabi
   - If artist genres contain "bollywood", "filmi" → Hindi
   - If artist genres contain "tollywood", "telugu" → Telugu
   - If artist genres contain "tamil", "kollywood" → Tamil
   - If artist is Western/International → English
   - Examples: "Yo Yo Honey Singh" → Punjabi (even if song title is English)

3. INDUSTRY (Choose ONE):
   - Bollywood: Hindi film industry, mainstream Indian cinema (T-Series, Zee Music, YRF, Eros)
   - Punjabi: Punjabi music industry (independent or Punjabi cinema)
   - Tollywood: Telugu film industry
   - Kollywood: Tamil film industry
   - Mollywood: Malayalam film industry
   - Indie: Independent artists, non-film music
   - International: Western/English music
   
   **Decision guide for INDUSTRY:**
   - If Spotify genres include "punjabi" → Punjabi
   - If Spotify genres include "bollywood" → Bollywood
   - If labels include "T-Series", "Zee Music", "YRF" → Bollywood
   - If labels include "Speed Records", "White Hill" → Punjabi
   - If Western artist with English language → International

4. GENRE (Choose ONE):
   - Hip-Hop: Rap, trap, urban music, hip hop beats
   - Party: Club bangers, EDM, high energy dance
   - Pop: Mainstream, catchy, radio-friendly
   - Folk: Traditional, acoustic, cultural
   - Devotional: Religious, spiritual, bhajans
   - LoFi: Chill beats, study music
   - EDM: Electronic dance music, house, techno
   - Rock: Guitar-driven, alternative, metal
   - Jazz: Jazz, blues, swing
   - Classical: Traditional instruments, ragas
   
   **Decision guide for GENRE:**
   - If Spotify genres contain "hip hop", "rap", "trap" → Hip-Hop
   - If Spotify genres contain "edm", "house", "electronic" → EDM
   - If danceability > 0.8 AND energy > 0.75 → Party
   - If acousticness > 0.6 → Folk
   - Default to → Pop

=== EXAMPLES ===
Example 1: "Mafia" by Yo Yo Honey Singh
- Mood: Aggressive (dark theme, mafia content)
- Language: Punjabi (Honey Singh is Punjabi artist)
- Industry: Punjabi (not Bollywood film)
- Genre: Hip-Hop (rap/hip hop style)

Example 2: "Kesariya" by Arijit Singh
- Mood: Romantic (love song from movie)
- Language: Hindi (Bollywood song)
- Industry: Bollywood (from film Brahmastra)
- Genre: Pop (mainstream Bollywood pop)

Example 3: "Brown Munde" by AP Dhillon
- Mood: Energetic (party vibe)
- Language: Punjabi
- Industry: Punjabi (independent Punjabi)
- Genre: Hip-Hop

=== OUTPUT FORMAT ===
Return ONLY valid JSON, no markdown, no explanation:
{{
  "mood": "Aggressive",
  "language": "Punjabi",
  "industry": "Punjabi",
  "genre": "Hip-Hop",
  "reasoning": "Brief 1-sentence explanation of each choice"
}}

**CRITICAL RULES:**
- NEVER return "Neutral" or "Unknown" if you have ANY data
- Trust Spotify audio features (valence, energy) for mood
- Trust Spotify genres for language/industry
- Punjabi artists → Punjabi language (even if title is English)
- Use examples as reference for similar songs
"""

    try:
        print("🤖 Sending comprehensive data to GROQ AI...")
        
        chat = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert music analyst. Analyze data carefully and return ONLY valid JSON. Be decisive - never return Unknown or Neutral if you have data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        content = chat.choices[0].message.content.strip()
        
        # Aggressive cleanup
        content = content.replace("```json", "").replace("```", "").strip()
        if content.startswith("json"):
            content = content[4:].strip()
        
        result = json.loads(content)
        
        print(f"✅ AI Analysis Complete:")
        print(f"   Mood: {result.get('mood')}")
        print(f"   Language: {result.get('language')}")
        print(f"   Industry: {result.get('industry')}")
        print(f"   Genre: {result.get('genre')}")
        
        return {
            "mood": result.get("mood", "Neutral").title(),
            "language": result.get("language", "Unknown").title(),
            "industry": result.get("industry", "International").title(),
            "genre": result.get("genre", "Pop").title(),
            "reasoning": result.get("reasoning", "")
        }
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw response: {content}")
        return {
            "mood": "Neutral",
            "language": "Unknown",
            "industry": "International",
            "genre": "Pop",
            "reasoning": "Parse error"
        }
    except Exception as e:
        print(f"❌ GROQ AI Error: {e}")
        return {
            "mood": "Neutral",
            "language": "Unknown",
            "industry": "International",
            "genre": "Pop",
            "reasoning": str(e)
        }

def ensure_ready():
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Maintenance: {startup_error}")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    return {
        "status": "Active" if not startup_error else "Maintenance",
        "version": "11.0.0-PureAI",
        "detection": "100% AI-Powered (Spotify + Last.fm + MusicBrainz + GROQ)",
        "sources": {
            "spotify": "✅" if os.getenv("SPOTIFY_CLIENT_ID") else "⚠️ Missing",
            "lastfm": "✅" if os.getenv("LASTFM_API_KEY") else "⚠️ Missing",
            "musicbrainz": "✅ Always Available",
            "groq": "✅" if groq_client else "⚠️ Missing"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if not startup_error else "unhealthy",
        "spotify_token": "✅" if spotify_token_cache.get("token") else "⚠️",
        "groq": "✅" if groq_client else "❌"
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
    
    # Check cache
    cache_key = f"{clean_artist.lower()}:{clean_title.lower()}:{clean_album.lower()}"
    if cache_key in metadata_cache:
        cached = metadata_cache[cache_key]
        if time.time() - cached.get("cached_at", 0) < CACHE_TTL:
            print(f"💾 Returning cached result for: {clean_title}")
            return cached["data"]
    
    print(f"\n🎵 ==========================================")
    print(f"🎵 Analyzing: '{clean_title}' by {clean_artist}")
    print(f"🎵 ==========================================")
    
    # STEP 1: Gather data from all sources
    print("\n📡 Fetching data from sources...")
    spotify_data = search_spotify(clean_artist, clean_title)
    lastfm_data = search_lastfm(clean_artist, clean_title)
    mb_data = search_musicbrainz(clean_artist, clean_title)
    
    # STEP 2: Let AI analyze everything
    print("\n🤖 Sending to GROQ AI for comprehensive analysis...")
    ai_result = analyze_with_groq(
        clean_artist, clean_title, clean_album,
        spotify_data, lastfm_data, mb_data
    )
    
    # Calculate confidence score
    confidence_score = 0
    sources_used = []
    
    if spotify_data.get("found"):
        confidence_score += 50
        sources_used.append("spotify")
    if lastfm_data.get("found"):
        confidence_score += 25
        sources_used.append("lastfm")
    if mb_data.get("found"):
        confidence_score += 10
        sources_used.append("musicbrainz")
    if groq_client:
        confidence_score += 15
        sources_used.append("groq_ai")
    
    print(f"\n✅ Final Classification:")
    print(f"   📊 {ai_result['mood']};{ai_result['language']};{ai_result['industry']};{ai_result['genre']}")
    print(f"   🎯 Confidence: {confidence_score}%")
    print(f"   📚 Sources: {', '.join(sources_used)}")
    
    # Build result
    result = {
        "formatted": f"{ai_result['mood']};{ai_result['language']};{ai_result['industry']};{ai_result['genre']}",
        "mood": ai_result["mood"],
        "language": ai_result["language"],
        "industry": ai_result["industry"],
        "genre": ai_result["genre"],
        "confidence_score": confidence_score,
        "sources_used": sources_used,
        "ai_reasoning": ai_result.get("reasoning", ""),
        "raw_data": {
            "spotify": {
                "found": spotify_data.get("found", False),
                "genres": spotify_data.get("artist", {}).get("genres", [])[:5] if spotify_data.get("found") else [],
                "popularity": spotify_data.get("track", {}).get("popularity", 0) if spotify_data.get("found") else 0,
                "audio_features": spotify_data.get("audio_features", {}) if spotify_data.get("found") else {}
            },
            "lastfm": {
                "found": lastfm_data.get("found", False),
                "tags": lastfm_data.get("tags", [])[:5] if lastfm_data.get("found") else [],
                "listeners": lastfm_data.get("listeners", 0) if lastfm_data.get("found") else 0
            },
            "musicbrainz": {
                "found": mb_data.get("found", False),
                "labels": mb_data.get("labels", [])[:3] if mb_data.get("found") else [],
                "tags": mb_data.get("tags", [])[:5] if mb_data.get("found") else []
            }
        }
    }
    
    # Cache the result
    metadata_cache[cache_key] = {
        "data": result,
        "cached_at": time.time()
    }
    
    print(f"🎵 ==========================================\n")
    
    return result

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
        print(f"❌ Database error: {e}")
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
        print(f"❌ Upload error: {e}")
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
        # Uncomment when ready to store play statistics
        # supabase.table("play_history").insert(stat.dict()).execute()
        return {"status": "recorded"}
    except Exception as e:
        print(f"❌ Play record error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
