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
# ENVIRONMENT SETUP
# ============================================================
load_dotenv()
app = FastAPI(title="AudioVibe API", version="13.0.0-PureSpotifyAI")

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
    print("✅ Pure Spotify + AI Detection Engine Online")
except Exception as e:
    startup_error = str(e)
    print(f"❌ Startup Warning: {e}")

# ============================================================
# PYDANTIC MODELS
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
# SECURITY MIDDLEWARE
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
    """Remove special characters and clean text"""
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    text = re.sub(r'feat\.|ft\.|featuring', '', text, flags=re.IGNORECASE)
    return text.strip()

def get_spotify_token() -> Optional[str]:
    """Get or refresh Spotify access token"""
    global spotify_token_cache
    if spotify_token_cache["token"] and time.time() < spotify_token_cache["expires_at"]:
        return spotify_token_cache["token"]
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    
    try:
        auth_str = f"{client_id}:{client_secret}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()
        
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {b64_auth}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            spotify_token_cache["token"] = data["access_token"]
            spotify_token_cache["expires_at"] = time.time() + data["expires_in"] - 300
            print("✅ Spotify token refreshed")
            return data["access_token"]
    except Exception as e:
        print(f"❌ Spotify auth error: {e}")
    return None

def search_spotify(artist: str, title: str) -> Dict:
    """Search Spotify for track data with multiple fallback queries"""
    token = get_spotify_token()
    if not token:
        print("⚠️ Spotify unavailable")
        return {"found": False}
    
    try:
        queries = [
            f'track:"{title}" artist:"{artist}"',
            f'{title} {artist}',
            f'{artist} {title}'
        ]
        
        for query in queries:
            try:
                response = requests.get(
                    "https://api.spotify.com/v1/search",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"q": query, "type": "track", "limit": 5, "market": "IN"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    tracks = response.json().get("tracks", {}).get("items", [])
                    if tracks:
                        track = tracks[0]
                        track_id = track["id"]
                        
                        print(f"✅ Spotify found: {track['name']} by {track['artists'][0]['name']}")
                        
                        # Get audio features
                        features = {}
                        try:
                            f_res = requests.get(
                                f"https://api.spotify.com/v1/audio-features/{track_id}",
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=5
                            )
                            if f_res.status_code == 200:
                                features = f_res.json()
                        except Exception as e:
                            print(f"⚠️ Audio features error: {e}")
                        
                        # Get artist details
                        artist_data = {}
                        try:
                            artist_id = track["artists"][0]["id"]
                            a_res = requests.get(
                                f"https://api.spotify.com/v1/artists/{artist_id}",
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=5
                            )
                            if a_res.status_code == 200:
                                artist_data = a_res.json()
                        except Exception as e:
                            print(f"⚠️ Artist data error: {e}")
                        
                        genres = artist_data.get("genres", [])
                        print(f"   Genres: {', '.join(genres[:5]) if genres else 'None'}")
                        
                        return {
                            "found": True,
                            "track": {
                                "name": track["name"],
                                "popularity": track.get("popularity", 0)
                            },
                            "audio_features": {
                                "valence": features.get("valence", 0.5),
                                "energy": features.get("energy", 0.5),
                                "danceability": features.get("danceability", 0.5),
                                "acousticness": features.get("acousticness", 0.5),
                                "instrumentalness": features.get("instrumentalness", 0),
                                "tempo": features.get("tempo", 120),
                                "speechiness": features.get("speechiness", 0),
                                "loudness": features.get("loudness", -5)
                            },
                            "artist": {
                                "name": track["artists"][0]["name"],
                                "genres": genres,
                                "popularity": artist_data.get("popularity", 0)
                            },
                            "album": {
                                "name": track["album"]["name"]
                            }
                        }
            except Exception as e:
                print(f"⚠️ Query failed: {e}")
                continue
        
        print("⚠️ Spotify: No results found")
        return {"found": False}
        
    except Exception as e:
        print(f"❌ Spotify search error: {e}")
        return {"found": False}

def analyze_with_ai(artist: str, title: str, album: str, spotify_data: Dict) -> Dict:
    """
    Pure AI analysis using ONLY Spotify data.
    AI analyzes genres + audio features to make all classification decisions.
    """
    
    if not groq_client:
        print("❌ GROQ unavailable - returning fallback")
        return {
            "mood": "Neutral",
            "language": "Unknown",
            "industry": "International",
            "genre": "Pop"
        }
    
    if not spotify_data.get("found"):
        print("⚠️ No Spotify data - AI cannot classify")
        return {
            "mood": "Neutral",
            "language": "Unknown",
            "industry": "International",
            "genre": "Pop"
        }
    
    features = spotify_data.get("audio_features", {})
    genres = spotify_data.get("artist", {}).get("genres", [])
    track_name = spotify_data.get("track", {}).get("name", title)
    artist_name = spotify_data.get("artist", {}).get("name", artist)
    album_name = spotify_data.get("album", {}).get("name", album or "Unknown")
    
    context = f"""=== SONG INFORMATION ===
Artist: {artist_name}
Title: {track_name}
Album: {album_name}

=== SPOTIFY GENRES ===
{', '.join(genres[:10]) if genres else 'No genre data available'}

=== AUDIO FEATURES (Spotify Analysis) ===
Valence (happiness): {features.get('valence', 0.5):.2f} (0=sad, 1=happy)
Energy: {features.get('energy', 0.5):.2f} (0=calm, 1=intense)
Danceability: {features.get('danceability', 0.5):.2f} (0=not danceable, 1=very danceable)
Acousticness: {features.get('acousticness', 0.5):.2f} (0=electronic, 1=acoustic)
Speechiness: {features.get('speechiness', 0):.2f} (0=music, 1=speech/rap)
Instrumentalness: {features.get('instrumentalness', 0):.2f} (0=vocals, 1=instrumental)
Tempo: {features.get('tempo', 120):.0f} BPM
Loudness: {features.get('loudness', -5):.1f} dB

=== YOUR TASK ===
Analyze the Spotify genres and audio features above to classify this song.

1. MOOD (choose ONE):
   - Aggressive: High energy + Low valence + Fast tempo (mafia, gang, dark themes)
   - Energetic: High energy + High danceability (party, club, upbeat)
   - Romantic: Moderate energy + Mid valence + Acoustic (love songs)
   - Melancholic: Low valence (sad, emotional, breakup)
   - Chill: Low energy + Mid valence (relaxed, lo-fi, calm)
   - Uplifting: High valence + Moderate energy (happy, motivational)

2. LANGUAGE (lyrics language - analyze Spotify genres):
   - If genres contain "punjabi", "bhangra", "desi hip hop" → Punjabi
   - If genres contain "bollywood", "filmi", "hindi" → Hindi
   - If genres contain "tollywood", "telugu" → Telugu
   - If genres contain "kollywood", "tamil" → Tamil
   - If genres contain "mollywood", "malayalam" → Malayalam
   - If genres are Western (pop, rock, hip hop, etc.) → English
   - Default → Unknown (ONLY if no genre data)

3. INDUSTRY (music industry):
   - Punjabi: Punjabi music scene (punjabi, bhangra genres)
   - Bollywood: Hindi film industry (bollywood, filmi genres)
   - Tollywood: Telugu film industry (tollywood, telugu genres)
   - Kollywood: Tamil film industry (kollywood, tamil genres)
   - Mollywood: Malayalam film industry (mollywood genres)
   - International: Western music (pop, rock, hip hop, etc.)
   - Indie: Independent artists (indie, alternative genres)

4. GENRE (music style - analyze BOTH Spotify genres AND audio features):
   - Hip-Hop: Spotify genres contain "hip hop", "rap", "trap", "desi hip hop" OR speechiness > 0.33
   - Party: Spotify genres contain "party", "club" OR (danceability > 0.8 AND energy > 0.75)
   - Pop: Spotify genres contain "pop" (bollywood pop, dance pop, etc.)
   - Folk: Spotify genres contain "folk", "traditional", "bhangra" OR acousticness > 0.7
   - EDM: Spotify genres contain "edm", "house", "techno", "electronic", "trance"
   - Rock: Spotify genres contain "rock", "metal", "alternative", "indie rock"
   - Jazz: Spotify genres contain "jazz", "blues", "swing"
   - Classical: Spotify genres contain "classical", "carnatic", "hindustani", "symphony"
   - LoFi: Spotify genres contain "lo-fi", "lofi" OR (energy < 0.4 AND instrumentalness > 0.5)
   - Devotional: Spotify genres contain "devotional", "bhajan", "spiritual", "worship"
   
   IMPORTANT: Use BOTH genre keywords AND audio features to make the best decision.
   If Spotify genres clearly indicate a genre, trust that first.
   If genres are vague or missing, use audio features as backup.

=== EXAMPLES ===
Example 1: Genres: ["punjabi hip hop", "desi hip hop"]
Audio: valence=0.35, energy=0.82, danceability=0.72, speechiness=0.25, tempo=140
Analysis: Genres clearly indicate "hip hop" → Hip-Hop
→ {{"mood": "Aggressive", "language": "Punjabi", "industry": "Punjabi", "genre": "Hip-Hop"}}

Example 2: Genres: ["bollywood", "filmi"]
Audio: valence=0.52, energy=0.45, acousticness=0.73, danceability=0.45
Analysis: Genres indicate "bollywood" (which is pop-style), audio is romantic
→ {{"mood": "Romantic", "language": "Hindi", "industry": "Bollywood", "genre": "Pop"}}

Example 3: Genres: ["pop", "dance pop"]
Audio: valence=0.75, energy=0.73, danceability=0.85, acousticness=0.15
Analysis: Genres say "pop", high danceability confirms party/dance style
→ {{"mood": "Energetic", "language": "English", "industry": "International", "genre": "Pop"}}

Example 4: Genres: ["electronic", "house"]
Audio: valence=0.65, energy=0.88, danceability=0.90, acousticness=0.05
Analysis: Genres clearly indicate "electronic" + "house" → EDM
→ {{"mood": "Energetic", "language": "English", "industry": "International", "genre": "EDM"}}

Example 5: Genres: ["indie folk", "singer-songwriter"]
Audio: valence=0.55, energy=0.35, acousticness=0.85, danceability=0.30
Analysis: Genres say "folk", high acousticness confirms → Folk
→ {{"mood": "Chill", "language": "English", "industry": "Indie", "genre": "Folk"}}

Example 6: Genres: [] (No genre data)
Audio: valence=0.40, energy=0.30, acousticness=0.20, speechiness=0.45, danceability=0.55
Analysis: No genres, but high speechiness indicates rap/hip-hop → Hip-Hop
→ {{"mood": "Chill", "language": "Unknown", "industry": "International", "genre": "Hip-Hop"}}

=== OUTPUT ===
Return ONLY valid JSON (no markdown, no explanation):
{{"mood": "...", "language": "...", "industry": "...", "genre": "..."}}

Be decisive. Trust the Spotify genres - they are expert-curated. Use audio features to confirm or override when needed."""

    try:
        print("🤖 Sending to AI for analysis...")
        
        chat = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert music classifier. Analyze Spotify data (genres + audio features) and return ONLY valid JSON. Be decisive - make the best choice using all available data."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        content = chat.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        if content.startswith("json"):
            content = content[4:].strip()
        
        result = json.loads(content)
        
        final = {
            "mood": result.get("mood", "Neutral").title(),
            "language": result.get("language", "Unknown").title(),
            "industry": result.get("industry", "International").title(),
            "genre": result.get("genre", "Pop").title()
        }
        
        print(f"✅ AI Classification:")
        print(f"   Mood: {final['mood']}")
        print(f"   Language: {final['language']}")
        print(f"   Industry: {final['industry']}")
        print(f"   Genre: {final['genre']}")
        
        return final
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"   Raw response: {content}")
        return {
            "mood": "Neutral",
            "language": "Unknown",
            "industry": "International",
            "genre": "Pop"
        }
    except Exception as e:
        print(f"❌ AI Error: {e}")
        return {
            "mood": "Neutral",
            "language": "Unknown",
            "industry": "International",
            "genre": "Pop"
        }

def ensure_ready():
    """Check if backend services are ready"""
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Maintenance: {startup_error}")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def read_root():
    """Root endpoint - API info"""
    return {
        "status": "Active",
        "version": "13.0.0-PureSpotifyAI",
        "method": "Pure AI analysis of Spotify data (genres + audio features)",
        "sources": {
            "spotify": "✅" if os.getenv("SPOTIFY_CLIENT_ID") else "⚠️ Missing",
            "groq": "✅" if groq_client else "⚠️ Missing"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "spotify": "✅" if spotify_token_cache.get("token") else "⚠️",
        "groq": "✅" if groq_client else "⚠️"
    }

@app.post("/enrich-metadata")
async def enrich_metadata(
    song: SongRequest,
    x_app_timestamp: str = Header(...),
    x_app_integrity: str = Header(...),
    x_device_id: str = Header(...),
    x_app_version: str = Header(...)
):
    """Main endpoint - Enrich song metadata using Spotify + AI"""
    ensure_ready()
    
    clean_title = clean_text(song.title)
    clean_artist = clean_text(song.artist)
    clean_album = clean_text(song.album) if song.album else ""
    
    # Check cache
    cache_key = f"{clean_artist.lower()}:{clean_title.lower()}:{clean_album.lower()}"
    if cache_key in metadata_cache:
        cached = metadata_cache[cache_key]
        if time.time() - cached.get("cached_at", 0) < CACHE_TTL:
            print(f"💾 Cache hit: {clean_title}")
            return cached["data"]
    
    print(f"\n🎵 ==========================================")
    print(f"🎵 Analyzing: '{clean_title}' by {clean_artist}")
    print(f"🎵 ==========================================")
    
    # Get Spotify data
    spotify_data = search_spotify(clean_artist, clean_title)
    
    # Let AI analyze
    ai_result = analyze_with_ai(clean_artist, clean_title, clean_album, spotify_data)
    
    # Build result
    confidence_score = 90 if spotify_data.get("found") else 0
    sources = ["spotify", "groq_ai"] if spotify_data.get("found") else []
    
    result = {
        "formatted": f"{ai_result['mood']};{ai_result['language']};{ai_result['industry']};{ai_result['genre']}",
        "mood": ai_result["mood"],
        "language": ai_result["language"],
        "industry": ai_result["industry"],
        "genre": ai_result["genre"],
        "confidence_score": confidence_score,
        "sources_used": sources,
        "raw_data": {
            "spotify": {
                "found": spotify_data.get("found", False),
                "genres": spotify_data.get("artist", {}).get("genres", [])[:5] if spotify_data.get("found") else [],
                "audio_features": spotify_data.get("audio_features", {}) if spotify_data.get("found") else {}
            }
        }
    }
    
    print(f"\n✅ Final Result: {result['formatted']}")
    print(f"   Confidence: {confidence_score}%")
    print(f"🎵 ==========================================\n")
    
    # Cache result
    metadata_cache[cache_key] = {"data": result, "cached_at": time.time()}
    
    return result

@app.get("/tracks")
async def get_tracks(
    x_app_timestamp: str = Header(...),
    x_app_integrity: str = Header(...),
    x_device_id: str = Header(...),
    x_app_version: str = Header(...)
):
    """Get all tracks from database"""
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
    """Add new track to database"""
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
    """Record play statistics"""
    ensure_ready()
    return {"status": "recorded"}

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
