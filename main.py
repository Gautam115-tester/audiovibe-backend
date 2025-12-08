from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from collections import defaultdict
import hashlib
import os
import json
import time
import re
import requests # Required for Last.fm
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq
from duckduckgo_search import DDGS

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES & SETUP
# ============================================================
load_dotenv()

app = FastAPI(title="AudioVibe Secure API", version="7.2.0-LastFM")

# ============================================================
# 🔐 CONFIGURATION
# ============================================================

APP_INTEGRITY_SECRET = os.getenv("APP_INTEGRITY_SECRET", "DONOTTOUCHAPI")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY") # 🔴 NEW: Add this to .env
ALLOWED_APP_VERSIONS = ["1.0", "1.0.0", "1.0.1", "1.1.0"]
RATE_LIMIT_WINDOW = 60  
RATE_LIMIT_MAX_REQUESTS = 100  
rate_limit_storage = defaultdict(list)

# ============================================================
# 2. INITIALIZE CLIENTS
# ============================================================

supabase: Optional[Client] = None
groq_client: Optional[Groq] = None
GROQ_MODELS = {"primary": "llama-3.3-70b-versatile"}

metadata_cache = {}

try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase Credentials Missing")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY Missing")
    if not LASTFM_API_KEY:
        print("⚠️ WARNING: LASTFM_API_KEY not found. Metadata accuracy will be lower.")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ System Online: All Clients Initialized")

except Exception as e:
    print(f"❌ Startup Error: {e}")

# ============================================================
# 3. MODELS
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
# 4. HELPER: LAST.FM + WEB SEARCH
# ============================================================

def get_lastfm_data(artist: str, track: str) -> dict:
    """
    Queries Last.fm specifically matching Artist AND Track name.
    This solves the '1 million songs' duplicate issue.
    """
    if not LASTFM_API_KEY:
        return {"found": False, "tags": [], "summary": ""}

    try:
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            "method": "track.getInfo",
            "api_key": LASTFM_API_KEY,
            "artist": artist,
            "track": track,
            "autocorrect": 1, # Fixes typos like 'Linkin Park' vs 'LinkinPark'
            "format": "json"
        }
        
        print(f"🎵 Querying Last.fm for: {artist} - {track}")
        response = requests.get(url, params=params, timeout=3)
        data = response.json()

        if "track" in data:
            t_data = data["track"]
            
            # Extract Tags (Genres)
            tags = [t["name"] for t in t_data.get("toptags", {}).get("tag", [])]
            
            # Extract Wiki Summary (Context)
            summary = t_data.get("wiki", {}).get("summary", "")
            
            # Extract Album Name (for verification)
            album_match = t_data.get("album", {}).get("title", "")

            return {
                "found": True, 
                "tags": tags, 
                "summary": summary, 
                "album_match": album_match
            }
            
    except Exception as e:
        print(f"⚠️ Last.fm Error: {e}")
    
    return {"found": False, "tags": [], "summary": ""}

def perform_web_search(artist: str, title: str) -> str:
    """Fallback if Last.fm fails"""
    query = f"{artist} {title} song genre mood review"
    print(f"🌍 Fallback Web Search: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            return "\n".join([f"- {r['body']}" for r in results])
    except:
        return ""

def extract_json_robust(text: str) -> dict:
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return {}

# ============================================================
# 📌 API ENDPOINTS
# ============================================================

@app.post("/enrich-metadata")
async def enrich_metadata(
    song: SongRequest,
    x_app_integrity: str = Header(...) # Shortened for brevity
):
    clean_title = song.title.strip()
    clean_artist = song.artist.strip()
    clean_album = song.album.strip() if song.album else ""

    # 1. Check Cache
    cache_key = f"{clean_artist}:{clean_title}".lower()
    if cache_key in metadata_cache:
        return metadata_cache[cache_key]

    # 2. 🎵 Query Last.fm (Primary Source)
    lfm_data = get_lastfm_data(clean_artist, clean_title)
    
    context_source = ""
    
    if lfm_data["found"]:
        # We found a specific match for Artist + Track
        context_source = (
            f"Last.fm Tags: {', '.join(lfm_data['tags'])}\n"
            f"Last.fm Summary: {lfm_data['summary']}\n"
            f"Official Album: {lfm_data.get('album_match', 'Unknown')}"
        )
        print("✅ Data sourced from Last.fm")
    else:
        # Fallback to Web Search
        search_data = perform_web_search(clean_artist, clean_title)
        context_source = f"Web Search Results: {search_data}"
        print("⚠️ Data sourced from Web Search (Last.fm miss)")

    # 3. 🤖 AI Analysis
    prompt = (
        f"Analyze this song:\n"
        f"Title: {clean_title}\nArtist: {clean_artist}\nAlbum (User Input): {clean_album}\n\n"
        f"DATA SOURCE:\n{context_source}\n\n"
        "INSTRUCTIONS:\n"
        "1. MOOD: Detect based on tags/summary (e.g. Energetic, Chill, Romantic, Sad, Aggressive).\n"
        "2. LANGUAGE: Detect lyric language.\n"
        "3. GENRE: Standardize the genre (Pop, Hip-Hop, Bollywood, Rock, EDM, Jazz).\n"
        "4. INDUSTRY: (Bollywood, Tollywood, Punjabi, International, Indie).\n"
        "NOTE: If Last.fm 'Official Album' is different from 'User Input' album, trust Last.fm context for genre/mood, but don't fail.\n"
        "\nReturn JSON: { \"mood\": \"...\", \"language\": \"...\", \"genre\": \"...\", \"industry\": \"...\" }"
    )

    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.2, 
            response_format={"type": "json_object"}
        )
        
        data = extract_json_robust(chat.choices[0].message.content)
        
        # Defaults
        result = {
            "formatted": f"{data.get('mood', 'Neutral')};{data.get('language', 'Unknown')};{data.get('industry', 'International')};{data.get('genre', 'Pop')}",
            "mood": data.get("mood", "Neutral"),
            "language": data.get("language", "Unknown"),
            "industry": data.get("industry", "International"),
            "genre": data.get("genre", "Pop"),
            "match_source": "lastfm" if lfm_data["found"] else "web_search"
        }

        metadata_cache[cache_key] = result
        return result

    except Exception as e:
        print(f"❌ AI Error: {e}")
        return {"formatted": "Energetic;English;International;Pop"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
