from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import os
import json
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq
import re
from functools import lru_cache

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="AudioVibe Backend API", version="8.0.0 (Professional-Grade-Industry-Detection)")

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Global State for Clients
supabase: Optional[Client] = None
groq_client: Optional[Groq] = None
startup_error: Optional[str] = None

# 4. Safe Initialization
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

# 5. Config
EXPECTED_INTEGRITY_HEADER = 'clean-device-v1'
GROQ_MODELS = {"primary": "llama-3.3-70b-versatile"}
metadata_cache = {}

# 6. Models
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

# 7. Helper: Check Backend Readiness
def ensure_ready():
    if startup_error:
        raise HTTPException(status_code=503, detail=f"Backend Not Ready: {startup_error}")
    if not supabase or not groq_client:
        raise HTTPException(status_code=503, detail="Backend Not Ready: Clients not initialized")

# ============================================================================
# 8. COMPREHENSIVE INDUSTRY DATABASE (2024-2025 Updated)
# ============================================================================

class IndustryDatabase:
    """
    Professional-grade music industry classification database
    Updated with 2024-2025 market data from IFPI, Billboard, and regional sources
    """
    
    # BOLLYWOOD (India - Hindi Film Music) - 75-80% of Indian music revenue
    BOLLYWOOD = {
        "labels": [
            # Major Labels (Tier 1)
            "t-series", "tseries", "t series", "super cassettes", "sony music india",
            "zee music company", "zee music", "saregama india", "tips music", "tips industries",
            
            # Film Production Houses with Music Arms
            "yrf music", "yash raj films", "yash raj music", "dharma productions", "dharma music",
            "eros music", "eros now", "venus music", "venus worldwide", "shemaroo entertainment",
            "shemaroo", "rajshri music", "ultra bollywood", "ultra music india",
            
            # Digital-First Labels
            "gaana", "jio saavn", "hungama digital media", "times music", "speed records india",
            
            # Legacy Labels
            "hmv", "his master's voice", "music today", "magnasound", "plus music"
        ],
        "artists": [
            # Male Playback Singers
            "arijit singh", "sonu nigam", "kumar sanu", "udit narayan", "atif aslam",
            "armaan malik", "jubin nautiyal", "darshan raval", "papon", "mohit chauhan",
            "vishal dadlani", "shaan", "kailash kher", "mika singh", "sukhwinder singh",
            
            # Female Playback Singers
            "shreya ghoshal", "alka yagnik", "neha kakkar", "dhvani bhanushali", "neeti mohan",
            "sunidhi chauhan", "kanika kapoor", "tulsi kumar", "monali thakur", "jonita gandhi",
            
            # Music Composers/Directors
            "pritam", "a.r. rahman", "vishal-shekhar", "sachin-jigar", "salim-sulaiman",
            "amit trivedi", "tanishk bagchi", "badshah", "yo yo honey singh", "anu malik"
        ],
        "keywords": [
            "bollywood", "hindi film", "hindi cinema", "mumbai film", "filmi", "playback",
            "हिंदी", "बॉलीवुड", "फिल्म", "filmy", "hindi album"
        ],
        "confidence_boost": ["film", "movie", "soundtrack", "ost", "original motion picture"]
    }
    
    # TOLLYWOOD (South Indian Film Music - Telugu, Tamil, Malayalam, Kannada)
    TOLLYWOOD = {
        "labels": [
            # Telugu Labels
            "lahari music", "lahari recording company", "aditya music", "mango music", 
            "anand audio", "maa music", "sriranjani music",
            
            # Tamil Labels
            "think music", "think music india", "sony music south", "saregama south",
            "muzik247", "u1 records", "divo", "inreco",
            
            # Malayalam Labels
            "manorama music", "goodwill entertainments", "millennium audios",
            
            # Pan-South Labels
            "aditya videos", "volga videos", "sony music south india"
        ],
        "artists": [
            # Composers
            "devi sri prasad", "dsp", "anirudh ravichander", "thaman s", "s thaman",
            "yuvan shankar raja", "g.v. prakash", "ghibran", "harris jayaraj", "d. imman",
            
            # Singers
            "sid sriram", "chinmayi", "haricharan", "karthik", "vijay yesudas",
            "shreya ghoshal", "anirudh", "jonita gandhi", "armaan malik"
        ],
        "keywords": [
            "tollywood", "kollywood", "sandalwood", "mollywood",
            "telugu", "tamil", "malayalam", "kannada",
            "telugu film", "tamil cinema", "tamil film", "malayalam cinema",
            "తెలుగు", "தமிழ்", "ತೆಲುಗು"
        ],
        "confidence_boost": ["telugu movie", "tamil movie", "malayalam movie", "south indian"]
    }
    
    # PUNJABI MUSIC (India/Pakistan - Bhangra, Folk, Modern Punjabi Pop)
    PUNJABI = {
        "labels": [
            "speed records", "white hill music", "saga music", "saga hits",
            "t-series apna punjab", "47 records", "lokdhun punjabi", "brown town music",
            "jass records", "gene get Punjabi", "fresh media records"
        ],
        "artists": [
            "diljit dosanjh", "sidhu moose wala", "sidhu moosewala", "ap dhillon",
            "karan aujla", "hardy sandhu", "guru randhawa", "badshah", "yo yo honey singh",
            "amrinder gill", "ammy virk", "nimrat khaira", "jassi gill", "jassie gill",
            "babbu maan", "gurdas maan", "gippy grewal", "sharry mann"
        ],
        "keywords": [
            "punjabi", "bhangra", "desi", "pind", "jatt", "punjab",
            "ਪੰਜਾਬੀ", "pollywood"
        ],
        "confidence_boost": ["punjabi song", "punjabi music", "punjabi album"]
    }
    
    # HOLLYWOOD/INTERNATIONAL (Western Music - US, UK, Europe)
    HOLLYWOOD = {
        "labels": [
            # Universal Music Group (38% market share)
            "republic records", "interscope", "interscope records", "def jam", "island records",
            "capitol records", "capitol music group", "motown", "geffen", "virgin records",
            "universal music", "umg", "mercury records", "polydor", "decca",
            
            # Sony Music Entertainment (27% market share)
            "columbia records", "rca records", "epic records", "arista records",
            "sony music", "legacy recordings", "rca victor", "sony bmg", "syco music",
            
            # Warner Music Group (18% market share)
            "atlantic records", "warner records", "warner bros records", "elektra records",
            "parlophone", "warner music", "rhino records", "asylum records", "reprise records",
            "nonesuch", "sire records", "fueled by ramen",
            
            # Major Independent Labels
            "300 entertainment", "big machine", "big loud", "bmg rights", "domino",
            "xl recordings", "rough trade", "matador", "sub pop", "epitaph"
        ],
        "artists": [
            # Current Top Artists (2024-2025)
            "taylor swift", "drake", "bad bunny", "the weeknd", "ariana grande",
            "billie eilish", "post malone", "dua lipa", "ed sheeran", "beyonce",
            "harry styles", "olivia rodrigo", "sabrina carpenter", "sza", "travis scott",
            "morgan wallen", "zach bryan", "benson boone", "teddy swims", "hozier"
        ],
        "keywords": [
            "billboard", "grammy", "american", "british", "uk music", "us music",
            "hot 100", "pop music", "western music", "anglo", "english language"
        ],
        "confidence_boost": ["billboard hot 100", "grammy award", "american music", "uk charts"]
    }
    
    # K-POP (South Korean Pop Music)
    KPOP = {
        "labels": [
            # Big 4 Agencies
            "hybe", "hybe corporation", "big hit entertainment", "big hit music", "bighit",
            "sm entertainment", "sm town", "sm ent",
            "jyp entertainment", "jyp ent",
            "yg entertainment", "yg ent",
            
            # HYBE Subsidiaries
            "ador", "belift lab", "source music", "pledis entertainment", "koz entertainment",
            
            # Other Major Agencies
            "starship entertainment", "cube entertainment", "fnc entertainment",
            "rbw", "p nation", "the black label", "kakao entertainment"
        ],
        "artists": [
            # Groups
            "bts", "blackpink", "twice", "exo", "seventeen", "stray kids", "newjeans",
            "aespa", "txt", "tomorrow x together", "enhypen", "itzy", "ive", "le sserafim",
            "nct", "nct 127", "nct dream", "red velvet", "gidle", "(g)i-dle",
            
            # Soloists
            "psy", "iu", "jungkook", "jimin", "v", "lisa", "jennie", "rose", "jisoo"
        ],
        "keywords": [
            "k-pop", "kpop", "k pop", "korean pop", "한국", "케이팝", "hallyu",
            "idol", "comeback", "korean music"
        ],
        "confidence_boost": ["korean", "seoul", "korea", "kbs", "mnet"]
    }
    
    # INDIE/INDEPENDENT (Self-Released, Small Labels, Bedroom Pop)
    INDIE = {
        "labels": [
            "independent", "self-released", "unsigned", "indie music", "bedroom pop",
            "soundcloud", "bandcamp", "distrokid", "tunecore", "cdbaby",
            "artist first", "azadi records", "incink records", "pagal haina"
        ],
        "keywords": [
            "indie", "independent", "self released", "unsigned artist", "diy",
            "lo-fi", "lofi", "bedroom", "underground", "emerging artist"
        ],
        "confidence_boost": ["independent artist", "self-published", "unsigned"]
    }

# 9. MusicBrainz API Helper
def search_musicbrainz(artist: str, title: str, album: Optional[str] = None) -> dict:
    """Enhanced MusicBrainz search with better error handling"""
    try:
        base_url = "https://musicbrainz.org/ws/2/recording/"
        query_parts = [f'recording:"{title}"', f'artist:"{artist}"']
        if album and album.lower() not in ["unknown", "single", ""]:
            query_parts.append(f'release:"{album}"')
        
        query = " AND ".join(query_parts)
        params = {"query": query, "fmt": "json", "limit": 5}
        headers = {"User-Agent": "AudioVibe/8.0 (contact@audiovibe.app)"}
        
        response = requests.get(base_url, params=params, headers=headers, timeout=7)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("recordings"):
                recording = data["recordings"][0]
                releases = recording.get("releases", [])
                tags = [tag["name"].lower() for tag in recording.get("tags", [])]
                
                labels = []
                release_types = []
                countries = []
                
                for release in releases[:5]:
                    if "label-info" in release:
                        for li in release["label-info"]:
                            label_name = li.get("label", {}).get("name")
                            if label_name:
                                labels.append(label_name.lower())
                    
                    release_types.append(release.get("status", "").lower())
                    country = release.get("country", "")
                    if country:
                        countries.append(country)
                
                return {
                    "found": True,
                    "labels": list(set(labels))[:10],
                    "tags": tags[:10],
                    "release_types": list(set(release_types)),
                    "countries": list(set(countries)),
                    "artist_credit": recording.get("artist-credit", [{}])[0].get("name", "").lower()
                }
    except Exception as e:
        print(f"⚠️ MusicBrainz error: {e}")
    
    return {"found": False, "labels": [], "tags": [], "release_types": [], "countries": [], "artist_credit": ""}

# 10. Wikipedia Helper
def search_wikipedia(artist: str, album: str) -> dict:
    """Enhanced Wikipedia search for soundtrack/film information"""
    try:
        if not album or album.lower() in ["unknown", "single", ""]:
            return {"is_soundtrack": False, "is_film_album": False, "industry_hints": [], "categories": []}
        
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{album} {artist} soundtrack film album",
            "format": "json",
            "srlimit": 5
        }
        
        response = requests.get(search_url, params=params, timeout=7)
        if response.status_code == 200:
            data = response.json()
            results = data.get("query", {}).get("search", [])
            
            for result in results:
                snippet = result.get("snippet", "").lower()
                title = result.get("title", "").lower()
                combined = f"{snippet} {title}"
                
                # Film/Soundtrack detection
                is_soundtrack = any(word in combined for word in [
                    "soundtrack", "original motion picture", "film score", "movie soundtrack",
                    "original soundtrack", "ost", "film music"
                ])
                
                # Industry-specific hints
                industry_hints = []
                categories = []
                
                if any(word in combined for word in ["bollywood", "hindi film", "mumbai", "indian cinema"]):
                    industry_hints.append("Bollywood")
                    categories.append("indian_film")
                
                if any(word in combined for word in ["tollywood", "telugu film", "tamil film", "kollywood", "south indian"]):
                    industry_hints.append("Tollywood")
                    categories.append("south_indian_film")
                
                if any(word in combined for word in ["punjabi music", "punjabi film", "pollywood"]):
                    industry_hints.append("Punjabi")
                    categories.append("punjabi_music")
                
                if any(word in combined for word in ["hollywood", "american film", "warner bros", "universal pictures", "paramount"]):
                    industry_hints.append("Hollywood")
                    categories.append("hollywood_film")
                
                if any(word in combined for word in ["k-pop", "korean pop", "south korean", "korean music"]):
                    industry_hints.append("K-Pop")
                    categories.append("korean_music")
                
                if is_soundtrack or industry_hints:
                    return {
                        "is_soundtrack": is_soundtrack,
                        "is_film_album": is_soundtrack,
                        "industry_hints": industry_hints,
                        "categories": categories
                    }
        
    except Exception as e:
        print(f"⚠️ Wikipedia error: {e}")
    
    return {"is_soundtrack": False, "is_film_album": False, "industry_hints": [], "categories": []}

# 11. Advanced Scoring System
def calculate_industry_scores(
    artist: str,
    title: str,
    album: Optional[str],
    musicbrainz_data: dict,
    wiki_data: dict
) -> Dict[str, float]:
    """
    Calculate confidence scores for each industry using weighted signals
    Returns: Dictionary with industry names and confidence scores (0-100)
    """
    
    db = IndustryDatabase()
    scores = {
        "Bollywood": 0.0,
        "Tollywood": 0.0,
        "Punjabi": 0.0,
        "Hollywood": 0.0,
        "K-Pop": 0.0,
        "Indie": 0.0
    }
    
    artist_lower = artist.lower()
    title_lower = title.lower()
    album_lower = album.lower() if album else ""
    
    # Combine all available text
    mb_labels = " ".join(musicbrainz_data.get("labels", [])).lower()
    mb_tags = " ".join(musicbrainz_data.get("tags", [])).lower()
    mb_countries = " ".join(musicbrainz_data.get("countries", [])).lower()
    combined_text = f"{artist_lower} {title_lower} {album_lower} {mb_labels} {mb_tags}"
    
    # Film/Soundtrack detection
    is_film_music = (
        wiki_data.get("is_soundtrack", False) or
        any(word in album_lower for word in ["soundtrack", "ost", "original motion picture", "film", "movie"])
    )
    
    # Score each industry
    for industry_name in scores.keys():
        industry_data = getattr(db, industry_name.upper().replace("-", ""))
        
        # 1. Wikipedia hints (40 points - highest weight)
        if industry_name in wiki_data.get("industry_hints", []):
            scores[industry_name] += 40
        
        # 2. Label matching (30 points)
        label_matches = sum(1 for label in industry_data["labels"] if label in combined_text)
        scores[industry_name] += min(30, label_matches * 15)  # Max 30 points
        
        # 3. Artist matching (20 points)
        artist_matches = sum(1 for artist_name in industry_data["artists"] if artist_name in artist_lower)
        scores[industry_name] += min(20, artist_matches * 20)  # Max 20 points
        
        # 4. Keyword matching (10 points)
        keyword_matches = sum(1 for keyword in industry_data["keywords"] if keyword in combined_text)
        scores[industry_name] += min(10, keyword_matches * 5)  # Max 10 points
        
        # 5. Film music boost (10 points)
        if is_film_music and "confidence_boost" in industry_data:
            boost_matches = sum(1 for word in industry_data["confidence_boost"] if word in combined_text)
            scores[industry_name] += min(10, boost_matches * 5)
        
        # 6. Country/Region matching (5 points)
        if industry_name == "Bollywood" and "in" in mb_countries:
            scores[industry_name] += 5
        elif industry_name == "Tollywood" and "in" in mb_countries:
            scores[industry_name] += 5
        elif industry_name == "K-Pop" and "kr" in mb_countries:
            scores[industry_name] += 5
        elif industry_name == "Hollywood" and any(c in mb_countries for c in ["us", "gb", "uk"]):
            scores[industry_name] += 5
    
    # Normalize scores to 0-100 range
    max_possible = 115  # 40 + 30 + 20 + 10 + 10 + 5
    for industry in scores:
        scores[industry] = min(100, (scores[industry] / max_possible) * 100)
    
    return scores

# 12. Advanced Industry Detection with Confidence Scoring
def detect_industry_advanced(
    artist: str,
    title: str,
    album: Optional[str],
    musicbrainz_data: dict,
    wiki_data: dict
) -> Tuple[str, float, Dict[str, float]]:
    """
    Advanced industry detection with confidence scoring
    Returns: (industry_name, confidence_percentage, all_scores)
    """
    
    scores = calculate_industry_scores(artist, title, album, musicbrainz_data, wiki_data)
    
    # Get top industry
    top_industry = max(scores, key=scores.get)
    top_confidence = scores[top_industry]
    
    # If confidence is too low (< 20%), use Groq AI fallback
    if top_confidence < 20:
        print(f"⚠️ Low confidence ({top_confidence:.1f}%), will use Groq fallback")
    
    return top_industry, top_confidence, scores

# 13. Groq AI Fallback for Uncertain Cases
def groq_industry_fallback(artist: str, title: str, album: str, context: str, scores: Dict[str, float]) -> str:
    """Use Groq AI when confidence is low"""
    try:
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        candidates = ", ".join([f"{ind} ({score:.1f}%)" for ind, score in top_3])
        
        prompt = (
            f"Identify the music industry for this track:\n\n"
            f"Artist: {artist}\n"
            f"Title: {title}\n"
            f"Album: {album}\n"
            f"Context: {context}\n\n"
            f"Top candidates from analysis: {candidates}\n\n"
            f"Options: Bollywood, Tollywood, Punjabi, Hollywood, K-Pop, Indie\n\n"
            f"Based on the artist name, title, and context, which industry is most likely?\n"
            f"Output ONLY the industry name, nothing else."
        )
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a music industry expert. Output only the industry name."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.2,
            max_tokens=30
        )
        
        result = chat_completion.choices[0].message.content.strip()
        valid_industries = ["Bollywood", "Tollywood", "Punjabi", "Hollywood", "K-Pop", "Indie"]
        
        if result in valid_industries:
            print(f"🤖 Groq AI override: {result}")
            return result
            
    except Exception as e:
        print(f"⚠️ Groq fallback error: {e}")
    
    return max(scores, key=scores.get)  # Return highest scored if Groq fails

# 14. Endpoints
@app.get("/")
def read_root():
    if startup_error:
        return {"status": "Maintenance Mode", "error": startup_error}
    return {"status": "Active", "version": "8.0.0", "features": ["Advanced Industry Detection", "Confidence Scoring"]}

@app.get("/health")
def health_check():
    return {
        "status": "unhealthy" if startup_error else "healthy",
        "startup_error": startup_error,
        "env_vars": {
            "GROQ_KEY_SET": bool(GROQ_API_KEY),
            "SUPABASE_URL_SET": bool(SUPABASE_URL),
            "SUPABASE_KEY_SET": bool(SUPABASE_SERVICE_ROLE_KEY)
        }
    }

@app.get("/tracks")
async def get_tracks(x_app_integrity: Optional[str] = Header(None)):
    ensure_ready()
    try:
        response = supabase.table("music_tracks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tracks")
async def add_track(track: TrackUpload, x_app_integrity: Optional[str] = Header(None)):
    ensure_ready()
    try:
        data = track.dict()
        if not data.get('tier_required'): 
            data['tier_required'] = 'free'
        response = supabase.table("music_tracks").insert(data).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record-play")
async def record_play(stat: PlayRecord, x_app_integrity: Optional[str] = Header(None)):
    ensure_ready()
    try:
        completion_rate = min(1.0, stat.listen_time_ms / stat.total_duration_ms) if stat.total_duration_ms > 0 else 0
        supabase.rpc('upsert_listening_stat', {
            'p_user_id': stat.user_id, 
            'p_track_id': stat.track_id, 
            'p_listen_time_ms': stat.listen_time_ms, 
            'p_completion_rate': completion_rate
        }).execute()
        return {"status": "recorded"}
    except Exception:
        return {"status": "error"}

@app.post("/enrich-metadata")
async def enrich_metadata(song: SongRequest, x_app_integrity: Optional[str] = Header(None)):
    ensure_ready()
    
    clean_title = song.title.strip()
    clean_artist = song.artist.strip()
    clean_album = song.album.strip() if song.album else ""
    
    cache_key = f"{clean_artist.lower()}:{clean_title.lower()}:{clean_album.lower()}"
    if cache_key in metadata_cache:
        print(f"💾 Cache hit: {cache_key}")
        return metadata_cache[cache_key]

    print(f"🔍 Analyzing: '{clean_title}' by '{clean_artist}' from '{clean_album}'")
    
    # Step 1: External data collection
    musicbrainz_data = search_musicbrainz(clean_artist, clean_title, clean_album)
    wiki_data = search_wikipedia(clean_artist, clean_album)
    
    print(f"📊 MusicBrainz: Found={musicbrainz_data.get('found')}, Labels={len(musicbrainz_data.get('labels', []))}")
    print(f"📖 Wikipedia: Soundtrack={wiki_data.get('is_soundtrack')}, Hints={wiki_data.get('industry_hints')}")
    
    # Step 2: Advanced industry detection with confidence scoring
    industry, confidence, all_scores = detect_industry_advanced(
        clean_artist, clean_title, clean_album,
        musicbrainz_data, wiki_data
    )
    
    print(f"📈 Industry Scores: {json.dumps({k: f'{v:.1f}%' for k, v in all_scores.items()}, indent=2)}")
    print(f"🏆 Winner: {industry} (Confidence: {confidence:.1f}%)")
    
    # Step 3: Groq AI fallback for low confidence
    if confidence < 25:
        context = f"Labels: {', '.join(musicbrainz_data.get('labels', [])[:3])}, Tags: {', '.join(musicbrainz_data.get('tags', [])[:3])}"
        industry = groq_industry_fallback(clean_artist, clean_title, clean_album, context, all_scores)
        print(f"🤖 Final Industry (AI Override): {industry}")
    
    # Step 4: Groq Classification for Mood, Language, Genre
    context_info = f"Artist: {clean_artist}, Title: {clean_title}, Industry: {industry}"
    if clean_album:
        context_info += f", Album: {clean_album}"
    if musicbrainz_data.get("found"):
        context_info += f", Labels: {', '.join(musicbrainz_data['labels'][:2])}"
    
    prompt = (
        f"Analyze: {context_info}\n\n"
        "Classify this music track:\n\n"
        "1. MOOD: Aggressive, Energetic, Romantic, Melancholic, Spiritual, Chill, Uplifting, Party\n"
        "2. LANGUAGE: Hindi, Telugu, Tamil, Punjabi, English, Korean, etc.\n"
        "3. GENRE: Pop, Rock, Hip-Hop, Folk, Devotional, Classical, EDM, LoFi, Jazz, Bhangra\n\n"
        "Output JSON:\n"
        '{"mood": "...", "language": "...", "genre": "..."}'
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a music metadata classifier. Output strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODELS["primary"],
            temperature=0.1,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        content = chat_completion.choices[0].message.content.strip()
        data = json.loads(content)
        
        mood = data.get("mood", "Neutral").title()
        language = data.get("language", "Unknown").title()
        genre = data.get("genre", "Pop").title()

    except Exception as e:
        print(f"⚠️ Groq classification error: {e}")
        mood, language, genre = "Neutral", "Unknown", "Pop"

    # Final result with comprehensive metadata
    result = {
        "formatted": f"{mood};{language};{industry};{genre}",
        "mood": mood,
        "language": language,
        "industry": industry,
        "genre": genre,
        "confidence": round(confidence, 2),
        "industry_scores": {k: round(v, 2) for k, v in all_scores.items()},
        "sources_used": {
            "musicbrainz": musicbrainz_data.get("found", False),
            "wikipedia": bool(wiki_data.get("industry_hints")),
            "labels_found": len(musicbrainz_data.get("labels", [])),
            "is_soundtrack": wiki_data.get("is_film_album", False),
            "groq_fallback_used": confidence < 25
        },
        "detection_metadata": {
            "labels": musicbrainz_data.get("labels", [])[:5],
            "tags": musicbrainz_data.get("tags", [])[:5],
            "countries": musicbrainz_data.get("countries", []),
            "wiki_categories": wiki_data.get("categories", [])
        }
    }
    
    metadata_cache[cache_key] = result
    print(f"✅ Final Result: {result['formatted']} (Confidence: {confidence:.1f}%)")
    
    return result

@app.get("/industry-info")
async def get_industry_info():
    """
    Endpoint to get information about supported industries
    """
    db = IndustryDatabase()
    return {
        "version": "8.0.0",
        "supported_industries": [
            {
                "name": "Bollywood",
                "description": "Hindi Film Music Industry (India)",
                "major_labels": db.BOLLYWOOD["labels"][:5],
                "sample_artists": db.BOLLYWOOD["artists"][:5],
                "market_share": "~75-80% of Indian music revenue"
            },
            {
                "name": "Tollywood",
                "description": "South Indian Film Music (Telugu, Tamil, Malayalam, Kannada)",
                "major_labels": db.TOLLYWOOD["labels"][:5],
                "sample_artists": db.TOLLYWOOD["artists"][:5],
                "market_share": "~15-20% of Indian music revenue"
            },
            {
                "name": "Punjabi",
                "description": "Punjabi Pop, Folk & Bhangra",
                "major_labels": db.PUNJABI["labels"][:3],
                "sample_artists": db.PUNJABI["artists"][:5],
                "market_share": "Growing segment in Indian music"
            },
            {
                "name": "Hollywood",
                "description": "Western Music (US, UK, Europe)",
                "major_labels": db.HOLLYWOOD["labels"][:5],
                "sample_artists": db.HOLLYWOOD["artists"][:5],
                "market_share": "Global leader (~60% worldwide revenue)"
            },
            {
                "name": "K-Pop",
                "description": "South Korean Pop Music",
                "major_labels": db.KPOP["labels"][:5],
                "sample_artists": db.KPOP["artists"][:5],
                "market_share": "~$5B+ global industry"
            },
            {
                "name": "Indie",
                "description": "Independent & Self-Released Artists",
                "major_labels": db.INDIE["labels"][:3],
                "sample_artists": ["Various emerging artists"],
                "market_share": "Rapidly growing segment"
            }
        ],
        "detection_methodology": {
            "scoring_system": "Multi-factor weighted scoring (0-100)",
            "factors": [
                {"name": "Wikipedia Industry Hints", "weight": 40},
                {"name": "Record Label Matching", "weight": 30},
                {"name": "Artist Name Matching", "weight": 20},
                {"name": "Keyword Detection", "weight": 10},
                {"name": "Film Music Context", "weight": 10},
                {"name": "Geographic/Country Data", "weight": 5}
            ],
            "fallback": "Groq AI used when confidence < 25%",
            "data_sources": ["MusicBrainz", "Wikipedia", "Groq AI", "Internal Database"]
        }
    }

@app.post("/batch-enrich")
async def batch_enrich_metadata(songs: List[SongRequest], x_app_integrity: Optional[str] = Header(None)):
    """
    Batch endpoint for enriching multiple songs at once
    Useful for bulk imports
    """
    ensure_ready()
    
    results = []
    for song in songs[:50]:  # Limit to 50 songs per request
        try:
            result = await enrich_metadata(song, x_app_integrity)
            results.append({
                "artist": song.artist,
                "title": song.title,
                "success": True,
                "metadata": result
            })
        except Exception as e:
            results.append({
                "artist": song.artist,
                "title": song.title,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total": len(songs),
        "processed": len(results),
        "results": results
    }

@app.get("/cache-stats")
async def get_cache_stats():
    """
    Get statistics about the metadata cache
    """
    return {
        "cache_size": len(metadata_cache),
        "sample_keys": list(metadata_cache.keys())[:5] if metadata_cache else [],
        "memory_usage_estimate_mb": len(json.dumps(metadata_cache)) / (1024 * 1024)
    }

@app.delete("/cache")
async def clear_cache(x_app_integrity: Optional[str] = Header(None)):
    """
    Clear the metadata cache (admin endpoint)
    """
    global metadata_cache
    cache_size = len(metadata_cache)
    metadata_cache = {}
    return {
        "status": "cleared",
        "entries_removed": cache_size
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
