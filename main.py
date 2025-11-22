import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# --- DATA MODELS ---
class AIRequest(BaseModel):
    prompt: str  # The text sent from Flutter

# --- ROUTES ---

# 1. Health Check (For UptimeRobot)
@app.get("/")
def health_check():
    return {"status": "active", "message": "AudioVibe Backend is Running"}

# 2. Groq AI Endpoint (Secure)
@app.post("/ask-ai")
def ask_groq_ai(request: AIRequest):
    # Get Groq Key from Render Vault
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="Server Error: Groq API Key missing")

    client = Groq(api_key=api_key)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        return {"response": chat_completion.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}

# 3. Database Endpoint (Secure Proxy)
@app.get("/songs")
async def get_songs():
    # Get DB Secrets from Render Vault
    db_url = os.environ.get("DATABASE_URL")
    db_key = os.environ.get("DATABASE_API_KEY")

    if not db_url or not db_key:
        raise HTTPException(status_code=500, detail="Server Error: DB Config missing")

    try:
        # We act as a 'middleman' so the app never sees the DB key
        async with httpx.AsyncClient() as client:
            response = await client.get(
                db_url,
                headers={
                    "Authorization": f"Bearer {db_key}",
                    "apikey": db_key,
                    "Content-Type": "application/json"
                }
            )
            return response.json()
            
    except Exception as e:
        return {"error": str(e)}
