from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # Пример кастомизации:
    custom_message = {
        "role": "system",
        "content": "Ты — профессиональный AI-юрист. Отвечай кратко и строго в рамках законодательства РФ."
    }
    messages.insert(0, custom_message)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4",
        "messages": messages,
        "temperature": 0.7
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_URL, headers=headers, json=payload)
        return response.json()
