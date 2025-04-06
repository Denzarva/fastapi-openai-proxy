from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])

        # Кастомизация
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

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(OPENAI_URL, headers=headers, json=payload)
            response.raise_for_status()  # выбросит исключение, если не 200
            return response.json()

    except httpx.HTTPError as http_exc:
        return {"error": "HTTP error", "details": str(http_exc)}
    except Exception as e:
        return {"error": "Server error", "details": str(e)}
