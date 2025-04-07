from fastapi import FastAPI, Request
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import httpx
import uuid
import os

app = FastAPI()

# --- Qdrant
QDRANT_URL = "https://434bc49c-5c75-4a57-a104-55a27b6e5ba6.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ma_qYDVyytgTSBxdVLK_bj565_d56F1n80y_yOyb1BA"
COLLECTION = "jurist_docs"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- Проверяем и создаём коллекцию
if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# --- Получение эмбеддинга через OpenAI
async def embed_text(text: str):
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=30.0, connect=10.0)) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-ada-002",
                "input": text
            }
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

# --- Добавление документа
@app.post("/add-doc")
async def add_doc(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return {"error": "text is required"}

    vector = await embed_text(text)

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text}
            )
        ]
    )

    return {"status": "added", "text": text}

# --- Чат с использованием памяти из Qdrant
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_messages = body.get("messages", [])
    user_input = user_messages[-1]["content"] if user_messages else ""

    # 1. Векторизуем запрос
    query_vector = await embed_text(user_input)

    # 2. Ищем 1 короткий документ в Qdrant
    search_result = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=1  # Ограничили до одного
    )

    # 3. Собираем контекст
    context = "\n---\n".join([hit.payload["text"] for hit in search_result if "text" in hit.payload])

    # 4. Формируем сообщения
    system_message = {
        "role": "system",
        "content": f"Ты — профессиональный AI-юрист. Используй только проверенную информацию. Контекст:\n{context}"
    }

    messages = [system_message] + user_messages

    # 5. Отправляем в OpenAI Chat API
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=60.0)) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": messages,
                "temperature": 0.7
            }
        )

        response.raise_for_status()
        return response.json()


        response.raise_for_status()
        return response.json()
