from fastapi import FastAPI, Request
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import httpx
import uuid
import os

app = FastAPI()

# Qdrant
QDRANT_URL = "https://434bc49c-5c75-4a57-a104-55a27b6e5ba6.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ma_qYDVyytgTSBxdVLK_bj565_d56F1n80y_yOyb1BA"
COLLECTION = "jurist_docs"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Создаём коллекцию, если её нет
if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# --- Получение вектора из Qdrant API
async def embed_text(text: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://qdrant-api.qdrant.tech/v1/models/text-embedding-ada-002/infer",
            headers={"Authorization": f"Bearer {QDRANT_API_KEY}"},
            json={"input": text}
        )
        return resp.json()["result"]

# --- Загрузка документа
@app.post("/add-doc")
async def add_doc(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return {"error": "text is required"}

    vector = await embed_text(text)

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": text})]
    )

    return {"status": "added", "text": text}
