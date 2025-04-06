from fastapi import FastAPI, Request
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import httpx
import os
import uuid

# --- Настройки
app = FastAPI()
qdrant_host = "https://434bc49c-5c75-4a57-a104-55a27b6e5ba6.eu-central-1-0.aws.cloud.qdrant.io:6333"
qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ma_qYDVyytgTSBxdVLK_bj565_d56F1n80y_yOyb1BA"  # замени на свой
collection_name = "jurist_docs"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Qdrant клиент
qdrant = QdrantClient(
    url=qdrant_host,
    api_key=qdrant_api_key
)

# --- Создаём коллекцию при старте (если нет)
if collection_name not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# --- FastAPI эндпоинт: загрузка текста в память
@app.post("/add-doc")
async def add_doc(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return {"error": "text is required"}

    embedding = embedding_model.encode(text).tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": text}
    )

    qdrant.upsert(collection_name=collection_name, points=[point])
    return {"status": "added", "text": text}

# --- (Оставь старый /chat здесь если он был)
