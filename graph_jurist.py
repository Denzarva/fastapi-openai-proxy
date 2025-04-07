from langgraph.graph import StateGraph
from typing import TypedDict
import os
import httpx

# --- Тип состояния графа
class GraphState(TypedDict):
    messages: list
    context: str

# --- Запрос к OpenAI Chat Completion
async def call_openai(messages, context):
    system_prompt = f"Ты — AI-юрист. Используй только проверенные источники. Контекст:\n{context}"
    chat_messages = [{"role": "system", "content": system_prompt}] + messages

    print("📡 Отправляю запрос в OpenAI...")  # ✅ лог перед отправкой

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=60.0, connect=10.0)) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": chat_messages,
                "temperature": 0.7
            }
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

# --- Узел генерации ответа
async def generate_answer(state: GraphState):
    messages = state["messages"]
    context = state["context"]
    response = await call_openai(messages, context)
    return {"messages": messages, "context": context, "response": response}

# --- Сборка и запуск LangGraph
def run_jurist_graph(user_messages, context):
    builder = StateGraph(GraphState)
    builder.add_node("generate", generate_answer)
    builder.set_entry_point("generate")
    builder.set_finish_point("generate")
    graph = builder.compile()

    state = {
        "messages": user_messages,
        "context": context
    }

    result = graph.invoke(state)
    return result["response"]
