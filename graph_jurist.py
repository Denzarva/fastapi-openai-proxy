from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable import Runnable
from langchain.chat_models import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import os

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

def generate_answer(state):
    messages = state["messages"]
    context = state["context"]

    system = SystemMessage(content=f"Ты — профессиональный AI-юрист. Используй только проверенные данные из контекста:\n{context}")
    history = [system] + [HumanMessage(**m) if m["role"] == "user" else AIMessage(**m) for m in messages]

    answer = llm.invoke(history)
    state["messages"].append({"role": "assistant", "content": answer.content})
    return state

graph = StateGraph()
graph.add_node("generate", generate_answer)
graph.set_entry_point("generate")
graph.set_finish_point("generate")

app_graph: Runnable = graph.compile()

async def run_jurist_graph(messages: list[dict], context: str) -> str:
    state = {
        "messages": messages,
        "context": context
    }
    result = app_graph.invoke(state)
    return result["messages"][-1]["content"]
