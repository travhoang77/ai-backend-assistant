import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from rag import query_documents, get_memory

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini"


def planner_agent(user_input):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are an AI planner.

Decide which agents to use.

Available agents:
- memory → for user-related questions
- rag → for knowledge/explanations
- tool → for calculations

Return JSON:
{
  "steps": [
    {"type": "memory", "task": "..."},
    {"type": "rag", "task": "..."}
  ]
}
"""
            },
            {"role": "user", "content": user_input}
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    try:
        plan = json.loads(content)
        return plan.get("steps", [])
    except Exception as e:
        print("⚠️ Planner failed:", e)
        return [{"type": "rag", "task": user_input}]


def memory_agent(task, user_id):
    memories = get_memory(task, user_id)

    if not memories:
        return "I don’t have any saved information about you yet."

    return " ".join(memories)


def rag_agent(task):
    docs = query_documents(task)

    if docs and isinstance(docs[0], list) and docs[0]:
        return "\n".join(docs[0])

    return "I don’t have enough information to answer that yet."


def tool_agent(task):
    return "[Tool agent not fully implemented yet]"


def synthesizer_agent(results):
    clean = [r for r in results if r and r.strip()]

    if not clean:
        return json.dumps({
            "answer": "I couldn’t find a good answer.",
            "details": []
        })

    combined = "\n\n".join(clean)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are an AI assistant.

Combine the following pieces of information into ONE clear, natural answer.

Rules:
- Be concise
- Do NOT repeat information
- Do NOT add new facts
- Sound natural and conversational

Return JSON:
{
  "answer": "",
  "details": []
}
"""
            },
            {"role": "user", "content": combined}
        ],
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content