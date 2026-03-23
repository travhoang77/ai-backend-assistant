from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from tools import get_current_time, calculate_investment
from rag import query_documents

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns current time",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_investment",
            "description": "Calculate investment growth",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "rate": {"type": "number"},
                    "years": {"type": "number"}
                },
                "required": ["amount", "rate", "years"]
            }
        }
    }
]

conversation_history = []


def run_agent(user_input):
    retrieved_docs = query_documents(user_input)

    # 🔥 SAFE extraction
    context_chunks = []
    if retrieved_docs and len(retrieved_docs) > 0:
        context_chunks = retrieved_docs[0]

    context = "\n\n".join(
        [f"Context {i+1}: {doc}" for i, doc in enumerate(context_chunks)]
    )

    print("Final context passed to LLM:\n", context)  # 🔥 DEBUG

    conversation_history.append({"role": "user", "content": user_input})

    messages = [
        {
            "role": "system",
            "content": f"""
You are a backend engineering AI assistant.

Rules:
- If context is EMPTY → say:
  "No relevant documents found. Please upload a document first or try again shortly."
- If context EXISTS → you MUST use it

Context:
---------------------
{context}
---------------------

Return JSON:
{{
  "answer": "",
  "details": []
}}
"""
        }
    ] + conversation_history

    MAX_STEPS = 5

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                if tool_call.function.name == "get_current_time":
                    result = get_current_time()

                elif tool_call.function.name == "calculate_investment":
                    args = json.loads(tool_call.function.arguments)
                    result = calculate_investment(
                        args["amount"],
                        args["rate"],
                        args["years"]
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        else:
            final_output = message.content
            conversation_history.append({
                "role": "assistant",
                "content": final_output
            })
            return final_output