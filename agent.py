from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv
from tools import get_current_time, calculate_investment
from rag import query_documents

load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)

# -------- PLANNER --------
def create_plan(user_input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are an AI planner.

Break the user request into steps.

IMPORTANT:
- Each step MUST be a string
- Do NOT return objects

Return JSON:
{
  "steps": ["step 1", "step 2"]
}
"""
            },
            {"role": "user", "content": user_input}
        ]
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("Plan parsing failed:", e)
        return {"steps": [user_input]}


# -------- TOOLS --------
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


def execute_tool(tool_call):
    if tool_call.function.name == "get_current_time":
        return get_current_time()

    elif tool_call.function.name == "calculate_investment":
        args = json.loads(tool_call.function.arguments)
        return calculate_investment(
            args["amount"],
            args["rate"],
            args["years"]
        )

    return "Unknown tool"


# -------- AGENT --------
def run_agent(user_input):

    import re

    # 🔥 Helper: extract investment inputs safely
    def extract_investment_params(text):
        numbers = re.findall(r"\d+\.?\d*", text)

        if len(numbers) >= 3:
            amount = float(numbers[0])
            rate = float(numbers[1]) / 100
            years = float(numbers[2])
            return amount, rate, years

        return None

    # 🔥 Step 1: Create plan
    plan = create_plan(user_input)
    print("Plan:", plan)

    # 🔥 Normalize steps
    raw_steps = plan.get("steps", [user_input])
    steps = []

    for s in raw_steps:
        if isinstance(s, dict):
            steps.append(s.get("step", str(s)))
        else:
            steps.append(s)

    final_answers = []

    # 🔥 Step 2: Execute steps
    for step in steps:
        print("\n--- Executing step ---")
        print("Step:", step)

        # 🔥 Detect tool needs
        step_lower = step.lower()
        needs_calculation = any(word in step_lower for word in [
            "calculate", "investment", "growth", "future value"
        ])
        needs_time = "time" in step_lower

        # 🔥 FORCE deterministic calculation (NEW FIX)
        params = extract_investment_params(user_input)

        if params and needs_calculation:
            amount, rate, years = params

            result = calculate_investment(amount, rate, years)

            print("🔧 Forced calculation:", result)

            final_answers.append(json.dumps({
                "answer": f"An investment of ${amount:,.0f} at {rate*100:.1f}% for {years} years will grow to approximately ${result:,.2f}.",
                "details": [
                    f"Amount: {amount}",
                    f"Rate: {rate}",
                    f"Years: {years}",
                    f"Final Value: {result:.2f}"
                ]
            }))

            continue

        # 🔍 RAG retrieval
        retrieved_docs = query_documents(step)

        context_chunks = []
        if retrieved_docs and len(retrieved_docs) > 0:
            context_chunks = retrieved_docs[0]

        context = "\n\n".join(
            [f"Context {i+1}: {doc}" for i, doc in enumerate(context_chunks)]
        )

        print("\n=== CONTEXT ===\n", context)

        # 🔧 Tool instruction
        tool_instruction = ""
        if needs_calculation:
            tool_instruction = "You MUST call calculate_investment for this step."
        elif needs_time:
            tool_instruction = "You MUST call get_current_time for this step."

        messages = [
            {
                "role": "system",
                "content": f"""
You are a backend engineering AI assistant.

CRITICAL RULES:
- The CONTEXT below is retrieved from uploaded documents
- If CONTEXT is not empty, you MUST use it
- Do NOT ignore context
- Do NOT say you cannot access documents

TOOL RULES:
- If a tool is required, you MUST call it
- If you call a tool, you MUST use its result
- Do NOT manually calculate

{tool_instruction}

Available tools:
- get_current_time
- calculate_investment

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
            },
            {
                "role": "user",
                "content": f"""
Original request:
{user_input}

Current step:
{step}
"""
            }
        ]

        MAX_TOOL_STEPS = 3

        for i in range(MAX_TOOL_STEPS):
            print(f"\n--- Tool loop {i+1} ---")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if message.tool_calls:
                print("Tool call detected")

                messages.append(message)

                for tool_call in message.tool_calls:
                    print("Calling tool:", tool_call.function.name)

                    result = execute_tool(tool_call)

                    print("Tool result:", result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })

            else:
                result = message.content
                print("Step result:", result)

                final_answers.append(result)
                break

    # 🔥 Step 3: FINAL SYNTHESIS
    combined_input = "\n\n".join(final_answers)

    print("\n🔧 Synthesizing final answer...\n")

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are an AI assistant.

Combine the following step-by-step outputs into ONE clean, structured answer.

Rules:
- Do not repeat information
- Merge overlapping ideas
- Keep it concise but complete
- Ensure valid JSON output

Return JSON:
{
  "answer": "",
  "details": []
}
"""
            },
            {
                "role": "user",
                "content": combined_input
            }
        ]
    )

    final_output = final_response.choices[0].message.content

    print("Final Output:", final_output)

    return final_output