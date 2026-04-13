import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

from tools import get_current_time, calculate_investment
from rag import query_documents
from rag import get_memory

# -------- ENV & CLIENT SETUP --------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4o-mini"

# -------- CONSTANTS --------

FOLLOW_UP_KEYWORDS = ["explain", "simpler", "simplify", "clarify", "rephrase"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns current time",
            "parameters": {"type": "object", "properties": {}},
        },
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
                    "years": {"type": "number"},
                },
                "required": ["amount", "rate", "years"],
            },
        },
    },
]

# -------- HELPERS --------

def is_statement(user_input: str):
    user_input = user_input.strip().lower()

    # simple heuristic: not a question
    return not user_input.endswith("?")

def safe_json_loads(s, fallback):
    try:
        return json.loads(s)
    except Exception:
        return fallback

def format_memories(memories):
    return " ".join(memories)

def execute_tool(tool_call):
    name = tool_call.function.name

    if name == "get_current_time":
        return get_current_time()

    if name == "calculate_investment":
        args = safe_json_loads(tool_call.function.arguments, {})
        return calculate_investment(
            args.get("amount"),
            args.get("rate"),
            args.get("years"),
        )

    return "Unknown tool"


def is_follow_up_query(user_input: str, history) -> bool:
    if not history:
        return False

    normalized = user_input.lower().strip()
    is_short = len(normalized.split()) <= 5
    has_keyword = any(k in normalized for k in FOLLOW_UP_KEYWORDS)
    return has_keyword

def is_memory_question(user_input: str):
    user_input = user_input.lower()

    return any(x in user_input for x in [
        "what do i",
        "who am i",
        "what is my",
        "what do you know about me"
    ])

def extract_numbers(step: str):
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%?", step)
    numbers = []

    for m in matches:
        try:
            numbers.append(float(m))
        except:
            pass

    return numbers

def parse_investment_params(step: str):
    numbers = extract_numbers(step)
    print("🔢 Extracted numbers:", numbers)

    if len(numbers) < 3:
        return None, "Could not extract enough numbers."

    amount, rate, years = numbers[:3]

    if rate > 1:
        rate = rate / 100

    return (amount, rate, years), None


def is_calculation_query(step: str):
    step_lower = step.lower()

    has_numbers = bool(re.search(r"\d", step))
    has_keywords = any(
        word in step_lower for word in ["calculate", "%", "interest", "years"]
    )

    return has_numbers and has_keywords


# -------- PLANNER --------

def create_plan(user_input: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
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
""",
            },
            {"role": "user", "content": user_input},
        ],
    )

    content = response.choices[0].message.content or ""
    plan = safe_json_loads(content, {"steps": [user_input]})

    if not isinstance(plan, dict):
        plan = {"steps": [user_input]}

    if "steps" not in plan or not isinstance(plan["steps"], list):
        plan["steps"] = [user_input]

    return plan


# -------- FOLLOW-UP --------

def handle_follow_up(user_input: str, history, memory_context=""):
    print("🧠 Follow-up → skipping planner")

    messages = [
  {
        "role": "system",
        "content": f"""
You are a conversational AI assistant.

CRITICAL:
- This is a follow-up
- Use memory if relevant
- Do NOT introduce new ideas

Memory:
{memory_context}

Return JSON:
{{
  "answer": "",
  "details": []
}}
"""
    },
    *history,
    {"role": "user", "content": user_input},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return response.choices[0].message.content


# -------- UNIFIED STEP (RAG + TOOLS + LLM) --------

def execute_unified_step(step, context, history, memory_context=""):
    system_content = f"""
You are an AI assistant.

CRITICAL RULES:
- Answer ALL parts of the question
- Use context if relevant
- Use memory if it helps personalize the answer
- Do NOT perform financial calculations yourself
- If calculation already provided, do NOT repeat it

Return ONLY JSON:
{{
  "answer": "",
  "details": []
}}

Memory:
{memory_context}

Context:
{context}
"""

    messages = [
        {"role": "system", "content": system_content},
        *history,
        {"role": "user", "content": step},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        response_format={"type": "json_object"},
    )

    msg = response.choices[0].message

    # handle tool calls
    if msg.tool_calls:
        tool_messages = []
        for tool_call in msg.tool_calls:
            result = execute_tool(tool_call)

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": json.dumps(result),
            })

        final = client.chat.completions.create(
            model=MODEL,
            messages=messages + [msg] + tool_messages,
            response_format={"type": "json_object"},
        )

        return final.choices[0].message.content

    return msg.content

# -------- SYNTHESIS --------

def synthesize_final_answer(final_answers: list) -> str:
    combined_input = "\n\n".join(final_answers)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Combine into ONE answer.

Rules:
- Do NOT add new information
- Do NOT duplicate content
- Merge cleanly

Return JSON:
{
  "answer": "",
  "details": []
}
""",
            },
            {"role": "user", "content": combined_input},
        ],
    )

    return response.choices[0].message.content


# -------- MAIN AGENT --------
def run_agent(user_input: str, history=None, user_id="default"):
    if history is None:
        history = []

    history = history[-6:]

    # ✅ ADD HERE
    memories = get_memory(user_input, user_id)

    memory_context = ""

    if memories:
        memory_context = "Relevant past info:\n"
        for m in memories:
            memory_context += f"- {m}\n"

    # 🔥 ADD THIS LINE (THIS IS STEP 2)
    print("\n🧠 MEMORY CONTEXT PASSED TO LLM:\n", memory_context)

    # -------- MEMORY QUESTION (FIX) --------
    if is_memory_question(user_input):
        print("🧠 Memory question → direct answer")

        if memories:
            return json.dumps({
                "answer": format_memories(memories),
                "details": []
            })
    # -------- FOLLOW-UP --------
    if is_follow_up_query(user_input, history):
        return handle_follow_up(user_input, history, memory_context)
    
    user_lower = user_input.lower()

    # -------- MIXED QUERY (EXPLAIN + CALC) --------
    if is_calculation_query(user_input) and any(
        word in user_lower for word in ["what", "explain", "define", "partition"]
    ):
        print("🔥 Mixed query (manual split)")

        # ---- 1. RAG + LLM (EXPLANATION ONLY) ----
        retrieved_docs = query_documents(user_input)
        context = "\n\n".join(retrieved_docs[0]) if retrieved_docs else ""

        # remove calculation part so LLM doesn't do math
        clean_input = re.sub(
            r"(calculate.*|what will.*be.*)", 
            "", 
            user_input, 
            flags=re.IGNORECASE
        ).strip()

        explanation = execute_unified_step(
        clean_input or user_input,
        context,
        history,
        memory_context
    )
        # ---- 2. TOOL (CALCULATION ONLY) ----
        params, error = parse_investment_params(user_input)

        if error:
            calc_output = json.dumps({
                "answer": error,
                "details": []
            })
        else:
            amount, rate, years = params
            result = calculate_investment(amount, rate, years)

            calc_output = json.dumps({
                "answer": (
                    f"An investment of ${amount:,.0f} at {rate*100:.1f}% "
                    f"for {years} years will grow to approximately ${result:,.2f}."
                ),
                "details": []
            })

        return synthesize_final_answer([explanation, calc_output])

    # -------- PURE CALC (BYPASS PLANNER) --------
    if is_calculation_query(user_input):
        print("💰 Direct calculation (bypass planner)")

        params, error = parse_investment_params(user_input)

        if error:
            return json.dumps({
                "answer": error,
                "details": []
            })

        amount, rate, years = params
        result = calculate_investment(amount, rate, years)

        return json.dumps({
            "answer": (
               f"An investment of {amount:,.0f} at {rate*100:.1f}% for {years:.0f} years will grow to approximately ${result:,.2f}."
            ),
            "details": []
        })

    # -------- NORMAL FLOW (RAG + PLANNER) --------
    if is_statement(user_input):
        print("🧠 Statement detected → no planning")

        return json.dumps({
            "answer": "Got it.",
            "details": []
        })
    plan = create_plan(user_input)
    steps = plan.get("steps", [user_input])

    final_answers = []

    for step in steps:
        print(f"➡️ Step: {step}")

        retrieved_docs = query_documents(step)

        context = ""
        if retrieved_docs:
            if isinstance(retrieved_docs[0], list):
                context = "\n\n".join(retrieved_docs[0])
            else:
                context = "\n\n".join(retrieved_docs)

        res = execute_unified_step(step, context, history, memory_context)
        final_answers.append(res)

    return synthesize_final_answer(final_answers)