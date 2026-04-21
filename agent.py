import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI
from multi_agent import planner_agent, memory_agent, rag_agent, synthesizer_agent
from tools import get_current_time, calculate_investment
from rag import query_documents, get_memory


# -------- ENV & CLIENT SETUP --------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=API_KEY)

MODEL = "gpt-4o-mini"

# -------- CONSTANTS --------

FOLLOW_UP_KEYWORDS = ["explain", "simpler", "simplify", "clarify", "rephrase"]


# -------- HELPERS --------

def is_statement(user_input: str):
    user_input = user_input.strip().lower()

    return any(x in user_input for x in [
        "i am",
        "i like",
        "i work",
        "my name",
        "i prefer"
    ])


def safe_json_loads(s, fallback):
    try:
        return json.loads(s)
    except Exception:
        return fallback


def format_memories(memories):
    return " ".join(memories)


def is_follow_up_query(user_input: str, history) -> bool:
    if not history:
        return False

    normalized = user_input.lower().strip()
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
    return [float(m) for m in matches]


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


# -------- MAIN AGENT --------

def run_agent(user_input: str, history=None, user_id="default"):
    if history is None:
        history = []

    history = history[-6:]

    # -------- MEMORY --------
    memories = get_memory(user_input, user_id)

    memory_context = ""
    if memories:
        memory_context = "Relevant past info:\n"
        for m in memories:
            memory_context += f"- {m}\n"

    print("\n🧠 MEMORY CONTEXT PASSED TO LLM:\n", memory_context)

    # -------- MEMORY QUESTION --------
    if is_memory_question(user_input):
        print("🧠 Memory question → direct answer")

        if memories:
            return json.dumps({
                "answer": format_memories(memories),
                "details": []
            })
        else:
            return json.dumps({
                "answer": "I don’t have any saved information about you yet.",
                "details": []
            })

    # -------- FOLLOW-UP --------
    if is_follow_up_query(user_input, history):
        return handle_follow_up(user_input, history, memory_context)

    user_lower = user_input.lower()

    # -------- MIXED QUERY --------
    if is_calculation_query(user_input) and any(
        word in user_lower for word in ["what", "explain", "define"]
    ):
        print("🔥 Mixed query")

        params, error = parse_investment_params(user_input)

        if error:
            return json.dumps({"answer": error, "details": []})

        amount, rate, years = params
        result = calculate_investment(amount, rate, years)

        return json.dumps({
            "answer": f"${amount:,.0f} → ${result:,.2f}",
            "details": []
        })

    # -------- PURE CALC --------
    if is_calculation_query(user_input):
        print("💰 Direct calculation")

        params, error = parse_investment_params(user_input)

        if error:
            return json.dumps({"answer": error, "details": []})

        amount, rate, years = params
        result = calculate_investment(amount, rate, years)

        return json.dumps({
            "answer": (
                f"An investment of {amount:,.0f} at {rate*100:.1f}% "
                f"for {years:.0f} years will grow to approximately ${result:,.2f}."
            ),
            "details": []
        })

    # -------- STATEMENT --------
    if is_statement(user_input):
        print("🧠 Statement detected → saving memory")

        from rag import save_memory
        save_memory(user_input, user_id)

        return json.dumps({
            "answer": "Got it — I’ll remember that.",
            "details": []
        })

    # -------- MULTI-AGENT --------
    plan = planner_agent(user_input)

    results = []

    for step in plan:
        if step["type"] == "memory":
            results.append(memory_agent(step["task"], user_id))

        elif step["type"] == "rag":
            results.append(rag_agent(step["task"]))

        elif step["type"] == "tool":
            params, error = parse_investment_params(step["task"])

            if error:
                results.append(error)
            else:
                amount, rate, years = params
                result = calculate_investment(amount, rate, years)

                results.append(
                    f"An investment of {amount:,.0f} at {rate*100:.1f}% "
                    f"for {years:.0f} years will grow to approximately ${result:,.2f}."
                )

    return synthesizer_agent(results)