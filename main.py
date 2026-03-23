from agent import run_agent

while True:
    user_input = input("\nAsk something (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    result = run_agent(user_input)
    print("\nAI:\n", result)