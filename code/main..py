# main.py

import asyncio

from agent import Agent
from dotenv import load_dotenv

load_dotenv()

async def main():
    agent = Agent(
        mcp_config="mcp.json",
    )

    try:
        await agent.start()

        print()
        print("================================")
        print("        Gemini MCP Agent")
        print("================================")
        print("Type 'exit' or 'quit' to exit.")
        print()

        while True:
            try:
                query = input("You: ").strip()

            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                break

            try:
                response = await agent.get_llm_response(query)

                print()
                print(f"Assistant: {response}")
                print()

            except Exception as exc:
                print()
                print(
                    f"Error: {type(exc).__name__}: {exc}"
                )
                print()

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())