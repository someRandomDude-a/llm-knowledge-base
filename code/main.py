# main.py

import asyncio
import logging
import os

from agent import Agent
from dotenv import load_dotenv

load_dotenv()

# Configure logging so we see messages from MCP/Gemini
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    agent = Agent(
        mcp_config="mcp.json",
    )

    try:
        logger.info("Starting agent...")
        await agent.start()
        logger.info("Agent started successfully.")

        # List available tools
        tools = agent.list_tools()
        if tools:
            print("\nAvailable tools:")
            for t in tools:
                print(f"  - {t['name']}: {t['description']}")
        else:
            print("\nNo tools loaded. Check your mcp.json configuration.")

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
                logger.exception("Error during query")
                print()
                print(f"Error: {type(exc).__name__}: {exc}")
                print()

    except Exception as exc:
        logger.exception("Failed to start agent")
        print(f"Startup error: {exc}")
    finally:
        await agent.close()
        logger.info("Agent closed.")


if __name__ == "__main__":
    asyncio.run(main())