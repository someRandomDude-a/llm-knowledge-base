# main.py
"""
Interactive CLI for local testing.

Uses AgentManager (same as the HTTP server) so the code path exercised
here is identical to the one used in production, minus the HTTP layer.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from agent import AgentManager

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    manager = AgentManager(mcp_config="mcp.json")

    try:
        logger.info("Starting agent manager...")
        await manager.start()
        logger.info("Agent manager started.")

        tools = manager.list_tools()
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
                response = await manager.chat(user_id="cli", query=query)
                print()
                print(f"Assistant: {response}")
                print()
            except Exception as exc:
                logger.exception("Error during query")
                print()
                print(f"Error: {type(exc).__name__}: {exc}")
                print()

    except Exception as exc:
        logger.exception("Failed to start agent manager")
        print(f"Startup error: {exc}")
    finally:
        await manager.close()
        logger.info("Agent manager closed.")


if __name__ == "__main__":
    asyncio.run(main())