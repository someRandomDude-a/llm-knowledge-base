# agent.py

from gemini import GeminiCore
from context import (
    ContextBuilder,
    Conversation,
    MemoryStore,
)


DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant.

You have access to external tools through MCP.

Use tools when they are useful or necessary to answer
the user's request.

Do not claim that you performed an action unless you
actually performed it.

Use the conversation history and memories as context.

Be concise by default, but provide detail when useful.
""".strip()


class Agent:
    """
    High-level AI agent.

    This is the class the rest of the application interacts with.

    The application should generally not need to know that Gemini
    or MCP exists.
    """

    def __init__(
        self,
        mcp_config: str = "mcp.json",
        model: str = "gemini-3.6-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.llm = GeminiCore(
            mcp_config_path=mcp_config,
            model=model,
        )

        self.conversation = Conversation()

        self.memory = MemoryStore()

        self.context = ContextBuilder(
            system_prompt=system_prompt,
            conversation=self.conversation,
            memory=self.memory,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        await self.llm.start()

    async def close(self):
        await self.llm.close()

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    async def get_llm_response(
        self,
        query: str,
    ) -> str:
        """
        Main agent function.

        Everything goes through here.
        """

        # Build context BEFORE adding the current message.
        prompt = self.context.build(
            query
        )

        # Ask Gemini.
        response = await self.llm.get_response(
            prompt
        )

        # Store both sides of the exchange.
        self.conversation.add_user(
            query
        )

        self.conversation.add_assistant(
            response
        )

        return response

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def remember(
        self,
        memory: str,
    ):
        self.memory.add(memory)

    def forget(
        self,
        memory: str,
    ):
        self.memory.remove(memory)

    def get_memories(self) -> list[str]:
        return self.memory.all()

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def clear_conversation(self):
        self.conversation.clear()

    def get_conversation(self):
        return self.conversation.messages

    # ------------------------------------------------------------------
    # Interactive mode
    # ------------------------------------------------------------------

    async def run(self):

        print()
        print("Agent ready.")
        print("Type 'exit' or 'quit' to stop.")
        print()

        while True:

            try:
                query = input("You: ").strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):
                print()
                break

            if not query:
                continue

            if query.lower() in {
                "exit",
                "quit",
            }:
                break

            try:

                response = (
                    await self.get_llm_response(
                        query
                    )
                )

                print()
                print("Assistant:")
                print(response)
                print()

            except Exception as exc:

                print()
                print(
                    f"Error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
                print()