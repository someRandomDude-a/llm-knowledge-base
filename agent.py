# agent.py

import os
import threading
from typing import Any, Optional

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


def _default_window_size() -> int:
    """
    Rolling-window size in number of messages (user+assistant
    turns combined). Override with the CHAT_WINDOW_SIZE env var.
    """
    try:
        return max(1, int(os.getenv("CHAT_WINDOW_SIZE", "20")))
    except ValueError:
        return 20


class Agent:
    """
    High-level AI agent.

    Backwards compatible with the original CLI usage:

        agent = Agent(mcp_config="mcp.json")
        await agent.start()
        response = await agent.get_llm_response(query)
        await agent.close()

    In API mode a single `GeminiCore` is shared across many users
    and a `ChatHistoryDB` is passed in for persistent context.
    """

    def __init__(
        self,
        mcp_config: str = "mcp.json",
        model: str = "gemini-3.6-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        *,
        user_id: Optional[str] = None,
        db: Optional[Any] = None,
        llm: Optional[GeminiCore] = None,
        window_size: Optional[int] = None,
    ):

        if llm is None:
            # Backwards-compatible: own the LLM.
            self.llm = GeminiCore(
                mcp_config_path=mcp_config,
                model=model,
            )
            self._owns_llm = True
        else:
            # API mode: the manager owns the LLM.
            self.llm = llm
            self._owns_llm = False

        self.user_id = user_id
        self.db = db
        self.window_size = (
            window_size
            if window_size is not None
            else _default_window_size()
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
        if self._owns_llm:
            await self.llm.start()

    async def close(self):
        if self._owns_llm:
            await self.llm.close()

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, str]]:
        """
        Return the JSON-friendly list of available tools.

        Format:
            [{"name": "...", "description": "..."}, ...]
        """
        return self.llm.list_mcp_tools()

    # ------------------------------------------------------------------
    # Tool filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_tools(
        llm: GeminiCore,
        enabled_names: Optional[list[str]],
    ) -> Optional[list[Any]]:
        """
        Restrict the toolset exposed to Gemini for one request.

        - `enabled_names is None`  -> keep all tools
        - `enabled_names == []`    -> disable all tools
        - otherwise                -> keep only the named tools
        """
        if enabled_names is None:
            return None

        if not enabled_names:
            return []

        wanted = {
            str(name) for name in enabled_names
        }

        # Track which requested names we actually matched, so
        # the caller can spot typos / unknown tools.
        matched: set[str] = set()

        from google.genai import types  # local import keeps the
                                        # top of the file light

        filtered: list[types.Tool] = []

        for tool in llm.gemini_tools:

            declarations = (
                tool.function_declarations or []
            )

            kept = [
                decl
                for decl in declarations
                if decl.get("name") in wanted
            ]

            for decl in kept:
                matched.add(decl["name"])

            if kept:
                filtered.append(
                    types.Tool(
                        function_declarations=kept
                    )
                )

        unknown = wanted - matched
        if unknown:
            print(
                f"[Agent] Unknown tool names requested: "
                f"{sorted(unknown)}"
            )

        return filtered

    # ------------------------------------------------------------------
    # Persistent rolling window
    # ------------------------------------------------------------------

    def _load_window(self) -> None:
        """
        Replace the in-memory conversation with the most recent
        `window_size` messages from the DB.

        No rows are deleted - this is a read.
        """
        if self.db is None or not self.user_id:
            return

        recent = self.db.get_recent_messages(
            self.user_id,
            self.window_size,
        )

        self.conversation.messages.clear()
        self.conversation.messages.extend(recent)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    async def get_llm_response(
        self,
        query: str,
        enabled_tools: Optional[list[str]] = None,
    ) -> str:
        """
        Main agent function.

        Everything goes through here.
        """

        # 1. Pull the rolling window from the DB (if persistent).
        self._load_window()

        # 2. Build the prompt from the current conversation
        #    (which is now the rolling window) plus the new query.
        prompt = self.context.build(query)

        # 3. Ask Gemini, with an optional tool filter.
        tools_to_use = self._filter_tools(
            self.llm,
            enabled_tools,
        )

        response = await self.llm.get_response(
            prompt,
            tools=tools_to_use,
        )

        # 4. Persist both sides of the exchange (if persistent).
        #    Past interactions are never deleted.
        if self.db is not None and self.user_id:
            self.db.add_message(
                self.user_id,
                "user",
                query,
                enabled_tools,
            )
            self.db.add_message(
                self.user_id,
                "assistant",
                response,
                enabled_tools,
            )

        # 5. Update the in-memory conversation so the same
        #    process can keep chatting within this turn.
        self.conversation.add_user(query)
        self.conversation.add_assistant(response)

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


class AgentManager:
    """
    Manages one shared `GeminiCore` (and its MCP connections)
    across many per-user `Agent` instances.

    This is what the Flask API uses.
    """

    def __init__(
        self,
        mcp_config: str = "mcp.json",
        model: str = "gemini-3.6-flash",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        *,
        db: Optional[Any] = None,
        window_size: Optional[int] = None,
    ):
        self.llm = GeminiCore(
            mcp_config_path=mcp_config,
            model=model,
        )
        self.system_prompt = system_prompt
        self.db = db
        self.window_size = (
            window_size
            if window_size is not None
            else _default_window_size()
        )

        self._agents_lock = threading.Lock()
        self._agents: dict[str, Agent] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        await self.llm.start()

    async def close(self):
        await self.llm.close()

    # ------------------------------------------------------------------
    # Per-user agent cache
    # ------------------------------------------------------------------

    def get_agent(self, user_id: str) -> Agent:
        with self._agents_lock:
            agent = self._agents.get(user_id)
            if agent is None:
                agent = Agent(
                    llm=self.llm,
                    user_id=user_id,
                    db=self.db,
                    window_size=self.window_size,
                    system_prompt=self.system_prompt,
                )
                self._agents[user_id] = agent
            return agent

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, str]]:
        return self.llm.list_mcp_tools()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_id: str,
        query: str,
        enabled_tools: Optional[list[str]] = None,
    ) -> str:
        agent = self.get_agent(user_id)
        return await agent.get_llm_response(
            query,
            enabled_tools=enabled_tools,
        )
