# agent.py

import asyncio
import logging
import os
from typing import Any, Optional

from gemini_core import GeminiCore
from context import ContextBuilder, Conversation
from google.genai import types
from google.genai.types import FunctionDeclaration

logger = logging.getLogger(__name__)

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
    """Rolling-window size in number of messages."""
    try:
        return max(1, int(os.getenv("CHAT_WINDOW_SIZE", "20")))
    except ValueError:
        return 20


class Agent:
    """
    High-level AI agent, per user.
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
            self.llm = GeminiCore(mcp_config_path=mcp_config, model=model)
            self._owns_llm = True
        else:
            self.llm = llm
            self._owns_llm = False

        self.user_id = user_id
        self.db = db
        self.window_size = window_size if window_size is not None else _default_window_size()

        self.conversation = Conversation()
        self.context = ContextBuilder(
            system_prompt=system_prompt,
            conversation=self.conversation,
        )

        # Per‑agent lock to serialise requests for the same user
        self._request_lock = asyncio.Lock()

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
        return self.llm.list_mcp_tools()

    # ------------------------------------------------------------------
    # Tool filtering (unchanged, but now uses `decl.name`)
    # ------------------------------------------------------------------

    @classmethod
    def _filter_tools(
        cls,
        llm: GeminiCore,
        enabled_names: Optional[list[str]],
    ) -> Optional[list[types.Tool]]:
        if enabled_names is None:
            return None
        if not enabled_names:
            return []

        wanted = {str(name) for name in enabled_names}
        matched = set()

        filtered: list[types.Tool] = []
        for tool in llm.gemini_tools:
            declarations = tool.function_declarations or []
            kept = []
            for decl in declarations:
                # decl is now a FunctionDeclaration object
                name = decl.name
                if name is not None and name in wanted:
                    kept.append(decl)
                    matched.add(name)
            if kept:
                filtered.append(types.Tool(function_declarations=kept))

        unknown = wanted - matched
        if unknown:
            logger.warning("[Agent] Unknown tool names requested: %s", sorted(unknown))
        return filtered

    # ------------------------------------------------------------------
    # Persistent rolling window + memories
    # ------------------------------------------------------------------

    async def _load_context(self) -> tuple[list[Any], list[str]]:
        """
        Load recent messages and memories from DB.
        Returns (messages, memories).
        """
        if self.db is None or not self.user_id:
            # No persistence – use in‑memory conversation, empty memories
            return self.conversation.messages, []

        # Load window
        recent = await self.db.get_recent_messages(self.user_id, self.window_size)
        self.conversation.messages.clear()
        self.conversation.messages.extend(recent)

        # Load memories
        memories = await self.db.get_memories(self.user_id)
        return self.conversation.messages, memories

    # ------------------------------------------------------------------
    # Memory management (persistent)
    # ------------------------------------------------------------------

    async def remember(self, memory: str) -> None:
        if self.db and self.user_id:
            await self.db.add_memory(self.user_id, memory)
        else:
            logger.warning("No DB configured – memory not persisted.")

    async def forget(self, memory: str) -> None:
        if self.db and self.user_id:
            await self.db.delete_memory(self.user_id, memory)

    async def get_memories(self) -> list[str]:
        if self.db and self.user_id:
            return await self.db.get_memories(self.user_id)
        return []

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    async def get_llm_response(
        self,
        query: str,
        enabled_tools: Optional[list[str]] = None,
    ) -> str:
        """
        Main agent function – serialised per user.
        """
        async with self._request_lock:
            # 1. Load context (messages + memories) from DB
            _, memories = await self._load_context()

            # 2. Build prompt
            prompt = self.context.build(query, memories=memories)

            # 3. Ask Gemini with optional tool filter
            tools_to_use = self._filter_tools(self.llm, enabled_tools)
            response = await self.llm.get_response(prompt, tools=tools_to_use)

            # 4. Persist both sides (if persistent)
            if self.db is not None and self.user_id:
                await self.db.add_message(self.user_id, "user", query, enabled_tools)
                await self.db.add_message(self.user_id, "assistant", response, enabled_tools)

            # 5. Update in‑memory conversation for subsequent turns within same request
            self.conversation.add_user(query)
            self.conversation.add_assistant(response)

            return response

    # ------------------------------------------------------------------
    # Conversation (in‑memory)
    # ------------------------------------------------------------------

    def clear_conversation(self):
        self.conversation.clear()

    def get_conversation(self):
        return self.conversation.messages


class AgentManager:
    """
    Manages one shared `GeminiCore` (and its MCP connections)
    across many per‑user `Agent` instances.
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
        self.llm = GeminiCore(mcp_config_path=mcp_config, model=model)
        self.system_prompt = system_prompt
        self.db = db
        self.window_size = window_size if window_size is not None else _default_window_size()

        self._agents_lock = asyncio.Lock()
        self._agents: dict[str, Agent] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        await self.llm.start()

    async def close(self):
        await self.llm.close()

    # ------------------------------------------------------------------
    # Per‑user agent cache
    # ------------------------------------------------------------------

    async def get_agent(self, user_id: str) -> Agent:
        async with self._agents_lock:
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

    def get_mcp_status(self) -> list[dict[str, str]]:
        return self.llm.get_mcp_status()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_id: str,
        query: str,
        enabled_tools: Optional[list[str]] = None,
    ) -> str:
        agent = await self.get_agent(user_id)
        return await agent.get_llm_response(query, enabled_tools=enabled_tools)