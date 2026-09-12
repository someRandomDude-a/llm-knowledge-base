# agent.py
"""
Agent orchestration layer.

Responsible for:
    - Owning the shared GeminiCore (MCP connections + Gemini client)
    - Caching per-user Agent instances
    - Loading / persisting chat history and memories via db.py
    - Building prompts via context.py
    - Deciding which tools to expose to Gemini on each turn

Deliberately does NOT contain:
    - Gemini API specifics           (gemini_core.py)
    - MCP protocol / transport       (gemini_core.py)
    - Prompt formatting              (context.py)
    - SQL / database details         (db.py)
    - HTTP routing                   (api.py)
"""

import asyncio
import logging
import os
from typing import Any, Optional

from google.genai import types
from google.genai.types import Tool

from context import ContextBuilder, Conversation, Message
from gemini_core import GeminiCore

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
    try:
        return max(1, int(os.getenv("CHAT_WINDOW_SIZE", "20")))
    except ValueError:
        return 20


class Agent:
    """
    Per-user agent. Shares an externally-owned GeminiCore with every
    other Agent, but keeps its own Conversation / ContextBuilder.
    """

    def __init__(
        self,
        llm: GeminiCore,
        *,
        user_id: Optional[str] = None,
        db: Optional[Any] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        window_size: Optional[int] = None,
    ):
        self.llm = llm
        self.user_id = user_id
        self.db = db
        self.window_size = (
            window_size if window_size is not None else _default_window_size()
        )

        self.conversation = Conversation(window_size=self.window_size)
        self.context = ContextBuilder(
            system_prompt=system_prompt,
            conversation=self.conversation,
        )

        self._request_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Discovery (delegates to GeminiCore)
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, str]]:
        return self.llm.list_mcp_tools()

    def get_mcp_status(self) -> list[dict[str, str]]:
        return self.llm.get_mcp_status()

    # ------------------------------------------------------------------
    # Context + persistence
    # ------------------------------------------------------------------

    async def _load_context(self) -> list[str]:
        """
        Refresh the rolling window from persistence (if any) and return
        the current memories for this user.
        """
        if self.db is None or not self.user_id:
            return []

        recent = await self.db.get_recent_messages(self.user_id, self.window_size)
        self.conversation.extend(recent)

        return await self.db.get_memories(self.user_id)

    async def _persist_turn(
        self,
        query: str,
        response: str,
        enabled_tools: Optional[list[str]],
    ) -> None:
        if self.db is None or not self.user_id:
            return
        await self.db.add_messages(
            self.user_id,
            [
                ("user", query, enabled_tools),
                ("assistant", response, enabled_tools),
            ],
        )
    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    async def remember(self, memory: str) -> None:
        if self.db and self.user_id:
            await self.db.add_memory(self.user_id, memory)
        else:
            logger.warning("No DB configured - memory not persisted.")

    async def forget(self, memory: str) -> None:
        if self.db and self.user_id:
            await self.db.delete_memory(self.user_id, memory)

    async def get_memories(self) -> list[str]:
        if self.db and self.user_id:
            return await self.db.get_memories(self.user_id)
        return []

    # ------------------------------------------------------------------
    # Tool filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_tools(
        llm: GeminiCore,
        enabled_names: Optional[list[str]],
    ) -> Optional[list[Tool]]:
        if enabled_names is None:
            return None
        if not enabled_names:
            return []

        wanted = {str(name) for name in enabled_names}
        matched: set[str] = set()
        filtered: list[Tool] = []

        for tool in llm.gemini_tools:
            declarations = tool.function_declarations or []
            kept = []
            for decl in declarations:
                name = decl.name
                if name is not None and name in wanted:
                    kept.append(decl)
                    matched.add(name)
            if kept:
                filtered.append(Tool(function_declarations=kept))

        unknown = wanted - matched
        if unknown:
            logger.warning("[Agent] Unknown tool names requested: %s", sorted(unknown))
        return filtered

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    async def get_llm_response(
        self,
        query: str,
        enabled_tools: Optional[list[str]] = None,
    ) -> str:
        async with self._request_lock:
            memories = await self._load_context()
            prompt = self.context.build(query, memories=memories)

            tools_to_use = self._filter_tools(self.llm, enabled_tools)
            response = await self.llm.get_response(prompt, tools=tools_to_use)

            await self._persist_turn(query, response, enabled_tools)

            # Keep the in-memory window fresh so subsequent turns
            # within this process do not depend on a DB round-trip.
            self.conversation.add_user(query)
            self.conversation.add_assistant(response)

            return response

    # ------------------------------------------------------------------
    # In-memory conversation
    # ------------------------------------------------------------------

    def clear_conversation(self) -> None:
        self.conversation.clear()

    def get_conversation(self) -> list:
        return self.conversation.recent()


class AgentManager:
    """
    Owns a single shared GeminiCore (one set of MCP sessions, one
    Gemini client) and lazily creates per-user Agents on demand.
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
        self.window_size = (
            window_size if window_size is not None else _default_window_size()
        )

        self._agents_lock = asyncio.Lock()
        self._agents: dict[str, Agent] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.llm.start()

    async def close(self) -> None:
        await self.llm.close()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def users_cached(self) -> int:
        return len(self._agents)

    def list_tools(self) -> list[dict[str, str]]:
        return self.llm.list_mcp_tools()

    def get_mcp_status(self) -> list[dict[str, str]]:
        return self.llm.get_mcp_status()

    # ------------------------------------------------------------------
    # Per-user agent cache
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

    async def get_history(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> tuple[list[Message], list[str]]:
        """
        Read recent chat messages + memories for a user from persistence.
        Pure read: does not touch the per-user Agent cache.
        """
        if self.db is None:
            return [], []

        n = limit if limit is not None else self.window_size
        messages = await self.db.get_recent_messages(user_id, n)
        memories = await self.db.get_memories(user_id)
        return messages, memories