# context.py
"""
Conversation state, rolling window, and prompt construction.

Responsible for:
    - Representing chat messages (Message)
    - Holding the in-memory rolling window (Conversation)
    - Rendering a prompt string from system prompt + memories +
      recent messages + current query (ContextBuilder)

Deliberately does NOT know about:
    - Gemini or MCP
    - The database
    - Users or sessions
"""

from dataclasses import dataclass
from typing import Literal, Optional


Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


class Conversation:
    """
    In-memory rolling window of messages.

    The window is enforced on every write: once the buffer is full,
    the oldest messages are dropped. This makes the rolling window
    a property of the conversation itself, rather than something
    the caller has to remember to apply.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = max(1, int(window_size))
        self.messages: list[Message] = []

    # --- writes ---

    def add_user(self, content: str) -> None:
        self._append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._append(Message(role="assistant", content=content))

    def extend(self, messages: list[Message]) -> None:
        """
        Replace the buffer with the given messages, then apply the
        rolling window. Used when rehydrating from persistence.
        """
        self.messages = list(messages)
        self._trim()

    def clear(self) -> None:
        self.messages.clear()

    # --- reads ---

    def recent(self, count: Optional[int] = None) -> list[Message]:
        if count is None:
            return list(self.messages)
        return list(self.messages[-count:])

    def __len__(self) -> int:
        return len(self.messages)

    # --- internals ---

    def _append(self, message: Message) -> None:
        self.messages.append(message)
        self._trim()

    def _trim(self) -> None:
        if len(self.messages) > self.window_size:
            del self.messages[: len(self.messages) - self.window_size]


class ContextBuilder:
    """
    Renders a prompt string from the system prompt, long-term memories,
    the rolling conversation window, and the current user query.

    Format-only: no Gemini types, no MCP types.
    """

    def __init__(
        self,
        system_prompt: str,
        conversation: Conversation,
    ):
        self.system_prompt = system_prompt
        self.conversation = conversation

    def build(
        self,
        query: str,
        memories: Optional[list[str]] = None,
    ) -> str:
        sections: list[str] = []

        sections.append(
            "SYSTEM INSTRUCTIONS\n"
            "===================\n"
            f"{self.system_prompt}"
        )

        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            sections.append(
                "LONG-TERM MEMORY\n"
                "================\n"
                f"{memory_text}"
            )

        messages = self.conversation.recent()
        if messages:
            lines = []
            for msg in messages:
                role = "USER" if msg.role == "user" else "ASSISTANT"
                lines.append(f"{role}: {msg.content}")
            sections.append(
                "RECENT CONVERSATION\n"
                "===================\n" + "\n".join(lines)
            )

        sections.append(
            "CURRENT USER MESSAGE\n"
            "====================\n"
            f"{query}"
        )

        return "\n\n".join(sections)