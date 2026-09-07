# context.py

from dataclasses import dataclass
from typing import Literal, Optional


Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


class Conversation:
    """
    Stores the current conversation.
    This deliberately does not know anything about Gemini.
    """

    def __init__(self):
        self.messages: list[Message] = []

    def add_user(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str):
        self.messages.append(Message(role="assistant", content=content))

    def clear(self):
        self.messages.clear()

    def recent(self, count: int = 20) -> list[Message]:
        return self.messages[-count:]


class ContextBuilder:
    """
    Converts agent state into the prompt sent to Gemini.
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
        sections = []

        # System
        sections.append(
            "SYSTEM INSTRUCTIONS\n"
            "===================\n"
            f"{self.system_prompt}"
        )

        # Memories (if any)
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            sections.append(
                "LONG-TERM MEMORY\n"
                "================\n"
                f"{memory_text}"
            )

        # Recent conversation
        messages = self.conversation.recent(count=20)
        if messages:
            lines = []
            for msg in messages:
                role = "USER" if msg.role == "user" else "ASSISTANT"
                lines.append(f"{role}: {msg.content}")
            sections.append(
                "RECENT CONVERSATION\n"
                "===================\n" + "\n".join(lines)
            )

        # Current user message
        sections.append(
            "CURRENT USER MESSAGE\n"
            "====================\n"
            f"{query}"
        )

        return "\n\n".join(sections)