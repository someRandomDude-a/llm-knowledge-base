# context.py

from dataclasses import dataclass
from typing import Literal


Role = Literal[
    "user",
    "assistant",
]


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
        self.messages.append(
            Message(
                role="user",
                content=content,
            )
        )

    def add_assistant(self, content: str):
        self.messages.append(
            Message(
                role="assistant",
                content=content,
            )
        )

    def clear(self):
        self.messages.clear()

    def recent(
        self,
        count: int = 20,
    ) -> list[Message]:

        return self.messages[-count:]


class MemoryStore:
    """
    Very simple in-memory long-term memory.

    This will eventually become:
        SQLiteMemoryStore
        VectorMemoryStore
        FileMemoryStore
        etc.
    """

    def __init__(self):
        self.memories: list[str] = []

    def add(self, memory: str):
        if memory not in self.memories:
            self.memories.append(memory)

    def remove(self, memory: str):
        if memory in self.memories:
            self.memories.remove(memory)

    def clear(self):
        self.memories.clear()

    def all(self) -> list[str]:
        return self.memories.copy()


class ContextBuilder:
    """
    Converts agent state into the prompt sent to Gemini.
    """

    def __init__(
        self,
        system_prompt: str,
        conversation: Conversation,
        memory: MemoryStore,
    ):
        self.system_prompt = system_prompt
        self.conversation = conversation
        self.memory = memory

    def build(
        self,
        query: str,
    ) -> str:

        sections = []

        # --------------------------------------------------------------
        # System
        # --------------------------------------------------------------

        sections.append(
            "SYSTEM INSTRUCTIONS\n"
            "===================\n"
            f"{self.system_prompt}"
        )

        # --------------------------------------------------------------
        # Memories
        # --------------------------------------------------------------

        memories = self.memory.all()

        if memories:
            memory_text = "\n".join(
                f"- {memory}"
                for memory in memories
            )

            sections.append(
                "LONG-TERM MEMORY\n"
                "================\n"
                f"{memory_text}"
            )

        # --------------------------------------------------------------
        # Conversation
        # --------------------------------------------------------------

        messages = self.conversation.recent(
            count=20
        )

        if messages:

            conversation_lines = []

            for message in messages:

                role = (
                    "USER"
                    if message.role == "user"
                    else "ASSISTANT"
                )

                conversation_lines.append(
                    f"{role}: {message.content}"
                )

            sections.append(
                "RECENT CONVERSATION\n"
                "===================\n"
                + "\n".join(
                    conversation_lines
                )
            )

        # --------------------------------------------------------------
        # Current message
        # --------------------------------------------------------------

        sections.append(
            "CURRENT USER MESSAGE\n"
            "====================\n"
            f"{query}"
        )

        return "\n\n".join(sections)