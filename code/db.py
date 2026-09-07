# db.py
"""
Async PostgreSQL-backed persistent chat history and memories.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import asyncpg
from asyncpg import Pool, Record

from context import Message


# ======================================================================
# Connection string discovery
# ======================================================================

def get_connection_string() -> str:
    for var in ("DATABASE_URL", "POSTGRES_CONNECTION_STRING", "POSTGRES_URL"):
        value = os.getenv(var)
        if value:
            return value
    raise RuntimeError(
        "PostgreSQL connection string is not set. "
        "Set the DATABASE_URL environment variable, e.g.\n"
        "  DATABASE_URL=postgresql://user:password@host:5432/dbname"
    )


# ======================================================================
# ChatHistoryDB (async)
# ======================================================================

class ChatHistoryDB:
    def __init__(
        self,
        connection_string: Optional[str] = None,
        *,
        min_conn: int = 1,
        max_conn: int = 10,
        table_name: str = "user_chat_history",
        memories_table: str = "user_memories",
    ):
        self.connection_string = connection_string or get_connection_string()
        self.table_name = table_name
        self.memories_table = memories_table
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._pool: Optional[Pool] = None

    async def init(self) -> None:
        self._pool = await asyncpg.create_pool(
            dsn=self.connection_string,
            min_size=self.min_conn,
            max_size=self.max_conn,
        )
        await self._init_schema()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    @asynccontextmanager
    async def _acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        if not self._pool:
            raise RuntimeError("Database not initialised. Call init() first.")
        async with self._pool.acquire() as conn:
            yield conn  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def _init_schema(self) -> None:
        async with self._acquire() as conn:
            # Chat history table – enabled_tools as JSONB
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id              BIGSERIAL PRIMARY KEY,
                    user_id         VARCHAR(255) NOT NULL,
                    role            VARCHAR(20)  NOT NULL
                        CHECK (role IN ('user', 'assistant')),
                    content         TEXT         NOT NULL,
                    enabled_tools   JSONB,
                    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS
                    idx_{self.table_name}_user_id_id
                ON {self.table_name} (user_id, id DESC)
            """)

            # Memories table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.memories_table} (
                    user_id     VARCHAR(255) NOT NULL,
                    memory_text TEXT         NOT NULL,
                    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, memory_text)
                )
            """)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    async def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        enabled_tools: Optional[list[str]] = None,
    ) -> int:
        """
        Insert a message. Returns the new row id.
        """
        # Convert list to JSON string. This works for both JSONB and TEXT columns.
        enabled_tools_json = json.dumps(enabled_tools) if enabled_tools is not None else None

        async with self._acquire() as conn:
            row: Optional[Record] = await conn.fetchrow(
                f"""
                INSERT INTO {self.table_name}
                    (user_id, role, content, enabled_tools)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                user_id,
                role,
                content,
                enabled_tools_json,
            )
            if row is None:
                raise RuntimeError("Failed to insert message; no row returned.")
            return row["id"]

    async def get_recent_messages(
        self,
        user_id: str,
        limit: int,
    ) -> list[Message]:
        if limit <= 0:
            return []

        async with self._acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM {self.table_name}
                    WHERE user_id = $1
                    ORDER BY id DESC
                    LIMIT $2
                ) recent
                ORDER BY id ASC
                """,
                user_id,
                limit,
            )

        return [
            Message(
                role=row["role"],
                content=row["content"],
            )
            for row in rows
        ]

    async def count_messages(self, user_id: str) -> int:
        async with self._acquire() as conn:
            count: Optional[int] = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE user_id = $1",
                user_id,
            )
            return count or 0

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    async def get_memories(self, user_id: str) -> list[str]:
        async with self._acquire() as conn:
            rows = await conn.fetch(
                f"SELECT memory_text FROM {self.memories_table} WHERE user_id = $1",
                user_id,
            )
            return [row["memory_text"] for row in rows]

    async def add_memory(self, user_id: str, memory: str) -> None:
        async with self._acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.memories_table} (user_id, memory_text)
                VALUES ($1, $2)
                ON CONFLICT (user_id, memory_text) DO NOTHING
                """,
                user_id,
                memory,
            )

    async def delete_memory(self, user_id: str, memory: str) -> None:
        async with self._acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.memories_table} WHERE user_id = $1 AND memory_text = $2",
                user_id,
                memory,
            )