# db.py
"""
PostgreSQL-backed persistent chat history.

Schema
------
    user_chat_history
        id              BIGSERIAL PRIMARY KEY
        user_id         VARCHAR(255)   -- the userID hash from the client
        role            VARCHAR(20)    -- 'user' | 'assistant'
        content         TEXT           -- the message text
        enabled_tools   JSONB          -- tool list active for that turn
        created_at      TIMESTAMPTZ    -- wall clock, default now()

Rolling window
--------------
The window is a *read* against the most recent N rows for a user.
Past interactions are never deleted. The window size is controlled
by the CHAT_WINDOW_SIZE environment variable.
"""

import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from context import Message


# ======================================================================
# Connection string discovery
# ======================================================================

def get_connection_string() -> str:
    """
    Read the PostgreSQL connection string from the environment.

    Accepted variables (in order):
        DATABASE_URL
        POSTGRES_CONNECTION_STRING
        POSTGRES_URL
    """
    for var in (
        "DATABASE_URL",
        "POSTGRES_CONNECTION_STRING",
        "POSTGRES_URL",
    ):
        value = os.getenv(var)
        if value:
            return value

    raise RuntimeError(
        "PostgreSQL connection string is not set. "
        "Set the DATABASE_URL environment variable, e.g.\n"
        "  DATABASE_URL=postgresql://user:password@host:5432/dbname"
    )


# ======================================================================
# ChatHistoryDB
# ======================================================================

class ChatHistoryDB:
    """
    Thread-safe, pooled access to the `user_chat_history` table.

    The pool is created on construction. The schema is created on
    first use (idempotent `CREATE TABLE IF NOT EXISTS`).
    """

    _init_lock = threading.Lock()

    def __init__(
        self,
        connection_string: Optional[str] = None,
        *,
        min_conn: int = 1,
        max_conn: int = 10,
        table_name: str = "user_chat_history",
    ):
        self.connection_string = (
            connection_string or get_connection_string()
        )
        self.table_name = table_name

        self.pool = ThreadedConnectionPool(
            min_conn,
            max_conn,
            dsn=self.connection_string,
        )

        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._init_lock:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id              BIGSERIAL PRIMARY KEY,
                            user_id         VARCHAR(255) NOT NULL,
                            role            VARCHAR(20)  NOT NULL
                                CHECK (role IN ('user', 'assistant')),
                            content         TEXT         NOT NULL,
                            enabled_tools   JSONB,
                            created_at      TIMESTAMPTZ  NOT NULL
                                DEFAULT NOW()
                        )
                        """
                    )
                    # DESC index supports "last N for this user"
                    # without a sort step.
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS
                            idx_{self.table_name}_user_id_id
                        ON {self.table_name} (user_id, id DESC)
                        """
                    )
                conn.commit()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        enabled_tools: Optional[list[str]] = None,
    ) -> int:
        """
        Insert a message. Returns the new row id.
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name}
                        (user_id, role, content, enabled_tools)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        user_id,
                        role,
                        content,
                        Json(enabled_tools)
                        if enabled_tools is not None
                        else None,
                    ),
                )
                msg_id = cur.fetchone()[0]
            conn.commit()
        return int(msg_id)

    def get_recent_messages(
        self,
        user_id: str,
        limit: int,
    ) -> list[Message]:
        """
        Return the most recent `limit` messages for `user_id`,
        in chronological order (oldest first).

        The DB is never modified. This is purely a read.
        """
        if limit <= 0:
            return []

        with self._connection() as conn:
            with conn.cursor(
                cursor_factory=RealDictCursor,
            ) as cur:
                cur.execute(
                    f"""
                    SELECT role, content
                    FROM (
                        SELECT id, role, content
                        FROM {self.table_name}
                        WHERE user_id = %s
                        ORDER BY id DESC
                        LIMIT %s
                    ) recent
                    ORDER BY id ASC
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()

        return [
            Message(
                role=row["role"],
                content=row["content"],
            )
            for row in rows
        ]

    def count_messages(
        self,
        user_id: str,
    ) -> int:
        """Total rows stored for a user (for diagnostics)."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.table_name} "
                    "WHERE user_id = %s",
                    (user_id,),
                )
                return int(cur.fetchone()[0])

    def close(self) -> None:
        try:
            self.pool.closeall()
        except Exception:
            pass
