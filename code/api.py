# api.py
"""
FastAPI app for the LLM knowledge-base agent.

Endpoints
---------
    GET  /health      -> liveness probe
    GET  /tools       -> JSON list of available MCP tools
    POST /chat        -> one LLM turn, with persistent per-user
                         context backed by PostgreSQL

The whole thing is async-native now. The old `AsyncRunner` bridge
between sync Flask and async MCP/Gemini is gone — FastAPI handlers
are coroutines, so we just `await manager.chat(...)` directly.

Run locally with:
    uvicorn api:app --reload

Or for production:
    uvicorn api:app --host 0.0.0.0 --port 8005 --workers 2
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import AgentManager
from db import ChatHistoryDB


logger = logging.getLogger(__name__)


# ======================================================================
# Request / response models
# ======================================================================


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    query: str = Field(
        ...,
        min_length=1,
        description="The user's prompt. Must be a non-empty string.",
    )
    user: str = Field(
        ...,
        min_length=1,
        description=(
            "Opaque user identifier used as the chat-history key. "
            "Pick something stable per user (e.g. a hash on the client)."
        ),
    )
    tools: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional allow-list of tool names for this turn. "
            "Omitted / null = all tools. [] = no tools. "
            "Otherwise = restrict to the named tools; unknown names "
            "are logged and dropped."
        ),
    )


class ChatResponse(BaseModel):
    """Body returned by POST /chat."""

    user: str
    response: str


class HealthResponse(BaseModel):
    status: str
    window_size: int
    users_cached: int


class ToolInfo(BaseModel):
    name: str
    description: str


class ToolsResponse(BaseModel):
    tools: list[ToolInfo]


class ErrorResponse(BaseModel):
    error: str


# ======================================================================
# Lifespan: start/stop the AgentManager once per worker process
# ======================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the AgentManager (and the LLM + MCP sessions) on boot,
    shut it down cleanly on exit.

    Each uvicorn worker gets its own lifespan, so each worker owns
    its own AgentManager and its own MCP connections. This matches
    the design in agent.py: one GeminiCore is shared by many
    per-user Agent instances *inside one process*.
    """
    db = ChatHistoryDB()
    manager = AgentManager(db=db)

    try:
        await manager.start()
    except Exception:
        logger.exception("Failed to start AgentManager")
        db.close()
        raise

    # Stash on app.state so route handlers can reach them.
    app.state.db = db
    app.state.manager = manager

    logger.info(
        "AgentManager started (window_size=%d)",
        manager.window_size,
    )

    try:
        yield
    finally:
        logger.info("Shutting down AgentManager")
        try:
            await manager.close()
        except Exception:
            logger.exception("Error closing manager")
        try:
            db.close()
        except Exception:
            logger.exception("Error closing DB pool")


# ======================================================================
# App factory
# ======================================================================


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Knowledge Base",
        version="2.0.0",
        description=(
            "Gemini-powered agent that bridges to MCP servers, "
            "with persistent per-user chat history."
        ),
        lifespan=lifespan,
    )

    # Permissive CORS so a browser frontend can hit the API.
    # Tighten allow_origins in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------
    # Routes
    # ----------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        manager: AgentManager = app.state.manager
        return HealthResponse(
            status="ok",
            window_size=manager.window_size,
            users_cached=len(manager._agents),
        )

    @app.get("/tools", response_model=ToolsResponse)
    async def list_tools() -> ToolsResponse:
        """
        Return every available tool (across all MCP servers)
        in the format the frontend expects.
        """
        manager: AgentManager = app.state.manager
        return ToolsResponse(
            tools=[
                ToolInfo(name=t["name"], description=t["description"])
                for t in manager.list_tools()
            ]
        )

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            422: {"model": ErrorResponse},  # pydantic validation
            500: {"model": ErrorResponse},  # unhandled exception
        },
    )
    async def chat(payload: ChatRequest) -> ChatResponse:
        """
        Run one LLM turn for a user.

        Validation is handled by Pydantic:
            - `query` and `user` are required and non-empty
            - `tools`, if present, must be a list of strings
        """
        manager: AgentManager = app.state.manager

        try:
            response_text = await manager.chat(
                user_id=payload.user,
                query=payload.query,
                enabled_tools=payload.tools,
            )
        except Exception as exc:
            logger.exception("Chat failed for user=%s", payload.user)
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            )

        return ChatResponse(user=payload.user, response=response_text)

    return app


# Module-level app for `uvicorn api:app`
app = create_app()


# Allow `python api.py` for local dev.
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8005")),
        reload=os.getenv("UVICORN_RELOAD", "0") == "1",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
