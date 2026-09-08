# api.py
"""
FastAPI app for the LLM knowledge-base agent.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from agent import AgentManager
from db import ChatHistoryDB

logger = logging.getLogger(__name__)

# ─── Security ─────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN")
if INTERNAL_API_TOKEN is not None:
    INTERNAL_API_TOKEN = INTERNAL_API_TOKEN.strip()
    logger.info("Internal API token is set (length: %d)", len(INTERNAL_API_TOKEN))
else:
    logger.info("Internal API token is not set – authentication disabled")

async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> None:
    if INTERNAL_API_TOKEN is None:
        return

    if credentials is None:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        logger.warning("Invalid auth scheme: %s", credentials.scheme)
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme. Use Bearer.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_token = credentials.credentials.strip()
    if provided_token != INTERNAL_API_TOKEN:
        logger.warning(
            "Token mismatch. Provided: %s..., Expected: %s...",
            provided_token[:10],
            INTERNAL_API_TOKEN[:10]
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ─── Models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user: str = Field(..., min_length=1)
    tools: Optional[list[str]] = Field(default=None)


class ChatResponse(BaseModel):
    user: str
    response: str


class HealthResponse(BaseModel):
    status: str
    window_size: int
    users_cached: int
    mcp_servers: list[dict[str, str]]


class ToolInfo(BaseModel):
    name: str
    description: str


class ToolsResponse(BaseModel):
    tools: list[ToolInfo]


class ErrorResponse(BaseModel):
    error: str


# ======================================================================
# Lifespan
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    db = ChatHistoryDB()
    await db.init()
    manager = AgentManager(db=db)

    try:
        await manager.start()
    except Exception:
        logger.exception("Failed to start AgentManager")
        await db.close()
        raise

    app.state.db = db
    app.state.manager = manager

    logger.info("AgentManager started (window_size=%d)", manager.window_size)

    try:
        yield
    finally:
        logger.info("Shutting down AgentManager")
        try:
            await manager.close()
        except Exception:
            logger.exception("Error closing manager")
        try:
            await db.close()
        except Exception:
            logger.exception("Error closing DB pool")


# ======================================================================
# App factory
# ======================================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Knowledge Base",
        version="2.0.0",
        description="Gemini‑powered agent with MCP and persistent chat.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes (all protected by verify_token) ──
    @app.get("/health", response_model=HealthResponse, dependencies=[Depends(verify_token)])
    async def health() -> HealthResponse:
        manager: AgentManager = app.state.manager
        return HealthResponse(
            status="ok",
            window_size=manager.window_size,
            users_cached=len(manager._agents),
            mcp_servers=manager.get_mcp_status(),
        )

    @app.get("/tools", response_model=ToolsResponse, dependencies=[Depends(verify_token)])
    async def list_tools() -> ToolsResponse:
        manager: AgentManager = app.state.manager
        return ToolsResponse(
            tools=[ToolInfo(name=t["name"], description=t["description"])
                   for t in manager.list_tools()]
        )

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
    async def chat(payload: ChatRequest) -> ChatResponse:
        manager: AgentManager = app.state.manager
        try:
            response_text = await manager.chat(
                user_id=payload.user,
                query=payload.query,
                enabled_tools=payload.tools,
            )
        except Exception as exc:
            logger.exception("Chat failed for user=%s", payload.user)
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
        return ChatResponse(user=payload.user, response=response_text)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    uvicorn.run("api:app", host="0.0.0.0", port=8005, reload=True)