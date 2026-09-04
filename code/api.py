# api.py
"""
Flask API for the LLM knowledge-base agent.

Endpoints
---------
    GET  /health      -> liveness probe
    GET  /tools       -> JSON list of available MCP tools
    POST /chat        -> one LLM turn, with persistent per-user
                         context backed by PostgreSQL

The agent itself is async (it talks to Gemini and MCP).
Flask handlers are sync. We bridge the two with a single
background asyncio loop running in a daemon thread
(`AsyncRunner` below). One loop per worker process is enough -
`asyncio.run_coroutine_threadsafe` is thread-safe.
"""

import asyncio
import atexit
import logging
import os
import threading
from typing import Any, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from agent import AgentManager
from db import ChatHistoryDB
from gemini import GeminiCore


logger = logging.getLogger(__name__)


# ======================================================================
# Async -> sync bridge
# ======================================================================

class AsyncRunner:
    """
    Runs an asyncio event loop in a background thread so that
    sync Flask code can `await` async work.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start()

    def _start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="AsyncRunner",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=10)
        if self._loop is None:
            raise RuntimeError("AsyncRunner failed to start")

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    def run(self, coro: Any) -> Any:
        """
        Submit a coroutine and block until it completes.
        """
        if self._loop is None:
            raise RuntimeError("AsyncRunner is not running")
        future = asyncio.run_coroutine_threadsafe(
            coro,
            self._loop,
        )
        return future.result()

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)


# ======================================================================
# Process-wide state (so atexit can clean up)
# ======================================================================

_state: dict[str, Any] = {
    "manager": None,
    "db": None,
    "runner": None,
}


def _cleanup() -> None:
    manager = _state.get("manager")
    runner = _state.get("runner")
    db = _state.get("db")

    if manager is not None and runner is not None:
        try:
            runner.run(manager.close())
        except Exception:
            logger.exception("Error closing manager")

    if db is not None:
        try:
            db.close()
        except Exception:
            logger.exception("Error closing DB pool")

    if runner is not None:
        try:
            runner.stop()
        except Exception:
            pass


atexit.register(_cleanup)


# ======================================================================
# App factory
# ======================================================================

def create_app(
    *,
    manager: Optional[AgentManager] = None,
    db: Optional[ChatHistoryDB] = None,
    runner: Optional[AsyncRunner] = None,
) -> Flask:
    """
    Build the Flask app. If `manager` / `db` / `runner` are
    not provided, sensible defaults are created and the LLM is
    started in the background.
    """
    app = Flask(__name__)

    # Permissive CORS so a browser frontend can hit the API.
    # Tighten this in production.
    CORS(app)

    # ----------------------------------------------------------
    # Resolve shared resources.
    # ----------------------------------------------------------

    if db is None:
        db = ChatHistoryDB()

    if runner is None:
        runner = AsyncRunner()

    if manager is None:
        manager = AgentManager(db=db)

    # Start the LLM (loads MCP, talks to Gemini) inside the loop.
    # If it fails, surface the error to the caller.
    try:
        runner.run(manager.start())
    except Exception:
        logger.exception("Failed to start AgentManager")
        raise

    _state["manager"] = manager
    _state["db"] = db
    _state["runner"] = runner

    # ----------------------------------------------------------
    # Routes
    # ----------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "window_size": manager.window_size,
                "users_cached": len(manager._agents),
            }
        )

    @app.route("/tools", methods=["GET"])
    def list_tools():
        """
        Return every available tool (across all MCP servers)
        in the format the frontend expects:

            {
              "tools": [
                {"name": "...", "description": "..."},
                ...
              ]
            }
        """
        return jsonify({"tools": manager.list_tools()})

    @app.route("/chat", methods=["POST"])
    def chat():
        """
        Run one LLM turn for a user.

        Request body:
            {
              "query": "user prompt",
              "tools": ["tool_a", "tool_b"],   # optional
              "user":  "userID hash"           # required
            }

        Response body:
            {
              "user":     "userID hash",
              "response": "assistant text"
            }
        """
        payload = request.get_json(silent=True) or {}

        query = payload.get("query")
        user = payload.get("user")
        tools = payload.get("tools")

        # ---- input validation ----
        if not isinstance(query, str) or not query.strip():
            return (
                jsonify(
                    {"error": "'query' is required and must be a non-empty string"}
                ),
                400,
            )
        if not isinstance(user, str) or not user.strip():
            return (
                jsonify(
                    {"error": "'user' is required and must be a non-empty string"}
                ),
                400,
            )
        if tools is not None:
            if not isinstance(tools, list) or not all(
                isinstance(t, str) for t in tools
            ):
                return (
                    jsonify(
                        {"error": "'tools' must be a list of strings"}
                    ),
                    400,
                )

        try:
            response_text = runner.run(
                manager.chat(
                    user_id=user,
                    query=query,
                    enabled_tools=tools,
                )
            )
        except Exception as exc:
            logger.exception("Chat failed for user=%s", user)
            return (
                jsonify(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                500,
            )

        return jsonify(
            {
                "user": user,
                "response": response_text,
            }
        )

    return app


# Allow `python api.py` for local dev.
if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    application = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"

    # `threaded=True` so concurrent requests can both pump
    # the background asyncio loop. For real production use
    # gunicorn / uwsgi with multiple worker processes.
    application.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
    )
