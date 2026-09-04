# serve.py
"""
Entry point for running the FastAPI app under uvicorn.

Example:
    DATABASE_URL=postgresql://user:pass@localhost:5432/llm \
    GEMINI_API_KEY=... \
    CHAT_WINDOW_SIZE=20 \
    WORKERS=2 \
    python serve.py

For development with auto-reload, set UVICORN_RELOAD=1.
Note: reload mode spawns a subprocess and ignores WORKERS.
"""
import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8005"))
    workers = int(os.getenv("WORKERS", "2"))
    reload = os.getenv("UVICORN_RELOAD", "0") == "1"
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logging.info(
        "Starting uvicorn on %s:%s (workers=%s, reload=%s)",
        host,
        port,
        workers,
        reload,
    )

    if reload:
        # uvicorn manages its own subprocess here; --workers is ignored.
        uvicorn.run(
            "api:app",
            host=host,
            port=port,
            reload=True,
            log_level=log_level,
        )
    else:
        uvicorn.run(
            "api:app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
        )


if __name__ == "__main__":
    main()
