# serve.py
"""
Entry point for running the Flask API.

Example:
    DATABASE_URL=postgresql://user:pass@localhost:5432/llm \
    GEMINI_API_KEY=... \
    CHAT_WINDOW_SIZE=20 \
    python serve.py
"""
import logging
import os

from api import create_app


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = create_app()

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"

    logging.info("Starting Flask API on %s:%s (debug=%s)", host, port, debug)
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
