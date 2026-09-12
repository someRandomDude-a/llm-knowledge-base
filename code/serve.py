# serve.py
import logging
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8005"))
    workers = 1
    reload = os.getenv("UVICORN_RELOAD", "0") == "1"
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logging.info("Starting uvicorn on %s:%s (workers=%s, reload=%s)", host, port, workers, reload)
    if reload:
        uvicorn.run("api:app", host=host, port=port, reload=True, log_level=log_level)
    else:
        uvicorn.run("api:app", host=host, port=port, workers=workers, log_level=log_level)

if __name__ == "__main__":
    main()