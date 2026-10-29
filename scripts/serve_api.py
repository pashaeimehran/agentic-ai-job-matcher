"""Run the REST API.

    python -m scripts.serve_api            # http://127.0.0.1:8000/docs
    python -m scripts.serve_api --port 8080
    python -m scripts.serve_api --host 0.0.0.0   # container

Configuration comes from environment variables / .env (see .env.example).
APP_MODE=fixture (default without OPENAI_API_KEY) needs no credentials and makes
no paid calls; APP_MODE=live uses OpenAI. With DATABASE_URL the schema is
initialized at startup and startup fails clearly if that is impossible.
"""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from dotenv import load_dotenv

from src.api import create_app


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Serve the Job Match Agent REST API")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    # One JSON payload per application event; the logger name tells the layer.
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Single worker: conversation threads and the response cache are process-local.
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=level.lower(), proxy_headers=True,
                forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"), timeout_graceful_shutdown=20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
