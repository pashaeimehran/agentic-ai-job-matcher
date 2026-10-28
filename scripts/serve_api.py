"""Run the REST API locally.

    python -m scripts.serve_api            # http://127.0.0.1:8000/docs
    python -m scripts.serve_api --port 8080

Uses the synthetic CV and fixture catalog. /chat is enabled when OPENAI_API_KEY
is set (paid calls); every other endpoint is deterministic and offline.
"""
from __future__ import annotations

import argparse

import uvicorn

from src.api import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the Job Match Agent REST API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
