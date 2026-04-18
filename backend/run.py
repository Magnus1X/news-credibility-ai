"""
run.py — starts both Milestone 1 (port 8000) and Milestone 2 (port 8001)
in a single terminal.

Usage:
    python3 run.py
"""
import asyncio
import os
import signal
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import uvicorn


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _kill_port(port: int):
    """Kill whatever is using the port (macOS/Linux)."""
    os.system(f"lsof -ti:{port} | xargs kill -9 2>/dev/null")


async def main():
    for port in (8000, 8001):
        if not _port_free(port):
            print(f"⚠️  Port {port} in use — killing existing process...")
            _kill_port(port)
            await asyncio.sleep(1)

    config_m1 = uvicorn.Config(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="warning",
    )
    config_m2 = uvicorn.Config(
        "agent_app:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
        log_level="warning",
    )

    server_m1 = uvicorn.Server(config_m1)
    server_m2 = uvicorn.Server(config_m2)

    print("🚀 Both servers starting...")
    print("   Milestone 1 → http://127.0.0.1:8000  (POST /predict)")
    print("   Milestone 2 → http://127.0.0.1:8001  (POST /analyze, POST /analyze/pdf)")
    print("   Frontend    → http://localhost:5173")
    print("   Press Ctrl+C to stop.\n")

    await asyncio.gather(
        server_m1.serve(),
        server_m2.serve(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Both servers stopped.")
