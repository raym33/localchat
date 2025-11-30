#!/usr/bin/env python3
"""Run the Dia2 English Conversation server."""

import uvicorn
from backend.config import config

if __name__ == "__main__":
    print("=" * 60)
    print("  Dia2 English Conversation Practice")
    print("=" * 60)
    print(f"\n  Starting server at http://{config.host}:{config.port}")
    print(f"  Model: {config.dia2_model}")
    print(f"  Device: {config.dia2_device}")
    print("\n  Press Ctrl+C to stop\n")
    print("=" * 60)

    uvicorn.run(
        "backend.main:app",
        host=config.host,
        port=config.port,
        reload=True,
        log_level="info",
    )
