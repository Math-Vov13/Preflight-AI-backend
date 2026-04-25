"""Dev server — uvicorn with hot reload, listening on :5001."""
from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    # Make `app.*` importable when invoked from repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent.parent / "app")],
    )
