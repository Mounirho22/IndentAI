"""
Compatibility shim for the refactored IdentAI repository.

The full application now lives in ``app/main.py``. This module exists so that
all of the externals that historically did ``import main`` or ran
``uvicorn main:app`` continue to work without modification:

  * ``uvicorn main:app``            — the documented dev launcher
  * ``python main.py``              — the original dev command
  * ``agentcore/entrypoint.py``     -> ``import main as app_module``

It re-exports the canonical :mod:`app.main` namespace verbatim. Note that
:mod:`verify` and the demos intentionally import the canonical module
(``app.main``) instead of this shim, because they reassign module globals such
as ``DB_PATH`` / ``MISSION_*_LIMIT`` and monkeypatch ``_complete`` /
``build_agent`` at runtime — that only works when the module object they patch
is the same one the agent code calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Re-export the real application. `app` (the FastAPI instance) must be present
# in this module's namespace for `uvicorn main:app` to find it.
from app.main import *  # noqa: F401, F403  (intentional full re-export)
from app.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)