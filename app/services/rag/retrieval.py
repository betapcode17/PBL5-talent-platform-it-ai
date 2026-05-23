"""Compatibility shim for legacy RAG retrieval imports.

The actual implementation now lives in `app.services.rag.core.retrieval`,
but several tests and call sites still import from the older module path.
"""

from __future__ import annotations

from .core.retrieval import *  # noqa: F401,F403
