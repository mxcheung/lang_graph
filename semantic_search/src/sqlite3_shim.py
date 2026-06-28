"""Side-effect module: swap stdlib sqlite3 for pysqlite3-binary when available.

ChromaDB needs SQLite >= 3.35. Many Linux servers ship an older sqlite, so swap in
pysqlite3-binary when available. On modern macOS the stdlib sqlite3 is already new
enough (and pysqlite3-binary may have no Apple Silicon wheel) — fall back silently.

Import this module BEFORE chromadb so the swap takes effect.
"""
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
