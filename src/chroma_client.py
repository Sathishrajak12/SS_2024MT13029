"""Shared local Chroma client configuration for multiple memory collections."""
import os
from pathlib import Path


def create_chroma_client(path: Path):
    import chromadb

    if os.name == "nt":
        import ctypes

        runtime = ctypes.windll.msvcrt
        if runtime._getmaxstdio() < 2048 and runtime._setmaxstdio(2048) < 2048:
            raise RuntimeError("Could not configure the Windows file limit for multi-collection Chroma retrieval.")
    return chromadb.PersistentClient(path=str(path))
