"""Resolve 'auto' device string to a concrete torch device."""
from __future__ import annotations

from config import MODEL


def resolve_device() -> str:
    if MODEL.device != "auto":
        return MODEL.device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
