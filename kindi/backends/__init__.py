# SPDX-License-Identifier: AGPL-3.0-only
"""Backend plugin interface (T4.1).

Importing this package registers the fake backend. ``resolve`` serves the fake
backend and any externally registered backend, and says so when asked for
anything else.
"""
import os

from .base import Backend, BackendCapabilities, CancelToken, GenerationEvent, SamplingParams
from .registry import (
    BackendNotFound,
    BackendUnavailable,
    backend_classes,
    get_registration,
    list_backends,
    load_external_backends,
    register_backend,
    resolve,
    unregister_backend,
)
from .fake import FakeBackend
from .dsv41 import DeepseekV41Backend

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BackendNotFound",
    "BackendUnavailable",
    "CancelToken",
    "DeepseekV41Backend",
    "FakeBackend",
    "GenerationEvent",
    "SamplingParams",
    "backend_classes",
    "get_registration",
    "list_backends",
    "load_external_backends",
    "register_backend",
    "resolve",
    "unregister_backend",
]


def load_configured_backends():
    """Load KINDI_BACKEND_MODULES once. Safe to call repeatedly."""
    spec = os.environ.get("KINDI_BACKEND_MODULES", "")
    if not spec:
        return []
    return load_external_backends(spec)
