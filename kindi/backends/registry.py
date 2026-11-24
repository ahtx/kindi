# SPDX-License-Identifier: AGPL-3.0-only
"""Backend registry: registration, lookup and resolution (T4.1).

Resolution order for ``resolve(model_id_or_gguf_path, device)``:

1. if the argument is an existing ``.gguf`` file, read its
   ``general.architecture`` metadata with the existing GGUF reader and match
   backends that declare that architecture;
2. otherwise match on the model id against each backend's declared model ids;
3. a backend that declares no architectures and no model ids is a wildcard and
   matches only when nothing more specific matched, or when the caller asks for
   it by name.

External backends are loaded through ``KINDI_BACKEND_MODULES`` (a comma-separated
list of importable module names). Importing such a module is expected to call
``register_backend`` as a side effect. Python entry points are deferred:
importlib.metadata does not exist on Python 3.6.
"""
import importlib
import os

from .base import BackendCapabilities

# name -> (factory, BackendCapabilities)
_REGISTRY = {}
_ORDER = []


class BackendNotFound(LookupError):
    """No registered backend can serve the requested model or architecture."""


class BackendUnavailable(RuntimeError):
    """A registered backend exists but cannot run on this host."""


def _capabilities_for(factory):
    caps = getattr(factory, "capabilities", None)
    if caps is None:
        name = getattr(factory, "__name__", None) or str(factory)
        caps = BackendCapabilities(name=name)
    if not isinstance(caps, BackendCapabilities):
        raise TypeError("backend %r capabilities must be a BackendCapabilities" % (factory,))
    return caps


def register_backend(factory, name=None, model_ids=None, architectures=None,
                     devices=None, capabilities=None, replace=False):
    """Register a backend factory (a Backend subclass or a zero-arg callable).

    The name defaults to the factory's capabilities name. Refusing to replace an
    existing registration is deliberate: two backends claiming one name is a bug,
    not an override.
    """
    caps = capabilities if capabilities is not None else _capabilities_for(factory)
    backend_name = (name or caps.name or getattr(factory, "__name__", "backend")).lower()
    if model_ids is not None:
        caps.model_ids = tuple(model_ids)
    if architectures is not None:
        caps.architectures = tuple(architectures)
    if devices is not None:
        caps.devices = tuple(devices)
    if backend_name in _REGISTRY and not replace:
        raise ValueError("backend %r is already registered" % backend_name)
    if backend_name not in _REGISTRY:
        _ORDER.append(backend_name)
    _REGISTRY[backend_name] = (factory, caps)
    return factory


def unregister_backend(name):
    """Remove a registration. Primarily for tests."""
    backend_name = str(name).lower()
    if backend_name not in _REGISTRY:
        raise KeyError("no backend registered as %r" % name)
    del _REGISTRY[backend_name]
    _ORDER.remove(backend_name)


def _clear_registry():
    """Empty the registry. Test support only; production code never calls this."""
    _REGISTRY.clear()
    del _ORDER[:]


def list_backends():
    """All registrations as JSON-able dicts, in registration order."""
    out = []
    for backend_name in _ORDER:
        factory, caps = _REGISTRY[backend_name]
        row = caps.to_dict()
        row["factory"] = getattr(factory, "__name__", str(factory))
        out.append(row)
    return out


def get_registration(name):
    """The (factory, capabilities) pair registered under name."""
    backend_name = str(name).lower()
    if backend_name not in _REGISTRY:
        raise BackendNotFound("no backend registered as %r (registered: %s)"
                              % (name, ", ".join(_ORDER) or "none"))
    return _REGISTRY[backend_name]


def backend_classes():
    """The registered backend classes/factories, for contract tests."""
    return [_REGISTRY[backend_name][0] for backend_name in _ORDER]


def load_external_backends(spec=None):
    """Import the modules named by KINDI_BACKEND_MODULES. Returns what was loaded.

    Each module is expected to register its backend on import. A module that
    fails to import raises BackendUnavailable naming the module, so a typo in the
    environment variable is visible at startup instead of silently serving
    nothing.
    """
    if spec is None:
        spec = os.environ.get("KINDI_BACKEND_MODULES", "")
    loaded = []
    for module_name in [part.strip() for part in str(spec).split(",") if part.strip()]:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise BackendUnavailable("cannot load backend module %r: %s" % (module_name, exc))
        loaded.append(module_name)
    return loaded


def gguf_architecture(path):
    """The general.architecture metadata of a GGUF file, or None when unreadable."""
    try:
        from ..gguf import GGUF
        return GGUF(path).kv.get("general.architecture")
    except Exception:
        return None


def _matches(caps, model_id, architecture, device):
    if architecture and caps.architectures and architecture not in caps.architectures:
        return False
    if model_id and caps.model_ids and model_id not in caps.model_ids:
        return False
    if device and caps.devices and device not in caps.devices:
        return False
    return True


def _specificity(caps, model_id, architecture):
    score = 0
    if architecture and architecture in caps.architectures:
        score += 2
    if model_id and model_id in caps.model_ids:
        score += 1
    return score


def resolve(model_id_or_gguf_path, device=None, config=None):
    """Return a backend instance that can serve the model.

    ``model_id_or_gguf_path`` may be a model id (matched against declared model
    ids) or a path to a ``.gguf`` file (matched against declared architectures via
    its metadata). ``device`` filters by declared device names ("cuda", "cpu",
    "rocm", ...). When ``config`` is given the returned backend is loaded with it.

    Raises BackendNotFound listing what is registered when nothing matches. The
    fake backend and the T4.2 adapters for Bonsai, Nanbeige, Nemotron
    Lightning, Gemma 4 and DeepSeek V4.1 are registered; Cascade is not
    ported yet.
    """
    load_external_backends()
    hint = str(model_id_or_gguf_path or "")
    architecture = None
    model_id = hint
    if hint and os.path.isfile(hint):
        architecture = gguf_architecture(hint)
        model_id = None if architecture else hint
    candidates = []
    for backend_name in _ORDER:
        factory, caps = _REGISTRY[backend_name]
        if _matches(caps, model_id, architecture, device):
            candidates.append((_specificity(caps, model_id, architecture), backend_name, factory))
    if not candidates:
        raise BackendNotFound(
            "no registered backend serves %r%s (registered: %s; some engines are not "
            "ported yet - see docs/BACKENDS.md)" % (hint, " (architecture %r)" % architecture if architecture else "",
                                       ", ".join(_ORDER) or "none"))
    candidates.sort(key=lambda item: -item[0])
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        # two backends match equally well: registration order would pick one silently and build
        # the wrong engine, so say so instead (narrow the registrations or pass a model id)
        raise BackendNotFound(
            "%r is ambiguous: %s and %s match it equally well - pass a model id the backend "
            "declares exclusively" % (hint, candidates[0][1], candidates[1][1]))
    factory = candidates[0][2]
    backend = factory()
    if config is not None:
        backend.load(config)
    return backend
