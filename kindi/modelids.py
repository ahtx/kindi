# SPDX-License-Identifier: AGPL-3.0-only
"""One resolver for "which registry model id does this pack manifest claim".

The ids live with their adapters (``MODEL_ID`` plus a ``model_id_for_manifest``
hook in ``kindi/backends/*.py``). The config's ``model_name: auto`` inference
and the engine gateway both resolve through here, so a family cannot be added
to one and forgotten in the other.

The hook is read from each registered backend's module on first use, not at
import time: ``config`` is loaded by every entry point, including ones that
never build an engine. Registering an adapter is therefore the only wiring a
new family needs; there is no second list to edit here.
"""
import importlib


def adapter_model_id(man):
    """The registry model id a backend adapter claims for this pack manifest, or None.

    ``None`` means no adapter claims the pack (a foreign or hand-edited
    manifest); the caller falls back to its own heuristic rather than inventing a
    family. Every registered hook is asked, because a manifest claimed by two
    adapters is contradictory and must be refused rather than resolved by
    registration order.
    """
    if not isinstance(man, dict) or not man:
        return None
    from .backends import backend_classes

    claims = []
    for factory in backend_classes():
        module_name = getattr(factory, "__module__", None)
        if not module_name:
            continue
        module = importlib.import_module(module_name)
        hook = getattr(module, "model_id_for_manifest", None)
        if hook is None:
            continue
        mid = hook(man)
        if mid:
            claims.append(mid)
    if len(claims) > 1:
        raise SystemExit("kindi: the pack manifest is claimed by more than one backend (%s): its "
                         "`model` and `backend` fields contradict each other" % ", ".join(claims))
    return claims[0] if claims else None
