"""Kindi - custom CUDA inference for ternary and Gemma 4 language models.

Named after Abu Yusuf al-Kindi (c. 801-873), whose work on frequency analysis is the root of statistical
language modelling.
"""
import os
import subprocess

UPSTREAM_VERSION = "0.3.0"
LIGHTNING_BACKEND_VERSION = "V3"
VERSION = "0.3.0-lightning-v3"


def _commit():
    """the source commit (12 hex digits) or None: kindi/COMMIT, which git archive fills in (export-subst), else git"""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = open(os.path.join(here, "COMMIT")).read().strip()
        if c and not c.startswith("$"):
            return c[:12]
    except OSError:
        pass
    try:
        # only Kindi's own checkout: a copy of the source inside another repository (a home folder in git, for
        # example) must not report that repository's commit
        r = subprocess.run(["git", "-C", here, "rev-parse", "--show-toplevel", "HEAD"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, universal_newlines=True, timeout=5)
        lines = r.stdout.split()
        if len(lines) == 2 and os.path.realpath(lines[0]) == os.path.realpath(os.path.dirname(here)):
            return lines[1][:12]
        return None
    except (OSError, subprocess.SubprocessError):
        return None


COMMIT = _commit()
__version__ = VERSION + ("+" + COMMIT[:7] if COMMIT and "-" in VERSION else "")
MODEL_ID = "bonsai2-2b"
