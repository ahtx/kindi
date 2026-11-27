"""The compact draft head of Flash-Next: the MTP head file with its 512 experts as Q2_0 (about 0.7 GB instead of
1.4 GB on the card; the freed memory holds more of the main model's experts). The head only drafts, so this changes
the acceptance rate, never the output. The engine uses qwen4exp-mtp-q2.gguf next to the model when it is there
(KINDI_MTP_COMPACT=0: the full head).

usage: python tools/qwen4exp_mtp_compact.py HEAD.gguf [OUT.gguf]   (default OUT: qwen4exp-mtp-q2.gguf beside HEAD)"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kindi import qwen4exp_mtp_import as MI  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)),
                                                              "qwen4exp-mtp-q2.gguf")
    t0 = time.time()
    MI.compact(src, out)
    print("written %s (%.2f GB) in %.0f s" % (out, os.path.getsize(out) / 1e9, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
