"""Ignore inherited model/config settings from the reused base image."""
import os
import sys
for name in list(os.environ):
    if name.startswith('KINDI_') and not name.startswith('KINDI_LIGHTNING_'):os.environ.pop(name)
os.environ.setdefault('KINDI_LIGHTNING_KERNELS','optimized')
os.environ.setdefault('KINDI_LIGHTNING_DENSE_PREFILL','tile16')
os.environ.setdefault('KINDI_LIGHTNING_EXPERT_PREFILL','tile16')
os.environ.setdefault('KINDI_LIGHTNING_ATTENTION','split')
os.environ.setdefault('KINDI_LIGHTNING_PREFILL_CHUNK','128')
os.makedirs('/workspace/state/kernels/driver',exist_ok=True)
os.execv(sys.executable,[sys.executable,'-m','kindi','--config','/release/config/kindi.conf']+(sys.argv[1:] or ['serve']))
