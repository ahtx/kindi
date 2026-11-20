"""Host-RAM accounting follows the process's cgroup limit, not the host's MemTotal."""
from pathlib import Path

import pytest

from kindi import hostram


def test_cgroup_limit_reads_v2_memory_max(tmp_path):
    (tmp_path / "memory.max").write_text(str(8 * 2 ** 30))
    assert hostram._cgroup_limit(str(tmp_path)) == 8 * 2 ** 30


def test_cgroup_limit_reads_v1_and_takes_the_minimum(tmp_path):
    (tmp_path / "memory.max").write_text(str(16 * 2 ** 30))
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.limit_in_bytes").write_text(str(8 * 2 ** 30))
    assert hostram._cgroup_limit(str(tmp_path)) == 8 * 2 ** 30


def test_cgroup_limit_treats_max_and_huge_sentinels_as_unlimited(tmp_path):
    (tmp_path / "memory.max").write_text("max")
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.limit_in_bytes").write_text("9223372036854771712")
    assert hostram._cgroup_limit(str(tmp_path)) is None


def test_total_avail_capped_at_the_cgroup_limit(monkeypatch):
    monkeypatch.setattr(hostram, "_windows", lambda: None)
    monkeypatch.setattr(hostram, "_meminfo", lambda key: {"MemTotal": 64 * 2 ** 30, "MemAvailable": 32 * 2 ** 30}.get(key))
    monkeypatch.setattr(hostram, "_cgroup_limit", lambda root=None: 8 * 2 ** 30)
    assert hostram.total() == 8 * 2 ** 30
    assert hostram.available() == 8 * 2 ** 30


def test_no_cgroup_keeps_the_host_figures(monkeypatch):
    monkeypatch.setattr(hostram, "_windows", lambda: None)
    monkeypatch.setattr(hostram, "_meminfo", lambda key: {"MemTotal": 64 * 2 ** 30, "MemAvailable": 32 * 2 ** 30}.get(key))
    monkeypatch.setattr(hostram, "_cgroup_limit", lambda root=None: None)
    assert hostram.total() == 64 * 2 ** 30
    assert hostram.available() == 32 * 2 ** 30
