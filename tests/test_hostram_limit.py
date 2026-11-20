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


def test_cgroup_usage_reads_v2_and_takes_the_largest(tmp_path):
    (tmp_path / "memory.current").write_text(str(3 * 2 ** 30))
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.usage_in_bytes").write_text(str(4 * 2 ** 30))
    assert hostram._cgroup_usage(str(tmp_path)) == 4 * 2 ** 30


def test_cgroup_usage_unknown_without_the_files(tmp_path):
    assert hostram._cgroup_usage(str(tmp_path)) is None


def test_total_avail_capped_at_the_cgroup_limit(monkeypatch):
    monkeypatch.setattr(hostram, "_windows", lambda: None)
    monkeypatch.setattr(hostram, "_meminfo", lambda key: {"MemTotal": 64 * 2 ** 30, "MemAvailable": 32 * 2 ** 30}.get(key))
    monkeypatch.setattr(hostram, "_cgroup_limit", lambda root=None: 8 * 2 ** 30)
    monkeypatch.setattr(hostram, "_cgroup_usage", lambda root=None: None)
    assert hostram.total() == 8 * 2 ** 30
    assert hostram.available() == 8 * 2 ** 30


def test_available_subtracts_the_cgroup_usage_from_the_limit(monkeypatch):
    monkeypatch.setattr(hostram, "_windows", lambda: None)
    monkeypatch.setattr(hostram, "_meminfo", lambda key: {"MemTotal": 64 * 2 ** 30, "MemAvailable": 32 * 2 ** 30}.get(key))
    monkeypatch.setattr(hostram, "_cgroup_limit", lambda root=None: 8 * 2 ** 30)
    monkeypatch.setattr(hostram, "_cgroup_usage", lambda root=None: 6 * 2 ** 30)
    assert hostram.total() == 8 * 2 ** 30
    assert hostram.available() == 2 * 2 ** 30


def test_available_is_the_smaller_of_host_free_and_cgroup_room(monkeypatch):
    monkeypatch.setattr(hostram, "_windows", lambda: None)
    monkeypatch.setattr(hostram, "_meminfo", lambda key: {"MemTotal": 64 * 2 ** 30, "MemAvailable": 3 * 2 ** 30}.get(key))
    monkeypatch.setattr(hostram, "_cgroup_limit", lambda root=None: 8 * 2 ** 30)
    monkeypatch.setattr(hostram, "_cgroup_usage", lambda root=None: 4 * 2 ** 30)
    assert hostram.available() == 3 * 2 ** 30


def test_available_is_zero_when_the_cgroup_limit_is_used_up(monkeypatch):
    monkeypatch.setattr(hostram, "_windows", lambda: None)
    monkeypatch.setattr(hostram, "_meminfo", lambda key: {"MemTotal": 64 * 2 ** 30, "MemAvailable": 32 * 2 ** 30}.get(key))
    monkeypatch.setattr(hostram, "_cgroup_limit", lambda root=None: 8 * 2 ** 30)
    for used in (8 * 2 ** 30, 9 * 2 ** 30):  # at and over the limit
        monkeypatch.setattr(hostram, "_cgroup_usage", lambda root=None, used=used: used)
        assert hostram.available() == 0


def test_no_cgroup_keeps_the_host_figures(monkeypatch):
    monkeypatch.setattr(hostram, "_windows", lambda: None)
    monkeypatch.setattr(hostram, "_meminfo", lambda key: {"MemTotal": 64 * 2 ** 30, "MemAvailable": 32 * 2 ** 30}.get(key))
    monkeypatch.setattr(hostram, "_cgroup_limit", lambda root=None: None)
    assert hostram.total() == 64 * 2 ** 30
    assert hostram.available() == 32 * 2 ** 30
