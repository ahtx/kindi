# SPDX-License-Identifier: AGPL-3.0-only
"""The Intel pack probe: pack parsing, shape grouping and the projection, without a GPU."""
import json
import os

import numpy as np
import pytest

from kindi import intel_pack
from kindi.convert import phase_bytes


def _pack(tmp_path, shapes=((8, 256), (16, 128)), f32=True):
    """Write the smallest pack the probe needs: index.json, kindi_pack.json, bcodes.bin, dpm.bin."""
    tensors, bcodes, dpm = {}, [], []
    bc_off = dpm_off = 0
    for n, (rows, cols) in enumerate(shapes):
        trits = ((np.arange(rows)[:, None] * 131 + np.arange(cols)[None, :] * 17) % 3).astype(np.uint8)
        codes = phase_bytes(trits)
        packed = np.ascontiguousarray(codes.T)
        phases = cols // 128
        scales = np.ones((rows, phases), dtype=np.float16)
        tensors["blk.0.mat%d.weight" % n] = {
            "dims": [rows, cols], "type": 143, "folded": False, "rows": rows, "ne0": cols,
            "bc_off": bc_off, "dpm_off": dpm_off}
        bcodes.append(packed.tobytes())
        dpm.append(np.ascontiguousarray(scales.T).tobytes())
        bc_off += packed.nbytes
        dpm_off += rows * phases * 2
    if f32:
        tensors["blk.0.norm.weight"] = {"dims": [8], "type": 0, "folded": False, "rows": 1,
                                        "ne0": 8, "f32_off": 0, "nfloats": 8}
    os.makedirs(tmp_path, exist_ok=True)
    with open(tmp_path / "index.json", "w") as f:
        json.dump({"tensors": tensors, "base": 0}, f)
    with open(tmp_path / "kindi_pack.json", "w") as f:
        json.dump({"model": "Ternary-Bonsai-2.5-27B", "source_file": "x.gguf"}, f)
    with open(tmp_path / "bcodes.bin", "wb") as f:
        f.write(b"".join(bcodes))
    with open(tmp_path / "dpm.bin", "wb") as f:
        f.write(b"".join(dpm))
    return tmp_path


def test_pack_tensors_keeps_only_ternary_ones(tmp_path):
    pack = _pack(tmp_path)
    tensors, manifest, index = intel_pack.pack_tensors(pack)
    assert [t["name"] for t in tensors] == ["blk.0.mat0.weight", "blk.0.mat1.weight"]
    assert manifest["model"] == "Ternary-Bonsai-2.5-27B"
    assert len(index["tensors"]) == 3                   # the f32 norm is in the pack, not in the list
    assert tensors[0]["codes_bytes"] == 8 * 256 // 4 and tensors[0]["scale_bytes"] == 8 * 2 * 2


def test_shapes_group_and_count_tensors(tmp_path):
    pack = _pack(tmp_path, shapes=((8, 256), (8, 256), (16, 128)))
    tensors, _manifest, _index = intel_pack.pack_tensors(pack)
    shapes = intel_pack.shapes_of(tensors)
    by_shape = {(s["rows"], s["cols"]): s for s in shapes}
    assert by_shape[(8, 256)]["tensors"] == 2
    assert by_shape[(16, 128)]["tensors"] == 1
    assert shapes[0]["bytes"] >= shapes[-1]["bytes"]     # largest first


def test_projection_sums_every_tensor_once(tmp_path):
    pack = _pack(tmp_path, shapes=((8, 256), (8, 256), (16, 128)))
    tensors, _manifest, _index = intel_pack.pack_tensors(pack)
    per_shape = {(8, 256): 0.002, (16, 128): 0.001}
    projection = intel_pack.project(tensors, per_shape)
    assert projection["seconds"] == pytest.approx(0.002 * 2 + 0.001)   # one pass per tensor
    moved = sum(t["codes_bytes"] + t["scale_bytes"] for t in tensors)
    assert projection["weight_bytes"] == moved
    assert projection["tokens_per_second"] == pytest.approx(1 / 0.005, rel=1e-3)


def test_work_lists_index_every_row_of_every_tensor(tmp_path):
    pack = _pack(tmp_path, shapes=((8, 256), (16, 128), (9, 256)))
    tensors, _manifest, _index = intel_pack.pack_tensors(pack)
    lists = intel_pack.work_lists(tensors)
    each = [(t["rows"] + 7) // 8 for t in tensors]
    assert lists["work"] == sum(each)                       # ceil(rows/8) items per tensor
    assert list(lists["pref"]) == [0, each[0], each[0] + each[1]]
    assert list(lists["ppref"]) == [0, 8, 24]               # rows, not work items
    assert lists["rows"] == 8 + 16 + 9
    assert list(lists["coffs"])[0] == 0 and list(lists["soffs"])[0] == 0
    assert list(lists["coffs"])[1] == tensors[0]["codes_bytes"]
    assert list(lists["soffs"])[1] == tensors[0]["rows"] * tensors[0]["phases"] * 4


def test_chunks_stay_under_one_allocation(tmp_path):
    pack = _pack(tmp_path, shapes=((8, 256), (8, 256), (16, 128)))
    tensors, _manifest, _index = intel_pack.pack_tensors(pack)
    cap = 2 * tensors[0]["codes_bytes"] + 1
    chunks = list(intel_pack.chunks_of(tensors, cap))
    assert [len(c) for c in chunks] == [2, 1]               # the third tensor would overflow
    assert sum(len(c) for c in chunks) == len(tensors)       # every tensor exactly once
    assert [t["name"] for c in chunks for t in c] == [t["name"] for t in tensors]
    assert all(sum(t["codes_bytes"] for t in c) <= cap for c in chunks)


def test_chunk_bytes_follows_the_device_limit():
    assert intel_pack.chunk_bytes({"max_mem_alloc_mib": 4096}) == int(4096 * (1 << 20) * 0.94)
    assert intel_pack.chunk_bytes({"max_mem_alloc_mib": None}) == intel_pack.PERSIST_CHUNK_FALLBACK
    assert intel_pack.chunk_bytes(None) == intel_pack.PERSIST_CHUNK_FALLBACK


def test_missing_pack_fails_without_a_gpu(tmp_path):
    report = intel_pack.run_pack(str(tmp_path / "nope"))
    assert report["result"] == "fail" and "cannot read the pack" in report["error"]


def test_a_pack_without_ternary_tensors_fails(tmp_path):
    pack = _pack(tmp_path, shapes=())
    os.makedirs(pack, exist_ok=True)
    with open(pack / "index.json", "w") as f:
        json.dump({"tensors": {"blk.0.norm.weight": {"rows": 8, "ne0": 8, "type": 0}}}, f)
    with open(pack / "bcodes.bin", "wb"):
        pass
    with open(pack / "dpm.bin", "wb"):
        pass
    report = intel_pack.run_pack(str(pack))
    assert report["result"] == "fail" and "no ternary tensors" in report["error"]


def test_metadata_without_the_byte_files_is_a_fail_report(tmp_path):
    # round-5 audit F2: the two os.path.getsize calls and the two fromfile() reads ran outside every
    # guard, so a directory with index.json/kindi_pack.json but no bcodes.bin raised FileNotFoundError
    # through the CLI instead of answering the probe's {"result": "fail"} contract.
    pack = _pack(tmp_path)
    os.remove(pack / "bcodes.bin")
    report = intel_pack.run_pack(str(pack))
    assert report["result"] == "fail"
    assert "cannot read the pack's byte files" in report["error"]


def test_unavailable_opencl_loader_fails_the_report(tmp_path):
    pack = _pack(tmp_path)
    report = intel_pack.run_pack(str(pack), loader=str(tmp_path / "no-such-loader.dll"))
    assert report["result"] == "fail"
    assert report["pack"]["ternary"] == 2
    assert "loader" in report["error"].lower() or "OpenCL" in report["error"]


def test_report_is_json_serialisable(tmp_path):
    pack = _pack(tmp_path)
    report = intel_pack.run_pack(str(pack), loader=str(tmp_path / "none.dll"))
    json.dumps(report)
    text = "\n".join(intel_pack.render(report))
    assert "no model run" in text and "result: fail" in text
