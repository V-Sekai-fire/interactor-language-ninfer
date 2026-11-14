"""Parity: the Mojo artifact reader against the Python reference reader and the C++ rules.

Artifacts come from the production writer (`tools.artifact.writer`), single-file and sharded.
The Mojo side is `mojo/tests/dump_artifact.mojo`; the test skips when `mojo` is not on PATH.

Run: pytest mojo/tests   (with `mojo` on PATH)
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.artifact.framing import HEADER  # noqa: E402
from tools.artifact.layouts import encoded_size, row_split_geometry  # noqa: E402
from tools.artifact.reader import Artifact  # noqa: E402
from tools.artifact.schema import ResourceSpec, TensorSpec  # noqa: E402
from tools.artifact.writer import ArtifactWriter  # noqa: E402

MOJO = shutil.which("mojo")
pytestmark = pytest.mark.skipif(MOJO is None, reason="mojo is not on PATH")
DUMP = ROOT / "mojo" / "tests" / "dump_artifact.mojo"


def dump(path: Path, *objects: str):
    run = subprocess.run(
        [MOJO, "run", "-I", str(ROOT / "mojo"), str(DUMP), str(path), *objects],
        capture_output=True,
        text=True,
    )
    lines: dict[str, list[list[str]]] = {}
    for line in run.stdout.splitlines():
        head, *rest = line.split(" ")
        lines.setdefault(head, []).append(rest)
    return run, lines


def text(run) -> str:
    return run.stderr + run.stdout


def build(path: Path, limit: int | None):
    matrix = bytes((i * 37 + 11) % 251 for i in range(130 * 130 * 2))
    blob = bytes(range(251)) * 40
    scale = struct.pack("<f", 2.0)
    specs = [
        TensorSpec("matrix", (130, 130), "bf16", "contiguous_le_v1"),
        TensorSpec("scale", (), "fp32", "contiguous_le_v1"),
        TensorSpec("table", (5,), "int32", "contiguous_le_v1"),
        TensorSpec("q4", (4, 200), "q4_g64_fp16", "row_split_k128_v1"),
        TensorSpec("blocks", (3, 64), "gguf_q4_0", "gguf_blocks_v1"),
        ResourceSpec("template", len(blob)),
    ]
    table = struct.pack("<5i", 1, -2, 3, -2147483648, 2147483647)
    q4 = bytes((i * 13 + 5) % 256 for i in range(encoded_size("row_split_k128_v1", "q4_g64_fp16", (4, 200))))
    blocks = bytes((i * 7 + 3) % 256 for i in range(encoded_size("gguf_blocks_v1", "gguf_q4_0", (3, 64))))
    with ArtifactWriter(
        path,
        specs,
        components={
            "text": {
                "config": {"n": 2, "name": "caf\u00e9 \U0001f600"},
                "resources": {"chat_template.jinja": "template"},
            },
            "mtp": {"config": {}, "target": "text"},
        },
        bindings={
            "whole": {"object": "matrix"},
            "q4": {"object": "q4"},
            "blocks": {"object": "blocks"},
            "scalar": {"object": "scale"},
            "table": {"object": "table"},
            "reordered": {
                "parts": [
                    {"object": "matrix", "range": [16770, 16900]},
                    {"object": "matrix", "range": [130, 260]},
                ]
            },
        },
        uses=[
            {
                "parameter": "whole",
                "input": "x",
                "activation_policy": "AllowA8",
                "auxiliaries": {"b": {"object": "scale"}, "a": {"object": "scale"}},
            }
        ],
        max_file_bytes=limit,
    ) as writer:
        writer.write_object("matrix", matrix)
        writer.write_object("scale", scale)
        writer.write_object("table", table)
        writer.write_object("q4", q4)
        writer.write_object("blocks", blocks)
        writer.write_object("template", blob)
        directory = writer.directory.to_json()
    return directory, {"matrix": matrix, "scale": scale, "template": blob, "table": table,
                       "q4": q4, "blocks": blocks}


@pytest.mark.parametrize("limit", [None, 12288])
def test_mojo_reader_matches_python_reader(tmp_path, limit):
    path = tmp_path / "t.ninfer"
    directory, payloads = build(path, limit)
    run, out = dump(path, "matrix", "scale", "template", "table", "q4", "blocks")
    assert run.returncode == 0, text(run)
    with Artifact(path) as ref:
        assert out["payload_bytes"] == [[str(ref.payload_bytes)]]
        assert out["file_bytes"] == [[str(ref.file_bytes)]]
        for obj in ref.objects:
            if obj.kind == "tensor":
                shape = "".join(f"{d}," for d in obj.shape)
                want = [obj.id, shape, obj.format, obj.layout, str(obj.offset), str(obj.bytes), str(obj.divisors)]
                assert [r for r in out["tensor"] if r[0] == obj.id] == [want]
            else:
                want = [obj.id, obj.encoding, str(obj.offset), str(obj.bytes)]
                assert [r for r in out["resource"] if r[0] == obj.id] == [want]
        for object_id, data in payloads.items():
            assert [r for r in out["bytes"] if r[0] == object_id] == [[object_id, data.hex()]]
            assert ref.read_object(object_id) == data
    assert len(out["file"]) == len(directory["files"])
    if limit:
        assert len(out["file"]) > 1, "sharded fixture produced a single file"
    assert [r[0] for r in out["component"]] == ["mtp", "text"]
    assert out["component"][0][1] == "text"
    assert out["component"][1][2] == "chat_template.jinja=template,"
    reordered = [r for r in out["binding"] if r[0] == "reordered"][0]
    assert reordered == ["reordered", "False", "260", "matrix[16770:16900],matrix[130:260],"]
    # Geometry, interpreted on request, matches the reference sizes.
    geo = {r[0]: r for r in out["geometry"]}
    assert geo["matrix"][1:5] == ["bf16", "contiguous_le_v1", str(130 * 130), str(130 * 130 * 2)]
    g = row_split_geometry("q4_g64_fp16", (4, 200))
    q4 = geo["q4"]
    assert q4[1:3] == ["q4_g64_fp16", "row_split_k128_v1"]
    assert [int(v) for v in q4[4:8]] == [g.payload_bytes, g.k_pad, 64, g.base_row_bytes]
    assert int(geo["blocks"][4]) == encoded_size("gguf_blocks_v1", "gguf_q4_0", (3, 64))
    assert "geometry_error" not in out
    # Host values read through the Binder (single and sharded files, part order preserved).
    values = {r[0]: r for r in out["values"]}
    assert values["whole"] == ["whole", "bf16", str(130 * 130), payloads["matrix"][: 130 * 130 * 2].hex()]
    assert values["scalar"] == ["scalar", "fp32", "1", payloads["scale"].hex()]
    assert values["table"] == ["table", "int32", "5", payloads["table"].hex()]
    full = payloads["matrix"]
    want = full[16770 * 2 : 16900 * 2] + full[130 * 2 : 260 * 2]
    assert values["reordered"] == ["reordered", "bf16", "260", want.hex()]
    assert "q4" not in values and "blocks" not in values  # quantized: not Host values
    assert {r[0] for r in out["parameter"]} == {"whole", "q4", "blocks", "scalar", "table", "reordered"}
    assert out["resource_bytes"] == [["text", "chat_template.jinja", "template"]]
    assert out["missing_parameter"] == [["missing", "logical", "parameter", "nope"]]
    whole = [r for r in out["binding"] if r[0] == "whole"][0]
    assert whole == ["whole", "True", str(130 * 130), f"matrix[0:{130 * 130}],"]


def rewrite(path: Path, mutate) -> None:
    raw = bytearray(path.read_bytes())
    _, reserve, _ = HEADER.unpack(bytes(raw[:32]))
    directory = json.loads(raw[32 : 32 + reserve].decode().rstrip(" "))
    mutate(directory)
    data = json.dumps(directory, separators=(",", ":")).encode()
    assert len(data) <= reserve
    raw[32 : 32 + reserve] = data + b" " * (reserve - len(data))
    path.write_bytes(bytes(raw))


def make_small(path: Path) -> None:
    with ArtifactWriter(
        path,
        [TensorSpec("w", (2, 2), "bf16", "contiguous_le_v1"), ResourceSpec("r", 3)],
        components={"text": {"config": {}}},
        bindings={"b": {"object": "w"}},
    ) as writer:
        writer.write_object("w", bytes(8))
        writer.write_object("r", b"abc")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(extra=1), "directory: unknown member extra"),
        (lambda d: d.pop("uses"), "directory: missing uses"),
        (lambda d: d["objects"][0].update(kind="blob"), "unknown object kind blob"),
        (lambda d: d["objects"][1].update(offset=0), "object overlaps, is out of order or exceeds payload"),
        (lambda d: d["bindings"].update(b={"object": "nope"}), "missing object nope"),
        (lambda d: d["bindings"].update(b={"parts": [{"object": "w", "range": [0, 9]}]}), "Part exceeds source logical elements"),
        (lambda d: d["components"]["text"].update(target="ghost"), "missing target ghost"),
        (lambda d: d["components"].update(other={"config": {}, "proposal": {"domain": "full"}}), "only text may declare proposal"),
        (lambda d: d.update(uses=[{"parameter": "b", "input": "i", "activation_policy": "X"}]), "unknown activation policy X"),
        (lambda d: d.update(uses=[{"parameter": "b", "input": "i"}] * 2), "b@i: duplicate Use"),
        (lambda d: d["files"][0].update(path="x"), "files[0].path must be null"),
    ],
)
def test_mojo_rejects_what_the_reference_rejects(tmp_path, mutate, message):
    path = tmp_path / "s.ninfer"
    make_small(path)
    rewrite(path, mutate)
    run, _ = dump(path)
    assert run.returncode != 0
    assert message in text(run)


def test_mojo_rejects_framing_errors(tmp_path):
    path = tmp_path / "s.ninfer"
    make_small(path)
    good = path.read_bytes()
    cases = [
        (b"NOTNINFE" + good[8:], "expected NInfer v3 entry magic"),
        (good[:7] + b"\x02" + good[8:], "NInfer v2 artifact is not supported"),
        (good + b"\0", "entry length differs from directory"),
        (good[:8] + struct.pack("<Q", 1 << 40) + good[16:], "invalid directory length"),
    ]
    for data, message in cases:
        path.write_bytes(data)
        run, _ = dump(path)
        assert run.returncode != 0 and message in text(run), (message, text(run))


def test_mojo_rejects_a_foreign_continuation(tmp_path):
    path = tmp_path / "t.ninfer"
    build(path, 12288)
    part = tmp_path / "t.ninfer.part-0001"
    raw = bytearray(part.read_bytes())
    raw[16] ^= 0xFF  # artifact id
    part.write_bytes(bytes(raw))
    run, _ = dump(path, "matrix")
    assert run.returncode != 0
    assert "continuation header does not belong to this entry" in text(run)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["objects"][0].update(format="fp32"), "w: encoded size or object alignment differs from layout"),
        (lambda d: d["objects"][0].update(format="zzz"), "w: unknown tensor format: zzz"),
        (lambda d: d["objects"][0].update(layout="row_split_panel4_v1"), "w: unknown tensor layout: row_split_panel4_v1"),
        (lambda d: d["objects"][0].update(layout="row_split_k128_v1"), "w: RowSplit requires a grouped integer format"),
        (lambda d: d["objects"][0].update(divisors=2), "w: only an NVFP4 plane carries more than one divisor"),
    ],
)
def test_bad_tensors_fail_only_when_requested(tmp_path, mutate, message):
    # The directory parses; the tensor is interpreted (and rejected) on request, as in C++.
    path = tmp_path / "s.ninfer"
    make_small(path)
    rewrite(path, mutate)
    run, out = dump(path)
    assert run.returncode == 0, text(run)
    assert out["geometry_error"] == [["w"] + message.split(" ")]
    assert [r[0] for r in out["parameter_error"]] == ["b"]
    assert message in " ".join(out["parameter_error"][0])
