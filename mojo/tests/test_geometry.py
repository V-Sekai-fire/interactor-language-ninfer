"""Parity: Mojo `weight_geometry` against the Python reference layouts (`tools/artifact`).

A sweep of every format x layout pair over shapes that hit padding, alignment, divisor and
rank edge cases. Both sides must agree on accept/reject and, when they accept, on every
size the Python side computes.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.artifact.formats import NUMERIC_FORMATS  # noqa: E402
from tools.artifact.layouts import (  # noqa: E402
    LAYOUTS,
    encoded_size,
    gguf_blocks_geometry,
    row_split_geometry,
)

MOJO = shutil.which("mojo")
pytestmark = pytest.mark.skipif(MOJO is None, reason="mojo is not on PATH")
PROBE = ROOT / "mojo" / "tests" / "geometry_probe.mojo"

SHAPES = [(), (7,), (3, 5), (1, 1), (4, 64), (8, 200), (5, 127), (5, 128), (6, 129), (2, 256),
          (128, 64), (128, 128), (256, 192), (130, 130), (256, 64), (384, 128), (2, 3, 4),
          (4, 31), (4, 32), (4, 33), (3, 255), (3, 256), (3, 257), (64, 4096)]
DIVISORS = [1, 2, 3, 4]


def python_side(format_name: str, layout_name: str, shape, divisors):
    try:
        size = encoded_size(layout_name, format_name, shape, divisors)
    except Exception as error:  # noqa: BLE001 - any reference rejection counts
        return None, str(error)
    return size, None


def cases():
    for layout in LAYOUTS:
        for fmt in NUMERIC_FORMATS:
            for shape in SHAPES:
                for divisors in DIVISORS if fmt == "nvfp4" else (1,):
                    yield fmt, layout, divisors, shape


def test_geometry_matches_the_python_reference(tmp_path):
    all_cases = list(cases())
    # Mojo-only probes: divisors on non-NVFP4 formats must be rejected by both.
    extra = [("bf16", "contiguous_le_v1", 2, (4, 4)), ("q4_g64_fp16", "row_split_k128_v1", 0, (4, 128))]
    lines = [
        f"{fmt} {layout} {divisors} {','.join(map(str, shape))}"
        for fmt, layout, divisors, shape in all_cases + extra
    ]
    case_file = tmp_path / "cases.txt"
    case_file.write_text("\n".join(lines) + "\n")
    run = subprocess.run(
        [MOJO, "run", "-I", str(ROOT / "mojo"), str(PROBE), str(case_file)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    results = run.stdout.splitlines()
    assert len(results) == len(lines)

    accepted = rejected = 0
    for (fmt, layout, divisors, shape), line, out in zip(all_cases + extra, lines, results):
        assert out.split(" ")[1:5] == line.split(" "), out
        fields = out.split(" ")
        ok = fields[0] == "ok"
        want, why = python_side(fmt, layout, shape, divisors)
        if (fmt, layout, divisors, shape) in extra:
            assert not ok, out
            rejected += 1
            continue
        assert ok == (want is not None), f"{line}: mojo={out!r} python={why!r}"
        if not ok:
            rejected += 1
            continue
        accepted += 1
        # ok FORMAT LAYOUT DIV SHAPE elements bytes padded group code_row high_row scale_row
        # stride code_bytes high_off high_bytes scale_off scale_bytes div_off div_count
        elements, nbytes, padded, group, code_row, high_row, scale_row, stride, code_bytes, \
            high_off, high_bytes, scale_off, scale_bytes, div_off, div_count = map(int, fields[5:])
        assert nbytes == want, f"{line}: bytes {nbytes} != {want}"
        n_elements = 1
        for d in shape:
            n_elements *= d
        assert elements == n_elements
        assert div_count == divisors
        if layout == "row_split_k128_v1":
            g = row_split_geometry(fmt, shape)
            assert (padded, code_row, high_row, scale_row) == (
                g.k_pad, g.base_row_bytes, g.high_row_bytes, g.scale_row_bytes), line
            assert (code_bytes, high_off, high_bytes, scale_off, scale_bytes) == (
                g.base_bytes, g.high_offset, g.high_bytes, g.scale_offset, g.scale_bytes), line
        if layout == "gguf_blocks_v1":
            g = gguf_blocks_geometry(fmt, shape)
            assert code_row == g.row_bytes and scale_off == g.payload_bytes and scale_bytes == 0, line
    assert accepted > 100 and rejected > 100, (accepted, rejected)
