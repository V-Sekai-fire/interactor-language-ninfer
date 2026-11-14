"""Parity: Mojo Unicode layer against the engine's own src/text/unicode.cpp over utf8proc.

The oracle is the real C++ translation unit, compiled here with the system C++ compiler, so a
mismatch means the Mojo port differs from what the CUDA engine does.
"""

from __future__ import annotations

from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
MOJO = shutil.which("mojo")
CXX = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
pytestmark = pytest.mark.skipif(MOJO is None or CXX is None, reason="needs mojo and a C++ compiler")
PROBE = ROOT / "mojo" / "tests" / "unicode_probe.mojo"


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    exe = tmp_path_factory.mktemp("oracle") / "unicode_oracle"
    subprocess.run(
        [CXX, "-std=c++20", "-O1", f"-I{ROOT / 'src'}", f"-I{ROOT / 'third_party'}",
         f"-I{ROOT / 'third_party' / 'utf8proc'}",
         str(ROOT / "mojo" / "tests" / "unicode_oracle.cpp"), str(ROOT / "src" / "text" / "unicode.cpp"),
         str(ROOT / "third_party" / "utf8proc" / "utf8proc.c"), "-o", str(exe)],
        check=True,
    )
    return exe


def run_mojo(*args: str) -> str:
    done = subprocess.run([MOJO, "run", "-I", str(ROOT / "mojo"), str(PROBE), *args],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def run_oracle(exe, *args: str) -> str:
    return subprocess.run([str(exe), *args], capture_output=True, text=True, check=True).stdout


def test_categories_match_for_every_code_point(oracle):
    mine = run_mojo("classes").strip()
    theirs = run_oracle(oracle, "classes").strip()
    assert len(mine) == len(theirs) == 0x110000
    bad = [i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    assert not bad, f"{len(bad)} code points differ, first {[hex(i) for i in bad[:5]]}"
    assert mine.count("1") > 100000  # the table really classifies letters


def corpus() -> list[bytes]:
    rng = random.Random(20261114)
    cases: list[bytes] = []
    # Every valid code point alone.
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        cases.append(chr(cp).encode())
    # Interesting pools: combining marks, decomposables, Hangul jamo, ASCII.
    marks = [0x300, 0x301, 0x302, 0x308, 0x323, 0x327, 0x328, 0x316, 0x31B, 0x345, 0x5B0, 0x5B4, 0x93C, 0x3099, 0x309A]
    starters = [0x41, 0x61, 0x45, 0x4F, 0x55, 0x1100, 0x1161, 0x11A8, 0xAC00, 0xAC01, 0x915, 0x928, 0x3C9, 0x1B05, 0x1B35, 0x1D157, 0x1D1BB]
    pool = marks + starters + [0x20, 0xDC, 0x1E9B, 0x212B, 0x2126, 0xFB1D, 0x1D15E]
    for _ in range(30000):
        n = rng.randint(1, 8)
        cases.append("".join(chr(rng.choice(pool)) for _ in range(n)).encode())
    for _ in range(5000):
        n = rng.randint(1, 12)
        cases.append("".join(chr(rng.choice([0x41, 0xE9, 0x300, 0x301, 0xAC00, 0x1100, 0x1161, 0x20AC, 0x1F600, 0x4E2D])) for _ in range(n)).encode())
    return cases


def invalid_corpus() -> list[bytes]:
    base = [b"\xff", b"\xc0\x80", b"\xc1\xbf", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xed\xbf\xbf",
            b"\xf4\x90\x80\x80", b"\xf0\x80\x80\x80", b"\xf5\x80\x80\x80", b"\xc3", b"\xe2\x82", b"\xf0\x9f\x98",
            b"a\x80", b"\x80", b"\xe2\x28\xa1", b"abc\xc3(", b"\xf8\x88\x80\x80\x80"]
    rng = random.Random(7)
    out = list(base)
    for _ in range(3000):
        n = rng.randint(1, 6)
        out.append(bytes(rng.randrange(256) for _ in range(n)))
    return out


@pytest.mark.parametrize("mode", ["iter", "nfc"])
def test_valid_text_matches(oracle, tmp_path, mode):
    cases = corpus()
    path = tmp_path / "cases.txt"
    path.write_text("\n".join(c.hex() for c in cases) + "\n")
    mine = run_mojo(mode, str(path)).splitlines()
    theirs = run_oracle(oracle, mode, str(path)).splitlines()
    assert len(mine) == len(theirs) == len(cases)
    bad = [i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    assert not bad, [(cases[i], mine[i], theirs[i]) for i in bad[:3]]
    if mode == "nfc":
        assert sum(1 for c, m in zip(cases, mine) if m != "ok " + c.hex()) > 2000  # NFC really changed things


@pytest.mark.parametrize("mode", ["iter", "nfc"])
def test_invalid_utf8_is_rejected_the_same_way(oracle, tmp_path, mode):
    cases = invalid_corpus()
    path = tmp_path / "cases.txt"
    path.write_text("\n".join(c.hex() for c in cases) + "\n")
    mine = run_mojo(mode, str(path)).splitlines()
    theirs = run_oracle(oracle, mode, str(path)).splitlines()
    assert len(mine) == len(theirs) == len(cases)
    bad = [i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    assert not bad, [(cases[i], mine[i], theirs[i]) for i in bad[:3]]
    assert sum(1 for m in mine if m.startswith("err")) > 1000
