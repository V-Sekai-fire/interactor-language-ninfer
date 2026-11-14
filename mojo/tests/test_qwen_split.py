"""The combinator pre-tokenizer against the regular expression it replaces, run by the `regex`
module (which has \\p{L} etc.). Texts draw on characters whose categories are stable across
Unicode versions, so a table-version difference cannot show up as a mismatch."""

import random
import subprocess

import pytest

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOJO = shutil.which("mojo")

regex = pytest.importorskip("regex")

PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

POOL = (
    list("abcxyzABCXYZ019 '.,!?-_(){}[]\"/:;")
    + ["\n", "\r", "\r\n", "\t", "  ", "   ", "\n\n", " \n", "\n ", " \r\n "]
    + ["'s", "'T", "'re", "'VE", "'m", "'LL", "'d", "'x", "''"]
    + ["é", "ß", "ñ", "Ω", "я", "漢", "字", "あ", "ア", "한", "٣", "٤", "५", "²", "½", "Ⅷ"]
    + ["\u0301", "\u0308", "\u0300é", "\u200b", "\u00a0", "\u2003", "\u3000", "\u2028"]
    + ["😀", "🎉", "€", "£", "—", "…", "«", "»", "→", "∑", "©"]
)


def _split_regex(text: str) -> list[int]:
    return [m.end() for m in regex.finditer(PATTERN, text)]


def _run(texts, tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("\n".join(t.encode().hex() for t in texts) + "\n")
    out = subprocess.run(
        [MOJO, "run", "-I", str(ROOT / "mojo"),
         str(ROOT / "mojo/tests/qwen_split_probe.mojo"), str(src)],
        capture_output=True, text=True, timeout=500,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return [list(map(int, l.split())) for l in out.stdout.strip("\n").split("\n")]


def _byte_ends(text: str) -> list[int]:
    ends = []
    for e in _split_regex(text):
        ends.append(len(text[:e].encode()))
    return ends


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_split_matches_regex(tmp_path, seed):
    rng = random.Random(seed)
    texts = [
        "".join(rng.choice(POOL) for _ in range(rng.randint(1, 24))) for _ in range(500)
    ] + ["", "a", " ", "\n", "  \n  x", "x  ", "x \n\n y", "don't DON'T it's", "1234 56", "a\u0301b", "\u0301a", " \u0301a"]
    texts = [t for t in texts if t]
    got = _run(texts, tmp_path)
    bad = [(t, _byte_ends(t), g) for t, g in zip(texts, got) if _byte_ends(t) != g]
    assert not bad, bad[:3]
