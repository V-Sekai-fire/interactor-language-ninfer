"""Parity: Mojo Jinja lexer against third_party/llama-jinja's lexer.

The oracle compiles the NInfer fork of llama.cpp's jinja library with the system C++ compiler.
Inputs: the three chat templates this repo ships, plus generated templates that mix statements,
expressions, comments, trimming markers, string escapes, numbers, unary operators and mistakes.
"""

from __future__ import annotations

from pathlib import Path
import os
import random
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOJO = shutil.which("mojo")
# The engine ships on Linux with libstdc++, so prefer a GNU compiler (Homebrew's g++-N) as the oracle;
# its std::sort and typeid names are the ones the port reproduces.
CXX = (shutil.which(os.environ.get("NINFER_CXX", "")) if os.environ.get("NINFER_CXX") else None) or next(
    (c for c in (shutil.which(f"g++-{v}") for v in range(20, 10, -1)) if c), None
) or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
pytestmark = pytest.mark.skipif(MOJO is None or CXX is None, reason="needs mojo and a C++ compiler")


def build_jinja_oracle(work: Path, source: Path, name: str, extra=()) -> Path:
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    obj = work / "utf8proc.o"
    if not obj.exists():
        subprocess.run([cc, "-O1", "-c", f"-I{ROOT / 'third_party/utf8proc'}",
                        str(ROOT / "third_party/utf8proc/utf8proc.c"), "-o", str(obj)], check=True)
    exe = work / name
    subprocess.run(
        [CXX, "-std=c++20", "-O1", f"-I{ROOT / 'src'}", f"-I{ROOT / 'third_party/llama-jinja'}",
         f"-I{ROOT / 'third_party'}", f"-I{ROOT / 'third_party/utf8proc'}", str(source),
         *map(str, sorted((ROOT / "third_party/llama-jinja/jinja").glob("*.cpp"))),
         str(ROOT / "src/text/unicode.cpp"), *map(str, extra), str(obj), "-o", str(exe)],
        check=True,
    )
    return exe


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    return build_jinja_oracle(tmp_path_factory.mktemp("jinja"), HERE / "jinja_lex_oracle.cpp", "lex_oracle")


FRAGMENTS = [
    "{{ x }}", "{{- x -}}", "{%- if a == 1 -%}", "{% if a %}", "{% endif %}", "{%- endif -%}", "{% for i in items %}",
    "{% endfor %}", "{# note #}", "{#- trimmed -#}", "text", "  indented\n", "\n", "\n\n", "  \t", "hello world\n  ",
    "{{ 'a\\nb\\'c' }}", '{{ "q\\"r\\\\s" }}', "{{ 1 + 2 }}", "{{ x - 1 }}", "{{ -1 }}", "{{ (-1) }}", "{{ [-1, +2.5] }}",
    "{{ a|trim }}", "{{ a.b[0] }}", "{{ {'k': 1} }}", "{{ {'k': {'j': 2}} }}", "{{ not x }}", "{{ a ~ b }}", "{{ a * b / c % d }}",
    "{{ a <= b != c }}", "{% set x = 3.14 %}", "{{ 007 }}", "{{ 1.5.5 }}", "\r\n", "\r", "{{ x }}\n", "{{x}}{{y}}",
    "<|im_start|>", "\u00e9\u4e2d", "}}", "{{ '}}' }}", "{{ x }}  \n  {{ y }}", "  {% set a = 1 %}\n",
]
BAD = ["{{ 'unterminated", "{# open comment", "{{ 'bad \\q escape' }}", "{{ a $ b }}", "{{ 'esc at end\\", "{%", "{{", "-", "{{ -",
       "{{ @ }}", "{%- x", "{{ a ? b }}"]


def make_cases(rng: random.Random, count: int) -> list[bytes]:
    out = []
    for _ in range(count):
        parts = [rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 10))]
        if rng.random() < 0.12:
            parts.insert(rng.randint(0, len(parts)), rng.choice(BAD))
        out.append("".join(parts).encode())
    return out


def run(oracle, tmp_path, cases: list[bytes]):
    path = tmp_path / "cases.txt"
    path.write_text("\n".join((c.hex() or "-") for c in cases) + "\n")
    mine = subprocess.run([MOJO, "run", "-I", str(ROOT / "mojo"), str(HERE / "jinja_lex_probe.mojo"), str(path)],
                          capture_output=True, text=True)
    assert mine.returncode == 0, mine.stderr[-1500:]
    theirs = subprocess.run([str(oracle), str(path)], capture_output=True, text=True, check=True)
    return mine.stdout.split("\ncase "), theirs.stdout.split("\ncase ")


def test_shipped_chat_templates_lex_identically(oracle, tmp_path):
    files = sorted((ROOT / "tools/chat_templates").glob("*.jinja"))
    assert len(files) == 3
    mine, theirs = run(oracle, tmp_path, [f.read_bytes() for f in files])
    assert mine == theirs
    assert all("\nerr " not in block and not block.startswith("err") for block in mine)
    assert sum(block.count("\n") for block in mine) > 5000  # thousands of tokens compared


def test_generated_templates_lex_identically(oracle, tmp_path):
    cases = make_cases(random.Random(20261114), 3000) + [b"", b"\n", b"\r\n", b"x\n", b"{{ x }}\n\n"]
    mine, theirs = run(oracle, tmp_path, cases)
    assert len(mine) == len(theirs) == len(cases)
    bad = [i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    assert not bad, [(cases[i], mine[i][:300], theirs[i][:300]) for i in bad[:2]]
    errors = sum("\nerr " in block for block in mine)
    assert errors > 100, errors  # the mistakes are really rejected, with identical messages
