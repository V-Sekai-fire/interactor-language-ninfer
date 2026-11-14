"""Parity: Mojo tokenizer against the engine's own tokenizer.cpp.

The oracle is src/models/qwen3_5/frontend/tokenizer.cpp (with unicode.cpp and utf8proc) built
with the system C++ compiler. Fixtures are small tokenizers trained in tokenizer_fixtures.py;
set NINFER_TEST_TOKENIZER_DIR to a directory with a real Qwen3.5-pattern tokenizer.json,
tokenizer_config.json and generation_config.json to run the same cases on a real vocabulary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import tokenizer_fixtures as fx  # noqa: E402

MOJO = shutil.which("mojo")
CXX = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
pytestmark = pytest.mark.skipif(MOJO is None or CXX is None, reason="needs mojo and a C++ compiler")
PROBE = HERE / "tokenizer_probe.mojo"


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    work = tmp_path_factory.mktemp("tok")
    exe = work / "tokenizer_oracle"
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    utf8proc_o = work / "utf8proc.o"
    subprocess.run([cc, "-O1", "-c", f"-I{ROOT / 'third_party' / 'utf8proc'}",
                    str(ROOT / "third_party/utf8proc/utf8proc.c"), "-o", str(utf8proc_o)], check=True)
    subprocess.run(
        [CXX, "-std=c++20", "-O1", f"-I{ROOT / 'src'}", f"-I{ROOT / 'third_party'}",
         f"-I{ROOT / 'third_party' / 'utf8proc'}", str(HERE / "tokenizer_oracle.cpp"),
         str(ROOT / "src/models/qwen3_5/frontend/tokenizer.cpp"), str(ROOT / "src/text/unicode.cpp"),
         str(utf8proc_o), "-o", str(exe)],
        check=True,
    )
    return exe


def run_both(oracle, paths, cases_path: Path):
    mojo = subprocess.run([MOJO, "run", "-I", str(ROOT / "mojo"), str(PROBE), *map(str, paths), str(cases_path)],
                          capture_output=True, text=True)
    assert mojo.returncode == 0, mojo.stderr[-2000:]
    cxx = subprocess.run([str(oracle), *map(str, paths), str(cases_path)], capture_output=True, text=True)
    assert cxx.returncode == 0, cxx.stderr
    return mojo.stdout.splitlines(), cxx.stdout.splitlines()


def check(oracle, paths, tmp_path, lines, expect_errors=True):
    cases = tmp_path / "cases.txt"
    cases.write_text("\n".join(lines) + "\n")
    mine, theirs = run_both(oracle, paths, cases)
    assert len(mine) == len(theirs) == len(lines) + 2, (len(mine), len(theirs))
    assert mine[:2] == theirs[:2]
    bad = [i for i, (a, b) in enumerate(zip(mine[2:], theirs[2:])) if a != b]
    assert not bad, [(lines[i], mine[2 + i], theirs[2 + i]) for i in bad[:3]]
    return mine[2:]


@pytest.mark.parametrize("flavour", ["qwen_array", "qwen_string_merges", "sentencepiece"])
def test_encode_and_decode_match(oracle, tmp_path, flavour):
    if flavour == "sentencepiece":
        paths = fx.write_sentencepiece(tmp_path)
    else:
        paths = fx.write_qwen(tmp_path, "array" if flavour == "qwen_array" else "string")
    rng = random.Random(hash(flavour) & 0xFFFF)
    header = subprocess.run([str(oracle), *map(str, paths), "/dev/null"], capture_output=True, text=True).stdout.splitlines()
    vocab_size = int(header[0].split()[1])
    stops = [int(s) for s in header[1].split()[1].strip(",").split(",")]
    lines = fx.encode_cases(rng, 2500) + fx.decode_cases(rng, vocab_size, 800, stops)
    out = check(oracle, paths, tmp_path, lines)
    ok = [o for o in out if o.startswith("ok")]
    assert len(ok) > 2000
    # The cases must exercise real merges, added tokens and boundaries, not just byte fallback.
    assert any(len(o.split(" ")[1].strip(",").split(",")) > 8 for o in ok if o.split(" ")[1])
    assert any("x/" in o for o in ok), "no inexact boundary was produced"
    assert sum(o.startswith("err") for o in out) > 20


def test_known_encoding(oracle, tmp_path):
    paths = fx.write_qwen(tmp_path)
    cases = ["encode " + "<|im_start|>hello<|im_end|>".encode().hex() + " 1 99999999 - -"]
    out = check(oracle, paths, tmp_path, cases)
    ids = [int(x) for x in out[0].split(" ")[1].strip(",").split(",")]
    assert ids[0] != ids[-1] and len(ids) >= 3  # begin/end markers are single added-token ids


def mutate(paths, tmp_path, which: str, fn):
    path = paths[which == "config" and 1 or which == "generation" and 2 or 0]
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data, ensure_ascii=False))


LOADERS = {
    "bad pipeline normalizer": ("tok", lambda d: d.update(normalizer={"type": "NFKC"})),
    "bad pre tokenizer": ("tok", lambda d: d["pre_tokenizer"]["pretokenizers"][0]["pattern"].update(Regex="x")),
    "ignore merges": ("tok", lambda d: d["model"].update(ignore_merges=True)),
    "dropout": ("tok", lambda d: d["model"].update(dropout=0.5)),
    "merge outside vocab": ("tok", lambda d: d["model"]["merges"].append(["zz\u0100", "q"])),
    "duplicate merge": ("tok", lambda d: d["model"]["merges"].append(d["model"]["merges"][0])),
    "malformed merge": ("tok", lambda d: d["model"].update(merges=[5])),
    "model type": ("tok", lambda d: d["model"].update(type="Unigram")),
    "empty vocab": ("tok", lambda d: d["model"].update(vocab={})),
    "negative id": ("tok", lambda d: d["model"]["vocab"].update(bad=-1)),
    "huge id": ("tok", lambda d: d["model"]["vocab"].update(bad=1000001)),
    "duplicate vocab id": ("tok", lambda d: d["model"]["vocab"].update(bad=d["model"]["vocab"]["a"])),
    "added overlaps": ("tok", lambda d: d["added_tokens"][0].update(id=5)),
    "added duplicate id": ("tok", lambda d: d["added_tokens"][1].update(id=d["added_tokens"][0]["id"])),
    "added lstrip": ("tok", lambda d: d["added_tokens"][0].update(lstrip=True)),
    "added empty": ("tok", lambda d: d["added_tokens"][0].update(content="")),
    "added missing field": ("tok", lambda d: d["added_tokens"][0].pop("special")),
    "added duplicate content": ("tok", lambda d: d["added_tokens"][1].update(content="a")),
    "config conflict": ("config", lambda d: list(d["added_tokens_decoder"].values())[0].update(special=False)),
    "config bad key": ("config", lambda d: d["added_tokens_decoder"].update({"01": {"content": "q", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False}})),
    "config overlaps vocab": ("config", lambda d: d["added_tokens_decoder"].update({"3": {"content": "q", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": False}})),
    "config new token": ("config", lambda d: d["added_tokens_decoder"].update({"99999": {"content": "<new>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}})),
    "generation no eos": ("gen", lambda d: d.pop("eos_token_id")),
    "generation empty eos": ("gen", lambda d: d.update(eos_token_id=[])),
    "generation bad eos": ("gen", lambda d: d.update(eos_token_id="x")),
    "generation negative": ("gen", lambda d: d.update(eos_token_id=-3)),
}


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_loader_rejections_match(oracle, tmp_path, name):
    paths = fx.write_qwen(tmp_path)
    which, fn = LOADERS[name]
    index = {"tok": 0, "config": 1, "gen": 2}[which]
    data = json.loads(paths[index].read_text())
    fn(data)
    paths[index].write_text(json.dumps(data, ensure_ascii=False))
    cases = tmp_path / "cases.txt"
    cases.write_text("encode 6162 1 99999999 - -\n")
    mine, theirs = run_both(oracle, paths, cases)
    assert mine == theirs, (mine, theirs)
    if name != "config new token":
        assert mine[0].startswith("load_err"), mine
    else:
        assert mine[0].startswith("vocab")


@pytest.mark.skipif("NINFER_TEST_TOKENIZER_DIR" not in os.environ, reason="no real tokenizer directory given")
def test_real_tokenizer_matches(oracle, tmp_path):
    d = Path(os.environ["NINFER_TEST_TOKENIZER_DIR"])
    paths = [d / "tokenizer.json", d / "tokenizer_config.json", d / "generation_config.json"]
    rng = random.Random(99)
    header = subprocess.run([str(oracle), *map(str, paths), "/dev/null"], capture_output=True, text=True).stdout.splitlines()
    vocab_size = int(header[0].split()[1])
    stops = [int(s) for s in header[1].split()[1].strip(",").split(",")]
    lines = fx.encode_cases(rng, 3000) + fx.decode_cases(rng, vocab_size, 1000, stops)
    out = check(oracle, paths, tmp_path, lines)
    assert sum(o.startswith("ok") for o in out) > 2500
